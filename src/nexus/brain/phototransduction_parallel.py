"""Research cascade with per-microvillus random streams and CPU/CUDA execution.

SPDX-License-Identifier: GPL-3.0-only
Shares the existing reaction equations; introduces explicitly different seeded
streams. It is not imported by the application's live sensory pathway.
"""
import copy
import math
import sys

import numpy as np
from numba import njit
from numba.cuda.random import (
    init_xoroshiro128p_states_cpu, xoroshiro128p_dtype,
    xoroshiro128p_uniform_float64,
)

from .photoreceptor import PhotoreceptorMembrane
from .phototransduction import reaction_rates_into, calcium_update_into, apply_reaction

CHUNK_MS = 20  # Bounds scratch memory, not molecular event resolution.


@njit(cache=not getattr(sys, 'frozen', False))
def prepare(y, rates, flux, currents, reversal, when, streams, cell):
    reaction_rates_into(y, rates)
    total = 0.
    for i in range(8):
        for j in range(2):
            value = rates[i, j]
            if not math.isfinite(value) or value < 0:
                return math.inf, -1, reversal, 1
            total += value
    calcium, reversal = calcium_update_into(y, reversal, flux, currents)
    if not math.isfinite(calcium) or calcium <= 0 or not math.isfinite(reversal):
        return math.inf, -1, reversal, 2
    y[1] = calcium
    if total == 0:
        return math.inf, -1, reversal, 0
    # Uniform draws are [0,1). Reject exactly zero for an open (0,1) wait,
    # rather than clipping its tail or permitting a zero-length event.
    u = xoroshiro128p_uniform_float64(streams, cell)
    while u == 0.:
        u = xoroshiro128p_uniform_float64(streams, cell)
    due = when-math.log(u)/total
    if not math.isfinite(due) or due <= when:
        return due, -1, reversal, 3
    target = xoroshiro128p_uniform_float64(streams, cell)*total
    cumulative = 0.
    for direction in range(2):
        for index in range(8):
            cumulative += rates[index, direction]
            if target < cumulative:
                return due, direction*8+index, reversal, 0
    return due, -1, reversal, 3


@njit(cache=not getattr(sys, 'frozen', False))
def until(y, rates, flux, currents, due, reaction, reversal, boundary, streams, cell):
    events = 0
    while due <= boundary:
        if not apply_reaction(y, reaction):
            return due, reaction, reversal, events, 4
        due, reaction, reversal, status = prepare(y, rates, flux, currents, reversal, due, streams, cell)
        events += 1
        if status:
            return due, reaction, reversal, events, status
        if events > 1000000:
            return due, reaction, reversal, events, 5
    return due, reaction, reversal, events, 0


@njit(cache=not getattr(sys, 'frozen', False))
def run_micro(y, rates, flux, currents, due, reaction, reversal,
              photons, start_ms, initialized, streams, cell, output):
    events = 0
    if not initialized:
        due, reaction, reversal, status = prepare(y, rates, flux, currents, reversal, start_ms, streams, cell)
        if status:
            return due, reaction, reversal, events, status
    for frame in range(len(photons)):
        boundary = start_ms+frame
        due, reaction, reversal, n, status = until(y, rates, flux, currents, due, reaction, reversal, boundary, streams, cell)
        events += n
        if status:
            return due, reaction, reversal, events, status
        if photons[frame, cell]:
            y[3] += photons[frame, cell]
            due, reaction, reversal, status = prepare(y, rates, flux, currents, reversal, boundary, streams, cell)
            if status:
                return due, reaction, reversal, events, status
        for sample in range(10):
            now = (boundary*10+sample)/10.
            due, reaction, reversal, n, status = until(y, rates, flux, currents, due, reaction, reversal, now, streams, cell)
            events += n
            if status:
                return due, reaction, reversal, events, status
            output[frame*10+sample, cell] = int(y[0])
        due, reaction, reversal, n, status = until(y, rates, flux, currents, due, reaction, reversal, boundary+1, streams, cell)
        events += n
        if status:
            return due, reaction, reversal, events, status
    return due, reaction, reversal, events, 0


@njit(cache=not getattr(sys, 'frozen', False))
def cpu_chunk(states, due, reactions, reversals, streams, photons, start_ms, initialized):
    output = np.empty((len(photons)*10, len(states)), dtype=np.uint8)
    events = 0
    rates, flux, currents = np.empty((8, 2)), np.empty(4), np.empty(4)
    for cell in range(len(states)):
        d, r, v, n, status = run_micro(states[cell], rates, flux, currents,
            due[cell], reactions[cell], reversals[cell], photons, start_ms, initialized, streams, cell, output)
        if status:
            raise RuntimeError('Parallel phototransduction rejected a molecular update')
        due[cell], reactions[cell], reversals[cell] = d, r, v
        events += n
    return output.sum(axis=1, dtype=np.int64), events


class ParallelPhototransduction:
    """One explicit receptor with independent event streams per microvillus.

CPU and CUDA use the same per-unit xoroshiro128+ streams and equations.
Photon allocation uses a separate NumPy generator. Seeded traces intentionally
differ from the older shared-stream Phototransduction class. The membrane runs
on CPU because a single cell is slower on the GPU in our measured backend.
All state, photon allocation and event RNG updates commit atomically.
"""
    def __init__(self, *, microvilli=30000, seed=73100, backend='cpu'):
        if (isinstance(microvilli, bool) or not isinstance(microvilli, (int, np.integer))
                or not 1 <= microvilli <= 30000):
            raise ValueError('Choose 1..30000 explicit microvilli')
        if (isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**64):
            raise ValueError('Seed must be an unsigned 64-bit integer')
        if backend not in ('cpu', 'cuda'):
            raise ValueError('Choose cpu or cuda explicitly')
        self.microvilli, self.seed, self.backend = int(microvilli), int(seed), backend
        if backend == 'cuda':
            from .phototransduction_cuda import require_cuda
            require_cuda()
        self.membrane = PhotoreceptorMembrane()
        self.reset()

    def reset(self):
        self.states = np.tile([0., 1., 0., 0., 0., 0., 0., 50.], (self.microvilli, 1))
        self.due = np.full(self.microvilli, math.inf)
        self.reactions = np.full(self.microvilli, -1, dtype=np.int64)
        self.reversals = np.full(self.microvilli, .009)
        photon_seed, event_seed = np.random.SeedSequence(self.seed).spawn(2)
        self.rng = np.random.default_rng(photon_seed)
        self.streams = np.empty(self.microvilli, dtype=xoroshiro128p_dtype)
        init_xoroshiro128p_states_cpu(self.streams, event_seed.generate_state(1, dtype=np.uint64)[0], 0)
        self.membrane.reset()
        self.time_ms = self.events = self.photons = 0
        self.initialized = False

    def advance(self, absorbed_photons):
        raw = np.asarray(absorbed_photons)
        if (raw.ndim != 1 or raw.dtype.kind not in 'iuf' or len(raw) == 0
                or not np.isfinite(raw).all() or (raw < 0).any()
                or (raw > 1000000).any() or (raw != np.floor(raw)).any()):
            raise ValueError('Expected integer absorbed photon counts per ms, from 0 to 1000000')
        photons = raw.astype(np.int64)
        arrays = [a.copy() for a in (self.states, self.due, self.reactions, self.reversals, self.streams)]
        rng = copy.deepcopy(self.rng)
        if self.backend == 'cuda':
            from .phototransduction_cuda import CudaWork
            work = CudaWork(arrays)
        pieces, events = [], 0
        for start in range(0, len(photons), CHUNK_MS):
            counts = photons[start:start+CHUNK_MS]
            allocated = np.zeros((len(counts), self.microvilli), dtype=np.int32)
            for frame, count in enumerate(counts):
                if count:
                    selected = rng.integers(0, self.microvilli, size=int(count))
                    allocated[frame] = np.bincount(selected, minlength=self.microvilli)
            initialized = self.initialized or start > 0
            if self.backend == 'cuda':
                result, n = work.advance(allocated, self.time_ms+start, initialized)
            else:
                result, n = cpu_chunk(*arrays, allocated, self.time_ms+start, initialized)
            pieces.append(result)
            events += n
        if self.backend == 'cuda':
            arrays = work.finish()
        channels = np.concatenate(pieces)
        voltage = self.membrane.advance_channels(channels[:, None])
        self.states, self.due, self.reactions, self.reversals, self.streams = arrays
        self.rng = rng
        self.time_ms += len(photons)
        self.events += events
        self.photons += int(photons.sum())
        self.initialized = True
        return {'open_channels': channels, 'voltage_mv': voltage[:, 0]}
