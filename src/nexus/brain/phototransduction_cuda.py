"""CUDA execution of the shared parallel phototransduction event loop.

SPDX-License-Identifier: GPL-3.0-only
See licenses/photoreceptor/NOTICE.md for the underlying reaction model.
"""
import numpy as np
from numba import cuda, float64, int32, int64
from .phototransduction_parallel import run_micro


def require_cuda():
    if not cuda.is_available() or cuda.config.ENABLE_CUDASIM:
        raise RuntimeError('An accessible physical CUDA device is required')


@cuda.jit(fastmath=False)
def _advance(states, due, reactions, reversals, streams, photons, start_ms, initialized, output, counts, status):
    cell = cuda.grid(1)
    if cell >= len(states):
        return
    y = cuda.local.array(8, float64)
    rates = cuda.local.array((8, 2), float64)
    flux = cuda.local.array(4, float64)
    currents = cuda.local.array(4, float64)
    for j in range(8):
        y[j] = states[cell, j]
    d, r, v, n, error = run_micro(y, rates, flux, currents, due[cell], reactions[cell], reversals[cell],
        photons, start_ms, initialized, streams, cell, output)
    status[cell] = error
    counts[cell] = n
    if error:
        return
    due[cell], reactions[cell], reversals[cell] = d, r, v
    for j in range(8):
        states[cell, j] = y[j]


@cuda.jit
def _sum_channels(values, result, microvilli):
    tick = cuda.blockIdx.x
    receptor = cuda.blockIdx.y
    lane = cuda.threadIdx.x
    scratch = cuda.shared.array(128, int64)
    total = 0
    for micro in range(lane, microvilli, 128):
        total += int(values[tick, receptor*microvilli+micro])
    scratch[lane] = total
    cuda.syncthreads()
    stride = 64
    while stride > 0:
        if lane < stride:
            scratch[lane] += scratch[lane+stride]
        cuda.syncthreads()
        stride //= 2
    if lane == 0:
        result[tick, receptor] = scratch[0]


@cuda.jit
def _summarize_updates(counts, status, events, errors, microvilli):
    """Exact integer totals and maximum error code, one block per receptor."""
    receptor = cuda.blockIdx.x
    lane = cuda.threadIdx.x
    event_scratch = cuda.shared.array(128, int64)
    error_scratch = cuda.shared.array(128, int32)
    total, error = 0, 0
    for micro in range(lane, microvilli, 128):
        cell = receptor*microvilli+micro
        total += counts[cell]
        if status[cell] > error:
            error = status[cell]
    event_scratch[lane], error_scratch[lane] = total, error
    cuda.syncthreads()
    stride = 64
    while stride > 0:
        if lane < stride:
            event_scratch[lane] += event_scratch[lane+stride]
            if error_scratch[lane+stride] > error_scratch[lane]:
                error_scratch[lane] = error_scratch[lane+stride]
        cuda.syncthreads()
        stride //= 2
    if lane == 0:
        events[receptor], errors[receptor] = event_scratch[0], error_scratch[0]


class CudaWork:
    """Disposable candidate state; callers commit only after all stages succeed."""
    def __init__(self, arrays, *, microvilli=None):
        self.arrays = [cuda.to_device(a) for a in arrays]
        self.cells = len(arrays[0])
        self.microvilli = self.cells if microvilli is None else microvilli
        self.receptors = self.cells // self.microvilli
        self.counts = cuda.device_array(self.cells, dtype=np.int64)
        self.status = cuda.device_array(self.cells, dtype=np.int32)
        self.event_totals = cuda.device_array(self.receptors, dtype=np.int64)
        self.error_codes = cuda.device_array(self.receptors, dtype=np.int32)
        self.capacity_ms = 0

    def advance(self, photons, start_ms, initialized):
        channels, events = self.advance_receptors(photons, start_ms, initialized)
        return channels[:, 0], int(events.sum())

    def advance_receptors(self, photons, start_ms, initialized):
        if len(photons) > self.capacity_ms:
            self.inputs = cuda.device_array(photons.shape, dtype=photons.dtype)
            self.output = cuda.device_array((len(photons)*10, self.cells), dtype=np.uint8)
            self.total = cuda.device_array((len(photons)*10, self.receptors), dtype=np.int64)
            self.capacity_ms = len(photons)
        inputs = self.inputs[:len(photons)]
        output = self.output[:len(photons)*10]
        total = self.total[:len(photons)*10]
        inputs.copy_to_device(photons)
        _advance[(self.cells+127)//128, 128](*self.arrays, inputs, start_ms, initialized,
                                          output, self.counts, self.status)
        _summarize_updates[self.receptors, 128](self.counts, self.status, self.event_totals,
                                              self.error_codes, self.microvilli)
        errors = self.error_codes.copy_to_host()
        if errors.any():
            raise RuntimeError(f'CUDA phototransduction rejected molecular update (status {int(errors.max())})')
        _sum_channels[(len(photons)*10, self.receptors), 128](output, total, self.microvilli)
        events = self.event_totals.copy_to_host()
        return total.copy_to_host(), events

    def finish(self):
        return [a.copy_to_host() for a in self.arrays]
