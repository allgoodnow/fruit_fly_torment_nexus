"""Persistent fixed-step LIF implementation of the pinned Shiu model.

Units here are mV and milliseconds. Numba supplies machine code without a system
C++ compiler. Brian2 remains the independent reference, not a runtime dependency.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import json
import hashlib
import sys

import numpy as np
from numba import njit

DT_MS = 0.1
DELAY_STEPS = 18
REFRACTORY_STEPS = 22


@njit(cache=not getattr(sys, "frozen", False), fastmath=False)
def _advance(v, g, last_spike, refractory, enabled, offsets, posts, weights,
             output_gain, queue, queue_size, counts, step, inputs, input_events,
             trace_ids):
    steps = len(input_events)
    trace = np.empty((steps, len(trace_ids)), dtype=np.float64)
    # Output bounded by one chunk, not total lifetime.
    capacity = min(20000, steps * len(v))
    spike_ids = np.empty(capacity, dtype=np.int32)
    spike_steps = np.empty(capacity, dtype=np.int64)
    recorded = 0
    em = math.exp(-DT_MS / 20.0)
    eg = math.exp(-DT_MS / 5.0)
    coupling = (em - eg) / 3.0
    for local in range(steps):
        tick = step + local
        slot = tick % len(queue_size)
        for i in range(len(v)):
            enabled[i] = tick - last_spike[i] >= refractory[i]
            if enabled[i]:
                v[i] = -52.0 + (v[i] + 52.0) * em + g[i] * coupling
                g[i] *= eg
        # Deliver spikes after the threshold pass, exactly as in Brian's schedule.
        future = (tick + DELAY_STEPS) % len(queue_size)
        queue_size[future] = 0
        for i in range(len(v)):
            if enabled[i] and v[i] > -45.0:
                enabled[i] = False
                last_spike[i] = tick
                counts[i] += 1
                queue[future, queue_size[future]] = i
                queue_size[future] += 1
                spike_ids[recorded % capacity] = i
                spike_steps[recorded % capacity] = tick
                recorded += 1
        for j in range(queue_size[slot]):
            pre = queue[slot, j]
            gain = output_gain[pre]
            if gain != 0:
                for edge in range(offsets[pre], offsets[pre + 1]):
                    post = posts[edge]
                    if enabled[post]:
                        g[post] += weights[edge] * gain
        queue_size[slot] = 0
        # N=1 PoissonInput is Bernoulli(rate*dt); it acts in the synapses slot.
        for j in range(len(inputs)):
            i = inputs[j]
            if enabled[i] and input_events[local, j]:
                v[i] += 0.275 * 250.0
        for j in range(queue_size[future]):
            i = queue[future, j]
            v[i] = -52.0
            g[i] = 0.0
        for j in range(len(trace_ids)):
            trace[local, j] = v[trace_ids[j]]
    size = min(recorded, capacity)
    chronological_ids = np.empty(size, dtype=np.int32)
    chronological_steps = np.empty(size, dtype=np.int64)
    for j in range(size):
        slot = (max(0, recorded-capacity) + j) % capacity
        chronological_ids[j] = spike_ids[slot]
        chronological_steps[j] = spike_steps[slot]
    return chronological_ids, chronological_steps, trace, recorded-size


@dataclass
class Connectome:
    ids: np.ndarray
    offsets: np.ndarray
    posts: np.ndarray
    weights: np.ndarray
    snapshot: str = "synthetic-test"

    @classmethod
    def from_edges(cls, ids, pre, post, weights, snapshot="synthetic-test"):
        ids = np.asarray(ids, dtype=np.int64)
        pre, post = np.asarray(pre, dtype=np.int32), np.asarray(post, dtype=np.int32)
        weights = np.asarray(weights, dtype=np.float64)
        if len(ids) == 0 or len(np.unique(ids)) != len(ids):
            raise ValueError("Neuron IDs must be unique and nonempty")
        if not (len(pre) == len(post) == len(weights)) or not np.isfinite(weights).all():
            raise ValueError("Invalid edge arrays")
        if len(pre) and (min(pre.min(), post.min()) < 0 or max(pre.max(), post.max()) >= len(ids)):
            raise ValueError("Edge index out of bounds")
        order = np.argsort(pre, kind="stable")
        offsets = np.concatenate(([0], np.cumsum(np.bincount(pre, minlength=len(ids))))).astype(np.int64)
        return cls(ids, offsets, post[order], weights[order], snapshot)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("format") != "nexus-connectome-1":
            raise ValueError("Unsupported brain pack")
        for name in ("ids.npy", "offsets.npy", "posts.npy", "weights.npy"):
            with (directory / name).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != manifest["files"].get(name):
                    raise ValueError(f"Brain-pack checksum mismatch: {name}")
        arrays = [np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                  for name in ("ids", "offsets", "posts", "weights")]
        ids, offsets, posts, weights = arrays
        if ids.dtype != np.int64 or offsets.dtype != np.int64 or posts.dtype != np.int32 or weights.dtype != np.float64:
            raise ValueError("Unexpected brain-pack array types")
        if any(a.ndim != 1 for a in arrays) or len(offsets) != len(ids)+1:
            raise ValueError("Invalid brain-pack dimensions")
        if offsets[0] != 0 or offsets[-1] != len(posts) or len(posts) != len(weights) or (np.diff(offsets) < 0).any():
            raise ValueError("Invalid adjacency offsets")
        if not len(ids) or len(np.unique(ids)) != len(ids) or not np.isfinite(weights).all():
            raise ValueError("Invalid neuron IDs or weights")
        if len(posts) and (posts.min() < 0 or posts.max() >= len(ids)):
            raise ValueError("Invalid postsynaptic index")
        return cls(*arrays, snapshot=manifest["snapshot"])


class Brain:
    def __init__(self, connectome: Connectome, *, seed=73100, history_limit=20000):
        self.graph = connectome
        self.lookup = {int(fid): i for i, fid in enumerate(connectome.ids)}
        n = len(connectome.ids)
        self.v = np.empty(n)
        self.g = np.empty(n)
        self.last_spike = np.empty(n, dtype=np.int64)
        self.refractory = np.empty(n, dtype=np.int32)
        self.enabled = np.empty(n, dtype=np.bool_)
        self.output_gain = np.ones(n)
        self.queue = np.zeros((DELAY_STEPS + 1, n), dtype=np.int32)
        self.queue_size = np.zeros(DELAY_STEPS + 1, dtype=np.int32)
        self.counts = np.zeros(n, dtype=np.int64)
        self.history = deque(maxlen=history_limit)
        self.events = deque(maxlen=2000)
        self.seed = seed
        self.inputs = np.array([], dtype=np.int32)
        self.rates = np.array([], dtype=np.float64)
        self.reset()

    @property
    def time(self):
        return self.step / 10000

    def resolve(self, ids):
        try:
            if any(isinstance(fid, (float, np.floating, bool)) for fid in ids):
                raise ValueError("Use integer or decimal-string neuron IDs, not floating point")
            result = np.array([self.lookup[int(fid)] for fid in ids], dtype=np.int32)
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError(f"Unknown or invalid neuron ID: {error}") from error
        if len(np.unique(result)) != len(result):
            raise ValueError("Duplicate targets")
        return result

    def stimulate(self, ids, rate_hz):
        targets = self.resolve(ids)
        rate = float(rate_hz)
        if not 0 < len(targets) <= 256 or not math.isfinite(rate) or not 0 < rate <= 1000:
            raise ValueError("Choose targets and a finite input rate in (0, 1000] Hz")
        self.manual_inputs = targets
        self.manual_rate = rate
        self._refresh_inputs()
        self.events.append({"kind": "stimulate", "time": self.time, "ids": [str(self.graph.ids[i]) for i in targets], "rate_hz": rate})

    def release(self):
        self.manual_inputs = np.array([], dtype=np.int32)
        self.manual_rate = 0.
        self._refresh_inputs()
        self.output_gain.fill(1)
        self.events.append({"kind": "release", "time": self.time})

    def set_sensory_input(self, ids, rate_hz):
        """Environmental input channel; overlapping manual/sensory rates use max."""
        targets = self.resolve(ids)
        rate = float(rate_hz)
        if len(targets) > 256 or not math.isfinite(rate) or not 0 <= rate <= 1000:
            raise ValueError('Sensory rate must be finite and in [0, 1000] Hz')
        if not len(targets) or rate == 0:
            targets, rate = np.array([], dtype=np.int32), 0.
        if np.array_equal(targets, self.sensory_inputs) and rate == self.sensory_rate:
            return
        self.sensory_inputs, self.sensory_rate = targets, rate
        self._refresh_inputs()
        self.events.append({'kind': 'sensory_input' if len(targets) else 'sensory_release',
                            'time': self.time, 'ids': [str(self.graph.ids[i]) for i in targets], 'rate_hz': rate})

    def _refresh_inputs(self):
        # Preserve manual target order so independent-mode seeded trials retain
        # exactly the original random draw order. Duplicate targets never receive
        # two independent pulses in the same tick.
        rates = {int(i): self.manual_rate for i in self.manual_inputs}
        for i in self.sensory_inputs:
            rates[int(i)] = max(rates.get(int(i), 0.), self.sensory_rate)
        self.refractory[self.inputs] = REFRACTORY_STEPS
        self.inputs = np.array(list(rates), dtype=np.int32)
        self.rates = np.array(list(rates.values()), dtype=np.float64)
        self.refractory[self.inputs] = 0

    def silence(self, ids):
        targets = self.resolve(ids)
        self.output_gain[targets] = 0
        self.events.append({"kind": "silence_outgoing", "time": self.time, "ids": [str(self.graph.ids[i]) for i in targets]})

    def reset(self):
        self.step = 0
        self.v.fill(-52)
        self.g.fill(0)
        self.last_spike.fill(-100000000)
        self.refractory.fill(REFRACTORY_STEPS)
        self.enabled.fill(True)
        self.output_gain.fill(1)
        self.queue.fill(0)
        self.queue_size.fill(0)
        self.counts.fill(0)
        self.inputs = np.array([], dtype=np.int32)
        self.rates = np.array([], dtype=np.float64)
        self.rng = np.random.default_rng(self.seed)
        self.manual_inputs = np.array([], dtype=np.int32)
        self.sensory_inputs = np.array([], dtype=np.int32)
        self.manual_rate = self.sensory_rate = 0.
        self.history.clear()
        self.events.clear()

    def advance(self, seconds, *, input_events=None, trace_ids=()):
        if not math.isfinite(float(seconds)):
            raise ValueError("Advance must be finite")
        steps = round(float(seconds) * 1000 / DT_MS)
        if not math.isfinite(float(seconds)) or steps < 1 or steps > 10000 or not math.isclose(steps*DT_MS/1000, seconds, abs_tol=1e-12):
            raise ValueError("Advance must be a multiple of 0.1 ms, between 0.1 ms and one second")
        traces = self.resolve(trace_ids)
        if len(traces) > 32:
            raise ValueError("At most 32 voltage traces per chunk")
        if input_events is None:
            events = self.rng.random((steps, len(self.inputs))) < self.rates * DT_MS / 1000
        else:
            events = np.asarray(input_events, dtype=np.bool_)
            if events.shape != (steps, len(self.inputs)):
                raise ValueError("Input replay shape does not match step count and target count")
        result = _advance(self.v, self.g, self.last_spike, self.refractory, self.enabled,
                          self.graph.offsets, self.graph.posts, self.graph.weights,
                          self.output_gain, self.queue, self.queue_size, self.counts,
                          self.step, self.inputs, events, traces)
        self.step += steps
        if not np.isfinite(self.v).all() or not np.isfinite(self.g).all():
            raise RuntimeError("Non-finite neural state")
        self.history.extend(zip(result[1].tolist(), result[0].tolist()))
        return {"indices": result[0], "steps": result[1], "voltage_mv": result[2], "unrecorded_spikes": result[3]}
