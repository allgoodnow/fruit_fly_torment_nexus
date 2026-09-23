"""CUDA execution of the shared parallel phototransduction event loop.

SPDX-License-Identifier: GPL-3.0-only
See licenses/photoreceptor/NOTICE.md for the underlying reaction model.
"""
import numpy as np
from numba import cuda, float64, int64
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
def _sum_channels(values, result):
    tick = cuda.blockIdx.x
    lane = cuda.threadIdx.x
    scratch = cuda.shared.array(128, int64)
    total = 0
    for cell in range(lane, values.shape[1], 128):
        total += int(values[tick, cell])
    scratch[lane] = total
    cuda.syncthreads()
    stride = 64
    while stride > 0:
        if lane < stride:
            scratch[lane] += scratch[lane+stride]
        cuda.syncthreads()
        stride //= 2
    if lane == 0:
        result[tick] = scratch[0]


class CudaWork:
    """Disposable candidate state; callers commit only after all stages succeed."""
    def __init__(self, arrays):
        self.arrays = [cuda.to_device(a) for a in arrays]
        self.cells = len(arrays[0])

    def advance(self, photons, start_ms, initialized):
        inputs = cuda.to_device(photons)
        output = cuda.device_array((len(photons)*10, self.cells), dtype=np.uint8)
        counts = cuda.device_array(self.cells, dtype=np.int64)
        status = cuda.device_array(self.cells, dtype=np.int32)
        _advance[(self.cells+127)//128, 128](*self.arrays, inputs, start_ms, initialized, output, counts, status)
        errors = status.copy_to_host()
        if errors.any():
            raise RuntimeError(f'CUDA phototransduction rejected molecular update (status {int(errors.max())})')
        total = cuda.device_array(len(photons)*10, dtype=np.int64)
        _sum_channels[len(photons)*10, 128](output, total)
        return total.copy_to_host(), int(counts.copy_to_host().sum())

    def finish(self):
        return [a.copy_to_host() for a in self.arrays]
