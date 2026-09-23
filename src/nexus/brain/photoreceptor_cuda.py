"""Optional research CUDA membrane backend; not imported by the live application.

SPDX-License-Identifier: GPL-3.0-only
Uses the shared Song/Juusola BG1 equations, not a separate GPU approximation.
Source attribution: licenses/photoreceptor/NOTICE.md.
"""
import math

import numpy as np
from numba import cuda, float64

from .photoreceptor import (
    DT_MS, PhotoreceptorMembrane, derivative_values, trp_current_na,
)


@cuda.jit(fastmath=False)
def _integrate_channels(initial, inputs, final, voltage, status):
    cell = cuda.grid(1)
    if cell >= initial.shape[0]:
        return
    y = cuda.local.array(9, float64)
    stage = cuda.local.array(9, float64)
    for j in range(9):
        y[j] = initial[cell, j]
    status[cell] = 0
    for tick in range(inputs.shape[0]):
        channels = inputs[tick, cell]
        k1 = derivative_values(y, trp_current_na(channels, y[0]))
        for j in range(9):
            stage[j] = y[j]+.5*DT_MS*k1[j]
        k2 = derivative_values(stage, trp_current_na(channels, stage[0]))
        for j in range(9):
            stage[j] = y[j]+.5*DT_MS*k2[j]
        k3 = derivative_values(stage, trp_current_na(channels, stage[0]))
        for j in range(9):
            stage[j] = y[j]+DT_MS*k3[j]
        k4 = derivative_values(stage, trp_current_na(channels, stage[0]))
        for j in range(9):
            y[j] = y[j]+DT_MS/6.*(k1[j]+2.*k2[j]+2.*k3[j]+k4[j])
            if (not math.isfinite(y[j]) or (1 <= j <= 5 and not 0 <= y[j] <= 1)
                    or (j >= 6 and y[j] <= 0)):
                status[cell] = tick+1
                return
        voltage[tick, cell] = y[0]
    for j in range(9):
        final[cell, j] = y[j]


class CudaPhotoreceptorMembrane(PhotoreceptorMembrane):
    """Host-facing float64 channel-to-voltage integration across independent cells.

Input validation, reset and public state layout match the CPU component.
Results are committed only after device completion and physical-range checks.
No photons, stochastic reactions or synaptic release are calculated here.
Current-driven advance() retains the CPU implementation; advance_channels()
uses CUDA. Each call transfers its input and output; transfer time is part of
the public operation, not hidden from benchmarks.
"""
    def __init__(self, cells=1):
        super().__init__(cells)
        if not cuda.is_available():
            raise RuntimeError('CUDA is unavailable; use PhotoreceptorMembrane for CPU execution')

    def _advance_validated(self, currents, *, channel_input):
        if not channel_input:
            return super()._advance_validated(currents, channel_input=False)
        initial = cuda.to_device(np.ascontiguousarray(self.state))
        inputs = cuda.to_device(np.ascontiguousarray(currents))
        candidate_device = cuda.device_array(self.state.shape, dtype=np.float64)
        voltage_device = cuda.device_array(currents.shape, dtype=np.float64)
        status_device = cuda.device_array(len(self.state), dtype=np.int32)
        threads = 128
        _integrate_channels[(len(self.state)+threads-1)//threads, threads](
            initial, inputs, candidate_device, voltage_device, status_device)
        # Blocking copies ensure any device/launch failure precedes commit.
        status = status_device.copy_to_host()
        if status.any():
            raise RuntimeError('Photoreceptor integration left the valid state range on CUDA')
        candidate = candidate_device.copy_to_host()
        voltage = voltage_device.copy_to_host()
        if not np.isfinite(candidate).all() or not np.isfinite(voltage).all():
            raise RuntimeError('CUDA returned nonfinite photoreceptor output')
        self.state[:] = candidate
        self.step += len(currents)
        return voltage
