"""Explicit, uncalibrated voltage-to-release curves for receptor replay.

Rates are equivalent events in the connectome's weight units. They are neither
photoreceptor spikes nor measured vesicles or histamine concentrations.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ReleaseCurve:
    voltage_mv: np.ndarray
    release_equivalent_hz: np.ndarray
    reference_mv: float
    description: str

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get('format') != 'nexus-photoreceptor-release-1' or data.get('parameters_fitted') is not False:
            raise ValueError('Receptor replay requires an explicitly unfitted release experiment')
        voltage = np.asarray(data['voltage_mv'], dtype=np.float64)
        rate = np.asarray(data['release_equivalent_hz'], dtype=np.float64)
        reference = float(data['reference_mv'])
        description = data.get('description', '')
        if (voltage.ndim != 1 or len(voltage) < 2 or rate.shape != voltage.shape
                or not np.isfinite(voltage).all() or not np.isfinite(rate).all()
                or (np.diff(voltage) <= 0).any() or (np.diff(rate) < 0).any()
                or (rate < 0).any() or not np.isfinite(reference)
                or not voltage[0] <= reference <= voltage[-1]
                or not isinstance(description, str) or not description.strip()):
            raise ValueError('Supply increasing voltages, nonnegative monotonic rates and a description')
        voltage.setflags(write=False)
        rate.setflags(write=False)
        return cls(voltage, rate, reference, description)

    def evaluate(self, voltage_mv):
        voltage = np.asarray(voltage_mv, dtype=np.float64)
        if (not np.isfinite(voltage).all() or (voltage < self.voltage_mv[0]).any()
                or (voltage > self.voltage_mv[-1]).any()):
            raise ValueError('Receptor voltage lies outside the declared release curve')
        return np.interp(voltage, self.voltage_mv, self.release_equivalent_hz)

    def snapshot(self):
        return {'format': 'nexus-photoreceptor-release-1', 'parameters_fitted': False,
                'voltage_mv': self.voltage_mv.tolist(),
                'release_equivalent_hz': self.release_equivalent_hz.tolist(),
                'reference_mv': self.reference_mv, 'description': self.description,
                'interpolation': 'piecewise linear; no extrapolation',
                'units': 'equivalent connectome-weight events/s; not spikes or vesicles'}
