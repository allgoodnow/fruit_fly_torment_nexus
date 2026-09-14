"""Experimental forward drive from delivered BDN2 spikes; no baseline excitation."""
import math
import numpy as np


class WalkingDecoder:
    def __init__(self, brain):
        ids = []
        if brain.graph.snapshot == 'male-cns:v1.0' and brain.graph.circuits:
            ids = brain.graph.circuits['readouts'].get('forward_walking', [])
            if 'forward_walking' in brain.graph.circuits['readouts'] and len(ids) != 2:
                raise ValueError('Forward walking requires a complete bilateral BDN2 readout')
        self.indices = brain.resolve(ids)
        self.ids = list(ids)
        self.reset()

    def reset(self):
        self.rates = np.zeros(len(self.indices))
        self.enabled = True

    def observe(self, counts, seconds, gains):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Decoder interval must be positive and finite')
        alpha = math.exp(-seconds / .1)
        self.rates = alpha * self.rates + (1 - alpha) * np.asarray(counts) * np.asarray(gains) / seconds

    def output(self, gain=1.):
        rate = float(self.rates.mean()) if len(self.rates) else 0.
        # The filter and 100 Hz -> unit-drive scaling are engineering choices.
        drive = float(np.clip(rate / 100 * gain, 0., 1.3)) if self.enabled else 0.
        return {'available': len(self.indices) == 2, 'enabled': self.enabled,
                'ids': list(self.ids),
                'rate_hz': rate, 'cell_rates_hz': self.rates.tolist(), 'drive': drive,
                'decoder': 'BDN2 forward drive v1; unfitted gain; engineered leg coordination'}
