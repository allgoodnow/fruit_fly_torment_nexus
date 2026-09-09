"""Experimental DNa02 decoder; calibrated engineering gains, not a VNC model."""
import math
import numpy as np

# FlyWire annotations v1.1.0, v630 root_id / hemibrain_type / side.
DNA02_LEFT = '720575940629327659'
DNA02_RIGHT = '720575940604737708'
DNA02 = [DNA02_LEFT, DNA02_RIGHT]


class SteeringDecoder:
    def __init__(self, brain):
        self.indices = brain.resolve(DNA02)
        self.reset()

    def reset(self):
        self.rates = np.zeros(2)
        self.enabled = True

    def observe(self, counts, seconds, gains):
        if seconds <= 0 or not math.isfinite(seconds):
            raise ValueError('Decoder interval must be positive and finite')
        # 100 ms low-pass filter on actual spike counts delivered to the decoder.
        alpha = math.exp(-seconds/.1)
        self.rates = alpha*self.rates + (1-alpha)*np.asarray(counts)*np.asarray(gains)/seconds

    def output(self, baseline):
        # DNa02 activation attenuates ipsilateral strides. The magnitude/scale
        # below is an explicit controller approximation, not a biological fit.
        attenuation = .7*np.clip(self.rates/200, 0, 1) if self.enabled else np.zeros(2)
        left, right = baseline*(1-attenuation)
        return {'drive': float((left+right)/2), 'turn': float((left-right)/2),
                'left_drive': float(left), 'right_drive': float(right),
                'left_hz': float(self.rates[0]), 'right_hz': float(self.rates[1]),
                'enabled': self.enabled, 'decoder': 'DNa02 ipsilateral attenuation v1'}
