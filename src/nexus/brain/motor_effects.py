"""Authored body response to actual escape spikes and distributed hyperactivity.

No button names, temperatures, or claimed emotional states enter this decoder.
The disruption waveform is an engineering extension, not a reconstructed VNC.
"""
import math
import numpy as np



class MotorEffects:
    def __init__(self, brain):
        self.gf = np.array([brain.lookup[int(root)] for root in brain.circuit_ids('giant_fiber')
                            if int(root) in brain.lookup], dtype=np.int32)
        self.reset()

    def reset(self):
        self.enabled = True
        self.escape_hz = 0.
        self.population_hz = 0.
        self.recruitment = 0.
        self.disruption = 0.

    def observe(self, counts, seconds, gains, directly_driven):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Motor effect interval must be positive')
        delivered = np.asarray(counts)*np.asarray(gains)
        escape = float(delivered[self.gf].mean()/seconds) if len(self.gf) == 2 else 0.
        downstream = delivered.copy()
        downstream[directly_driven] = 0
        population = float(downstream.sum()/seconds/len(downstream))
        recruitment = float(np.count_nonzero(downstream)/len(downstream))
        alpha = math.exp(-seconds/.05)
        self.escape_hz = alpha*self.escape_hz+(1-alpha)*escape
        self.population_hz = alpha*self.population_hz+(1-alpha)*population
        self.recruitment = alpha*self.recruitment+(1-alpha)*recruitment
        target = np.clip((self.population_hz-1.)/8., 0, 1)*np.clip(self.recruitment/.03, 0, 1)
        self.disruption = float(alpha*self.disruption+(1-alpha)*target)

    def output(self):
        return {'enabled': self.enabled, 'escape_hz': self.escape_hz,
                'escape': float(np.clip(self.escape_hz/150., 0, 1)) if self.enabled else 0.,
                'disruption': self.disruption if self.enabled else 0.,
                'downstream_hz_per_neuron': self.population_hz, 'recruitment_fraction': self.recruitment,
                'decoder': 'GF startle / distributed-activity motor disruption v1',
                'motor_mapping': 'authored; no VNC or validated convulsion model'}
