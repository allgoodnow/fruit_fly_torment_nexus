"""Authored body response to descending/escape spikes and distributed hyperactivity.

No button names, temperatures, or claimed emotional states enter this decoder.
The disruption waveform is an engineering extension, not a reconstructed VNC.
"""
import math
import numpy as np



class MotorEffects:
    def __init__(self, brain):
        self.gf = np.array([brain.lookup[int(root)] for root in brain.circuit_ids('giant_fiber')
                            if int(root) in brain.lookup], dtype=np.int32)
        # Official MaleCNS v1.0 type=MDN, two cells per side. Source and
        # MDN->LBL40 edges are audited by audit_descending_pathway.py.
        roots = [10763, 11288, 11332, 12348] if brain.graph.snapshot == 'male-cns:v1.0' else []
        present = [root in brain.lookup for root in roots]
        if any(present) and not all(present):
            raise ValueError('Incomplete MaleCNS MDN readout')
        self.mdn = np.array([brain.lookup[root] for root in roots] if all(present) else [], dtype=np.int32)
        self.reset()

    def reset(self):
        self.enabled = True
        self.descending_enabled = True
        self.escape_hz = 0.
        self.mdn_hz = 0.
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
        # Command-level readout, not a reconstruction of leg premotor dynamics.
        # Smoothing and rate-to-command gain remain engineering assumptions.
        mdn = float(delivered[self.mdn].mean()/seconds) if len(self.mdn) else 0.
        mdn_alpha = math.exp(-seconds/.08)
        self.mdn_hz = mdn_alpha*self.mdn_hz+(1-mdn_alpha)*mdn
        self.population_hz = alpha*self.population_hz+(1-alpha)*population
        self.recruitment = alpha*self.recruitment+(1-alpha)*recruitment
        target = np.clip((self.population_hz-1.)/8., 0, 1)*np.clip(self.recruitment/.03, 0, 1)
        self.disruption = float(alpha*self.disruption+(1-alpha)*target)

    def output(self):
        return {'enabled': self.enabled, 'escape_hz': self.escape_hz,
                'mdn_hz': self.mdn_hz, 'mdn_available': bool(len(self.mdn)),
                'descending_enabled': self.descending_enabled,
                'retreat': float(np.clip((self.mdn_hz-5.)/55., 0, 1)) if self.enabled and self.descending_enabled else 0.,
                'escape': float(np.clip(self.escape_hz/150., 0, 1)) if self.enabled else 0.,
                'disruption': self.disruption if self.enabled else 0.,
                'downstream_hz_per_neuron': self.population_hz, 'recruitment_fraction': self.recruitment,
                'decoder': 'MDN retreat v1 / GF startle / distributed disruption v1',
                'motor_mapping': 'authored; no reconstructed VNC-to-muscle or validated pain/convulsion model'}
