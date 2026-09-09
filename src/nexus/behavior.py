"""Seeded, authored ground behavior; separate from connectome dynamics."""
import math
import numpy as np


class GroundBehavior:
    def __init__(self, enabled=False, seed=73100):
        self.enabled, self.seed = enabled, seed
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.phase = 'exploring'
        self.remaining = 1.2
        self.elapsed = 0.
        self.turn = .12
        self.interrupted = False
        self.cycles = 0

    def advance(self, seconds, interrupted):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Behavior interval must be positive and finite')
        self.interrupted = bool(interrupted) if self.enabled else False
        if not self.enabled or self.interrupted:
            return
        self.elapsed += seconds
        self.remaining -= seconds
        while self.remaining <= 1e-10:
            if self.phase == 'exploring':
                self.phase = 'turning'
                self.turn = float(self.rng.choice([-1, 1]) * self.rng.uniform(.22, .4))
                duration = self.rng.uniform(.3, .6)
            elif self.phase == 'turning':
                self.phase = 'resting'
                self.turn = 0.
                duration = self.rng.uniform(.5, .9)
            else:
                self.phase = 'exploring'
                self.turn = float(self.rng.uniform(-.16, .16))
                duration = self.rng.uniform(1.2, 2.4)
                self.cycles += 1
            self.remaining += float(duration)

    def output(self, baseline):
        active = self.enabled and not self.interrupted
        resting = active and self.phase == 'resting'
        return {'enabled': self.enabled,
                'state': ('interrupted' if self.interrupted else self.phase) if self.enabled else 'constant walking',
                'pending_state': self.phase, 'behavior_time': self.elapsed,
                'phase_remaining_s': self.remaining, 'cycles': self.cycles,
                'drive': (0. if resting else baseline * (.7 if active and self.phase == 'turning' else 1.)),
                'turn': baseline*self.turn if active and not resting else 0., 'resting': resting,
                'controller': 'authored explore / turn / rest; not connectome-generated behavior'}
