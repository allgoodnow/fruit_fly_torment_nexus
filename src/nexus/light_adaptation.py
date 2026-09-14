"""Unfitted divisive light adaptation on the visual exposure clock."""
import math

import numpy as np

TAU_MS = 250.
STRENGTH = 3.


class LightAdaptation:
    def __init__(self):
        self.reset()

    def reset(self):
        self.background = {}
        self.held_light = {}
        self.elapsed_ticks = 0
        self.exposure_ticks = 0

    def current_background(self):
        decay = math.exp(-self.elapsed_ticks * .1 / TAU_MS)
        return {key: value + (self.background[key] - value) * decay
                for key, value in self.held_light.items()}

    def sample(self, values):
        candidate = {key: np.asarray(value, dtype=np.float64).copy() for key, value in values.items()}
        if (not candidate or any(not value.size or not np.isfinite(value).all()
                                  or ((value < 0) | (value > 1)).any() for value in candidate.values())):
            raise ValueError('Adaptation expects finite normalized light values')
        if self.held_light and (candidate.keys() != self.held_light.keys()
                               or any(value.shape != self.held_light[key].shape for key,value in candidate.items())):
            raise ValueError('Reset adaptation before changing visual channels')
        background = self.current_background() if self.held_light else {
            key: np.zeros_like(value) for key,value in candidate.items()}
        result = {key: value / (1. + STRENGTH * background[key]) for key,value in candidate.items()}
        self.background, self.held_light = background, candidate
        self.elapsed_ticks = 0
        return result

    def advance(self, ticks):
        if not isinstance(ticks, (int, np.integer)) or ticks < 0:
            raise ValueError('Visual exposure must advance by nonnegative integer ticks')
        if self.held_light:
            self.elapsed_ticks += int(ticks)
            self.exposure_ticks += int(ticks)

    def snapshot(self):
        background = self.current_background()
        return {'model': 'divisive-light-adaptation-v1', 'tau_ms': TAU_MS, 'strength': STRENGTH,
                'exposure_ms': self.exposure_ticks / 10,
                'mean_background': {key: float(value.mean()) for key,value in background.items()},
                'background_ranges': {key: [float(value.min()), float(value.max())]
                                      for key,value in background.items()},
                'parameters_fitted': False}
