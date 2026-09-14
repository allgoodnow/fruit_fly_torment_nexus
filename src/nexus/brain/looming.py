"""Virtual approach geometry and explicitly unfitted sensory input envelopes.

Ache et al., doi:10.1016/j.cub.2019.01.079, motivates size-sensitive LPLC2
and expansion-speed-sensitive LC4 channels. These envelopes are engineering
choices, NOT the paper's fitted giant-fiber voltage model or a retina model.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LoomingInput:
    origin: int
    duration: int

    @property
    def end(self):
        return self.origin + self.duration

    def sample(self, steps):
        elapsed = np.asarray(steps) - self.origin
        fraction = np.clip(elapsed / self.duration, 0., 1.)
        # Radius is one unit; distance decreases linearly. Full visual angle
        # grows from 5 to 120 degrees, stopping before a geometric collision.
        start, finish = 1 / np.tan(np.deg2rad([5., 120.]) / 2)
        distance = start + (finish - start) * fraction
        angle = np.rad2deg(2 * np.arctan(1 / distance))
        speed = np.rad2deg(2 * (start - finish) / (self.duration * .0001)
                          / (1 + distance * distance))
        active = (elapsed >= 0) & (elapsed < self.duration)
        # Chosen size tuning (45 +/- 20 degrees) and speed scaling (200 Hz at
        # 1000 deg/s) are not measured input firing rates. Both cap at 200 Hz.
        size_rate = np.where(active, 200 * np.exp(-.5 * ((angle - 45) / 20) ** 2), 0.)
        speed_rate = np.where(active, np.minimum(200., .2 * speed), 0.)
        return angle, speed, size_rate, speed_rate

    def snapshot(self, step, velocity_available):
        angle, speed, size_rate, speed_rate = self.sample(step)
        return {'mapping': 'virtual-approach-v1', 'origin_ms': self.origin / 10,
                'duration_ms': self.duration / 10, 'elapsed_ms': (step - self.origin) / 10,
                'angle_degrees': float(angle), 'expansion_degrees_per_second': float(speed),
                'lplc2_rate_hz': float(size_rate),
                'lc4_rate_hz': float(speed_rate) if velocity_available else None,
                'velocity_pathway_available': velocity_available}
