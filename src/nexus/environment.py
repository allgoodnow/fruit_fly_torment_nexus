"""A contact-driven food patch; geometry-to-taste mapping is an authored proxy."""
from collections import deque
import math

from .brain.targets import SUGAR


class FoodEnvironment:
    def __init__(self, *, center=(5., 0.), radius=2.5, present=True, enabled=True, rate_hz=200):
        self.center = list(center)
        self.radius = radius
        self.present, self.enabled, self.rate_hz = present, enabled, rate_hz
        self.events = deque(maxlen=500)
        self.reset()

    def reset(self):
        self.contact = self.active = False
        self.contacts = []
        self.active_seconds = 0.
        self.events.clear()

    def configure(self, value, time):
        if not isinstance(value, dict) or set(value)-{'center_mm', 'radius_mm', 'present', 'enabled', 'rate_hz'}:
            raise ValueError('Invalid food settings')
        center = list(value.get('center_mm', self.center))
        radius = float(value.get('radius_mm', self.radius))
        rate = float(value.get('rate_hz', self.rate_hz))
        if len(center) != 2 or not all(math.isfinite(float(x)) and abs(float(x)) <= 1000 for x in center):
            raise ValueError('Food center must have two finite coordinates within ±1000 mm')
        if not math.isfinite(radius) or not .25 <= radius <= 20:
            raise ValueError('Food radius must be in [0.25, 20] mm')
        if not math.isfinite(rate) or not 1 <= rate <= 1000:
            raise ValueError('Taste input rate must be in [1, 1000] Hz')
        for key in ('present', 'enabled'):
            if key in value and not isinstance(value[key], bool):
                raise ValueError(f'{key} must be a boolean')
        self.center = [float(x) for x in center]
        self.radius, self.rate_hz = radius, rate
        self.present, self.enabled = value.get('present', self.present), value.get('enabled', self.enabled)
        self.events.append({'kind': 'food_settings', 'time': time, 'settings': dict(value)})

    def sample(self, contacts, brain):
        self.contacts = [c for c in contacts if self.present and
                         sum((c['position_mm'][i]-self.center[i])**2 for i in (0, 1)) <= self.radius**2]
        contact = bool(self.contacts)
        active = contact and self.enabled
        if (contact, active) != (self.contact, self.active):
            self.events.append({'kind': 'food_contact', 'time': brain.time, 'contact': contact, 'active': active})
        self.contact, self.active = contact, active
        brain.set_sensory_input(SUGAR if active else [], self.rate_hz if active else 0)

    def snapshot(self):
        return {'center_mm': self.center.copy(), 'radius_mm': self.radius, 'present': self.present,
                'enabled': self.enabled, 'rate_hz': self.rate_hz, 'contact': self.contact,
                'active': self.active, 'contact_feet': sorted({c['foot'] for c in self.contacts}),
                'contact_points': self.contacts, 'active_seconds': self.active_seconds,
                'mapping': 'pooled foot-contact proxy to reference sugar cohort', 'events': list(self.events)}
