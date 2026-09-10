"""Observe post-input settling without changing either simulation.

These operational thresholds describe this model, not biological recovery.
"""
from collections import deque
from copy import deepcopy
import math


class RecoveryMonitor:
    RULES = {'neural_hz_per_neuron_max': .1, 'escape_max': .05,
             'disruption_max': .05, 'avoidance_max': .05, 'steering_hz_max': 10.,
             'upright_min': .8, 'hold_ms': 500.,
             'interpretation': 'Consecutive qualifying simulation intervals; not biological recovery or subjective relief.'}

    def __init__(self):
        self.reset()

    def reset(self):
        self.episodes = deque(maxlen=128)
        self.events = deque(maxlen=256)
        self.current = None
        self.active = False
        self.serial = 0
        self.event_serial = 0
        self.last_tick = 0

    def event(self, kind, tick):
        self.event_serial += 1
        self.events.append({'id': self.event_serial, 'episode': self.serial,
                            'kind': kind, 'time': tick*.0001})

    def controls(self, tick, active):
        if bool(active) == self.active:
            return
        self.active = bool(active)
        if active:
            if self.current and self.current['release_ms'] is not None:
                self.current['observation_end_ms'] = tick/10
                self.current['observation_end_reason'] = 'new_input'
            self.serial += 1
            self.current = {'id': self.serial, 'input_start_ms': tick/10, 'release_ms': None,
                            'phase': 'input_active', 'settled_at_ms': None,
                            'quiet_ticks': {'neural': 0, 'motor': 0, 'posture': 0, 'all': 0}}
            self.episodes.append(self.current)
        else:
            self.current.update(release_ms=tick/10, phase='observing', observed_until_ms=tick/10)
            self.event('release_observation', tick)

    def observe(self, tick, spikes, neurons, *, escape, disruption, steering_hz, upright, avoidance=0.):
        elapsed = tick-self.last_tick
        if elapsed <= 0 or neurons <= 0 or spikes < 0:
            raise ValueError('Recovery observations require advancing time and valid counts')
        if not all(math.isfinite(v) for v in (escape, disruption, steering_hz, avoidance)):
            raise ValueError('Recovery readouts must be finite')
        if upright is not None and not math.isfinite(upright):
            raise ValueError('Posture must be finite when available')
        self.last_tick = tick
        if self.current is None or self.active:
            return
        rate = spikes/(elapsed*.0001)/neurons
        rules = self.RULES
        checks = {'neural': rate <= rules['neural_hz_per_neuron_max'],
                  'motor': escape <= rules['escape_max'] and disruption <= rules['disruption_max']
                           and avoidance <= rules['avoidance_max']
                           and steering_hz <= rules['steering_hz_max'],
                  'posture': upright is not None and upright >= rules['upright_min']}
        checks['all'] = all(checks.values())
        row = self.current
        for key, qualifies in checks.items():
            row['quiet_ticks'][key] = row['quiet_ticks'][key]+elapsed if qualifies else 0
        settled = row['quiet_ticks']['all'] >= 5000
        phase = 'settled' if settled else 'observing'
        if phase != row['phase']:
            kind = 'response_settled' if settled else 'response_returned'
            self.event(kind, tick)
            row['settled_at_ms'] = tick/10 if settled else None
        row.update(phase=phase, observed_until_ms=tick/10,
                   neural_hz_per_neuron=rate, neural_quiet=checks['neural'],
                   motor_quiet=checks['motor'], upright=upright,
                   upright_enough=checks['posture'])

    def snapshot(self):
        return {'rules': dict(self.RULES), 'phase': self.current['phase'] if self.current else 'no_input_yet',
                'episodes': deepcopy(list(self.episodes)), 'events': list(self.events)}

    def status(self):
        if self.current is None:
            return {'phase': 'no_input_yet'}
        return deepcopy(self.current)
