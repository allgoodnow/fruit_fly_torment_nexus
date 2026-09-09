"""Inspectable presets built from ordinary, logged neural protocol commands."""
from .circuits import heat_rate
from .protocol import ticks
from .targets import SUGAR


def scenario_protocol(name, *, baseline_ms=100, stimulus_ms=500, recovery_ms=500, celsius=40):
    for duration in (baseline_ms, stimulus_ms, recovery_ms):
        if not 0 < ticks(duration) <= 600000:
            raise ValueError('Each phase must last between 0.1 ms and 60 seconds')
    onset, release = baseline_ms, baseline_ms+stimulus_ms
    events = [{'at_ms': 0, 'action': 'release'}]
    if name == 'defensive':
        commands = [{'action': 'circuit', 'name': 'looming', 'rate_hz': 200}]
        title = 'Looming-threat pathway / escape motor proxy'
    elif name == 'aversion':
        commands = [{'action': 'circuit', 'name': 'aversion_proxy', 'rate_hz': 200}]
        title = 'Central aversion candidate / nociception unresolved'
    elif name in ('heat', 'heat_overload'):
        heat_rate(celsius)
        commands = [{'action': 'heat', 'celsius': celsius}]
        if name == 'heat_overload':
            commands.append({'action': 'inhibition_gain', 'gain': .25})
        title = 'Nominal heat input'+(' + authored network overload' if name == 'heat_overload' else '')
    elif name == 'seizure':
        commands = [{'action': 'stimulate', 'ids': SUGAR, 'rate_hz': 200},
                    {'action': 'inhibition_gain', 'gain': .25}]
        title = 'Seizure-like experiment / network and motor disruption'
    else:
        raise ValueError(f'Unknown experiment: {name}')
    events.extend(dict(command, at_ms=onset) for command in commands)
    events.append({'at_ms': release, 'action': 'release'})
    return {'format': 'nexus-protocol-1', 'name': title,
            'duration_ms': release+recovery_ms, 'events': events}
