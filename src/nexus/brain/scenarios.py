"""Inspectable presets built from ordinary, logged neural protocol commands."""
from .circuits import heat_rate
from .protocol import ticks
from .targets import SUGAR


def scenario_protocol(name, *, baseline_ms=100, stimulus_ms=500, recovery_ms=500, celsius=40, dataset='630', pain_circuit='aversion_proxy'):
    if dataset not in ('630', 'male-cns:v1.0'):
        raise ValueError('Unknown scenario dataset')
    if pain_circuit not in ('aversion_proxy', 'nociception_proxy') or (pain_circuit == 'nociception_proxy' and dataset != 'male-cns:v1.0'):
        raise ValueError('Unsupported pain candidate circuit for this dataset')
    for duration in (baseline_ms, stimulus_ms, recovery_ms):
        if not 0 < ticks(duration) <= 600000:
            raise ValueError('Each phase must last between 0.1 ms and 60 seconds')
    onset, release = baseline_ms, baseline_ms+stimulus_ms
    events = [{'at_ms': 0, 'action': 'release'}]
    if name == 'defensive':
        commands = [{'action': 'circuit', 'name': 'looming', 'rate_hz': 200}]
        title = 'Looming-threat pathway / escape motor proxy'
    elif name == 'aversion':
        commands = [{'action': 'circuit', 'name': pain_circuit, 'rate_hz': 100 if pain_circuit == 'nociception_proxy' else 200}]
        title = 'Abdominal md candidates / experimental nociception' if pain_circuit == 'nociception_proxy' else 'Central aversion candidate / nociception unresolved'
    elif name in ('heat', 'heat_overload'):
        heat_rate(celsius)
        commands = [{'action': 'heat', 'celsius': celsius}]
        if name == 'heat_overload':
            commands.append({'action': 'inhibition_gain', 'gain': .25})
        title = 'Nominal heat input'+(' + authored network overload' if name == 'heat_overload' else '')
    elif name == 'seizure':
        drive = {'action': 'stimulate', 'ids': SUGAR, 'rate_hz': 200} if dataset == '630' else {'action': 'circuit', 'name': 'warmth', 'rate_hz': 300}
        commands = [drive, {'action': 'inhibition_gain', 'gain': .25}]
        title = 'Seizure-like experiment / network and motor disruption'
    else:
        raise ValueError(f'Unknown experiment: {name}')
    events.extend(dict(command, at_ms=onset) for command in commands)
    events.append({'at_ms': release, 'action': 'release'})
    return {'format': 'nexus-protocol-1', 'name': title,
            'duration_ms': release+recovery_ms, 'events': events, 'dataset': dataset}


def sequence_protocol(stages, *, dataset, pain_circuit='aversion_proxy'):
    """Join presets on one clock, releasing inputs without resetting neural state."""
    if not isinstance(stages, list) or not 1 <= len(stages) <= 50:
        raise ValueError('Choose between 1 and 50 sequence stages')
    offset, events, labels = 0, [], []
    for stage in stages:
        if not isinstance(stage, dict):
            raise ValueError('Each sequence stage must be an object')
        if set(stage)-{'name', 'baseline_ms', 'stimulus_ms', 'recovery_ms', 'celsius'}:
            raise ValueError('Unknown sequence stage setting')
        protocol = scenario_protocol(**stage, dataset=dataset, pain_circuit=pain_circuit)
        duration = ticks(protocol['duration_ms'])
        if offset+duration > 6000000:
            raise ValueError('Sequence cannot exceed 10 minutes')
        events.extend(dict(event, at_ms=(offset+ticks(event['at_ms']))/10)
                      for event in protocol['events'])
        labels.append({'name': protocol['name'], 'start_ms': offset/10, 'end_ms': (offset+duration)/10})
        offset += duration
    return {'format': 'nexus-protocol-1', 'dataset': dataset,
            'name': 'Continuous prepared experiments / no state reset',
            'duration_ms': offset/10, 'events': events, 'stages': labels}
