"""Display applied inputs, never infer a subjective state from neural activity."""


def active_stimulation(packet):
    circuits = {name for name, value in packet.get('circuit_inputs', {}).items()
                if value.get('rate_hz', 0) > 0 and value.get('ids')}
    temperature = packet.get('nominal_temperature_c')
    boiling = temperature is not None and temperature >= 100 and 'warmth' in circuits
    reduced = packet.get('inhibition_gain', 1) < 1
    labels = []
    if 'looming' in circuits:
        labels.append('FEAR')
    if circuits & {'aversion_proxy', 'nociception_proxy'}:
        labels.append('PAIN')
    if boiling:
        labels.append('BOILING')
    elif reduced:
        labels.append('SEIZURE')
    elif 'warmth' in circuits:
        labels.append('HEAT')
    if packet.get('manual_ids') and not reduced:
        labels.append('CUSTOM')
    if packet.get('sensory_ids') and packet.get('sensory_rate_hz', 0) > 0:
        labels.append('FOOD')
    if packet.get('silenced_count', 0):
        labels.append('SILENCING')
    return tuple(labels)
