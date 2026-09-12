from copy import deepcopy

import numpy as np
import pytest

from nexus.brain.protocol import Protocol
from nexus.brain.runtime import Brain, Connectome
from nexus.brain.telemetry import NeuralTelemetry


def brain():
    graph = Connectome.from_edges([1, 2, 3, 4], [1], [3], [100.], snapshot='male-cns:v1.0')
    graph.circuits = {'circuits': {'warmth': {'ids': ['1']},
                                 'nociception_proxy': {'ids': ['2', '3']}},
                      'readouts': {'mn9': ['4']}}
    return Brain(graph)


def test_heat_recruits_mapped_md_cells_and_cooling_preserves_direct_pain_input():
    b = brain()
    b.set_heat(39.9)
    assert b.graph.ids[b.inputs].tolist() == [1]
    b.set_heat(40)
    assert b.graph.ids[b.inputs].tolist() == [1, 2, 3]
    assert b.rates.tolist() == [300., 100., 100.]
    b.set_circuit_input('nociception_proxy', 180)
    assert b.rates.tolist() == [180., 180., 300.]
    b.set_heat(30)
    assert b.rates.tolist() == [180., 180., 100.]
    assert b.thermal_nociception_inputs.size == 0
    b.set_heat(None)
    assert b.graph.ids[b.inputs].tolist() == [2, 3]
    assert b.rates.tolist() == [180., 180.]


def test_direct_pain_release_does_not_remove_heat_input_and_telemetry_identifies_source():
    b = brain()
    b.set_heat(100)
    b.set_circuit_input('nociception_proxy', 50)
    assert sorted(b.rates) == [100., 100., 300.]
    b.set_circuit_input('nociception_proxy', 0)
    packet = NeuralTelemetry(b).snapshot(b, False, 0)
    assert packet['thermal_nociception'] == {'ids': ['2', '3'], 'rate_hz': 100.}
    assert 'nociception_proxy' not in packet['circuit_inputs']
    assert set(packet['stimulated_ids']) == {'1', '2', '3'}
    b.set_circuit_input('warmth', 300)  # Direct warmth input, not a temperature scenario.
    assert b.nominal_temperature is None and b.thermal_nociception_inputs.size == 0
    assert b.graph.ids[b.inputs].tolist() == [1]


def test_release_preserves_neural_state_and_reset_clears_thermal_channel():
    b = brain()
    b.set_heat(40)
    b.advance(.01)
    arrays = {name: getattr(b, name).copy() for name in ['v', 'g', 'counts', 'queue', 'queue_size']}
    rng = deepcopy(b.rng.bit_generator.state)
    b.release()
    for name, value in arrays.items():
        np.testing.assert_array_equal(getattr(b, name), value)
    assert b.time == .01 and b.rng.bit_generator.state == rng
    assert not b.inputs.size and not b.thermal_nociception_inputs.size
    b.set_heat(100)
    b.reset()
    assert b.time == 0 and not b.inputs.size and b.thermal_nociception_rate == 0


def test_invalid_thermal_target_is_rejected_before_changing_state_or_starting_protocol():
    b = brain()
    b.set_heat(30)
    b.graph.circuits['circuits']['nociception_proxy']['ids'] = ['999']
    before = b.inputs.copy(), b.rates.copy(), list(b.events)
    for action in [lambda: b.set_heat(40), lambda: Protocol({
            'format': 'nexus-protocol-1', 'duration_ms': 10,
            'events': [{'at_ms': 0, 'action': 'heat', 'celsius': 40}]}, b)]:
        with pytest.raises(ValueError, match='Unknown or invalid'):
            action()
        np.testing.assert_array_equal(b.inputs, before[0])
        np.testing.assert_array_equal(b.rates, before[1])
        assert list(b.events) == before[2] and b.nominal_temperature == 30


def test_thermal_protocol_is_chunk_independent_and_does_not_invent_extra_100c_damage():
    a, b, hot = brain(), brain(), brain()
    description = {'format': 'nexus-protocol-1', 'duration_ms': 20, 'events': [
        {'at_ms': 1, 'action': 'heat', 'celsius': 40},
        {'at_ms': 4, 'action': 'circuit', 'name': 'nociception_proxy', 'rate_hz': 150},
        {'at_ms': 9, 'action': 'heat', 'celsius': None},
        {'at_ms': 16, 'action': 'release'}]}
    p, q = Protocol(description, a), Protocol(description, b)
    p.advance(a, 200)
    for _ in range(20):
        q.advance(b, 10)
    for field in ['v', 'g', 'counts', 'queue', 'queue_size']:
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    assert p.completed and q.completed and not a.inputs.size
    a.reset()
    a.set_heat(40)
    hot.set_heat(100)
    a.advance(.03)
    hot.advance(.03)
    np.testing.assert_array_equal(a.counts, hot.counts)
    np.testing.assert_array_equal(a.v, hot.v)
