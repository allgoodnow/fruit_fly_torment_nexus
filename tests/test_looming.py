"""Approach geometry, exact timing, input isolation, and causal graph delivery."""
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest

from nexus.brain.looming import LoomingInput
from nexus.brain.runtime import Brain, Connectome, REFRACTORY_STEPS
from nexus.brain.protocol import Protocol
from nexus.brain.scenarios import scenario_protocol, sequence_protocol
from nexus.brain.telemetry import NeuralTelemetry
from nexus.datasets.looming import attach_velocity_circuit
from nexus.stimulation import active_stimulation


def brain():
    graph = Connectome.from_edges([1, 2, 3, 4], [0, 1, 2], [2, 2, 3], [50., 50., 50.],
                                  snapshot='male-cns:v1.0')
    graph.circuits = {'circuits': {name: {'ids': [str(root)]} for name, root in
                       [('looming', 1), ('looming_velocity', 2), ('giant_fiber', 3),
                        ('nociception_proxy', 4), ('warmth', 4)]},
                      'readouts': {'mn9': ['3']}}
    return Brain(graph)


def test_approach_geometry_is_constant_speed_and_duration_changes_expansion_speed():
    model = LoomingInput(37, 5000)
    angle, speed, size, velocity = model.sample(np.arange(37, 5038))
    assert angle[0] == pytest.approx(5) and angle[-1] == pytest.approx(120)
    assert (np.diff(angle) > 0).all() and (np.diff(speed) > 0).all()
    distance = 1 / np.tan(np.deg2rad(angle) / 2)
    np.testing.assert_allclose(np.diff(distance), np.diff(distance)[0], atol=1e-12)
    np.testing.assert_allclose((angle[2:] - angle[:-2]) / .0002, speed[1:-1], rtol=1e-5)
    assert angle[np.argmax(size)] == pytest.approx(45, abs=.2)
    assert size.min() == velocity.min() == 0 and size.max() <= 200 and velocity.max() <= 200
    slow = LoomingInput(0, 10000).sample(np.array([0, 5000]))
    fast = model.sample(np.array([37, 2537]))
    np.testing.assert_allclose(slow[0], fast[0])
    np.testing.assert_allclose(slow[1] * 2, fast[1])


def test_chunking_and_automatic_expiry_preserve_exact_neural_state_and_trace():
    a, b = brain(), brain()
    for item in (a, b):
        item.set_looming(113.7)
    whole = a.advance(.2, trace_ids=[3])
    pieces = [b.advance(duration, trace_ids=[3]) for duration in [.0037, .04, .0801, .0762]]
    for field in ('v', 'g', 'counts', 'last_spike', 'queue', 'queue_size', 'refractory'):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    np.testing.assert_array_equal(whole['voltage_mv'], np.concatenate([p['voltage_mv'] for p in pieces]))
    np.testing.assert_array_equal(whole['steps'], np.concatenate([p['steps'] for p in pieces]))
    assert a.rng.bit_generator.state == b.rng.bit_generator.state
    assert list(a.activity) == list(b.activity)
    assert a.looming_input is b.looming_input is None and not a.inputs.size
    assert np.all(a.refractory == REFRACTORY_STEPS)
    assert [(e['kind'], e['time']) for e in a.events] == [('looming_input', 0), ('looming_release', .1137)]


def test_release_pause_overlap_and_constant_override_do_not_reset_the_brain():
    b = brain()
    b.stimulate([1], 500)
    b.set_circuit_input('nociception_proxy', 100)
    b.set_looming(500)
    assert b.rates[b.inputs == 0] == 500
    b.advance(.1)
    monitor = NeuralTelemetry(b)
    packet = monitor.snapshot(b, False, 0)
    assert set(active_stimulation(packet)) == {'FEAR', 'PAIN', 'CUSTOM'}
    assert monitor.snapshot(b, False, 0)['looming_input'] == packet['looming_input']
    before = b.v.copy(), b.counts.copy(), deepcopy(b.rng.bit_generator.state)
    b.set_circuit_input('looming', 150)
    assert b.looming_input is None and b.circuit_inputs['nociception_proxy'][1] == 100
    assert set(b.inputs) == {0, 3}  # LC4 goes away when constant input replaces the approach.
    b.release()
    np.testing.assert_array_equal(before[0], b.v)
    np.testing.assert_array_equal(before[1], b.counts)
    assert before[2] == b.rng.bit_generator.state and b.time == .1
    assert active_stimulation(monitor.snapshot(b, False, 0)) == ()
    b.set_looming(100)
    b.reset()
    assert b.looming_input is None and not b.rates.size and not b.inputs.size


def test_expiry_preserves_other_input_sources_and_protocols_stay_compact():
    b = brain()
    b.stimulate([1], 500)
    b.set_circuit_input('nociception_proxy', 100)
    b.set_looming(3.7)
    b.advance(.01)
    assert set(b.inputs) == {0, 3} and set(b.rates) == {500, 100}
    stages = [{'name': 'defensive', 'baseline_ms': 1, 'stimulus_ms': 500, 'recovery_ms': 1}] * 50
    plan = sequence_protocol(stages, dataset=b.graph.snapshot)
    assert len(plan['events']) == 150
    Protocol(plan, b)


@pytest.mark.parametrize('invalid', [True, '500', 0, -1, 1.01, 60000.1, float('nan')])
def test_invalid_approach_is_atomic(invalid):
    b = brain()
    b.set_looming(50)
    before = b.looming_input, b.inputs.copy(), b.rates.copy(), list(b.events)
    with pytest.raises(ValueError):
        b.set_looming(invalid)
    assert b.looming_input == before[0] and list(b.events) == before[3]
    np.testing.assert_array_equal(b.inputs, before[1])
    np.testing.assert_array_equal(b.rates, before[2])


def test_protocol_validation_and_replay_reject_before_changing_state():
    b = brain()
    plan = scenario_protocol('defensive', dataset=b.graph.snapshot)
    plan['events'][1]['duration_ms'] = 1500
    with pytest.raises(ValueError, match='beyond'):
        Protocol(plan, b)
    assert b.step == 0 and not b.events
    b.set_looming(5)
    with pytest.raises(ValueError, match='Split input replay'):
        b.advance(.01, input_events=np.zeros((100, 2), dtype=bool))
    assert b.step == 0


def test_looming_output_blockade_changes_downstream_spikes_with_identical_input_draws():
    a, b = brain(), brain()
    for item in (a, b):
        item.set_looming(500)
    b.silence([1, 2])
    for item in (a, b):
        item.advance(.5)
    np.testing.assert_array_equal(a.counts[:2], b.counts[:2])
    assert a.counts[2] > 0 and a.counts[3] > 0
    assert b.counts[2] == b.counts[3] == 0
    assert a.rng.bit_generator.state == b.rng.bit_generator.state


def test_legacy_pack_reports_missing_velocity_pathway_without_foreign_ids():
    b = brain()
    del b.graph.circuits['circuits']['looming_velocity']
    b.set_looming(20)
    assert b.looming_snapshot()['lc4_rate_hz'] is None
    assert not b.looming_snapshot()['velocity_pathway_available']
    assert b.inputs.tolist() == [0]
    b.advance(.03)
    assert not b.inputs.size


def test_lc4_mapping_uses_current_annotations_and_rejects_wrong_registry():
    b = brain()
    registry = dict(snapshot=b.graph.snapshot, circuits=b.graph.circuits['circuits'])
    table = pd.DataFrame({'bodyId': [1, 2, 999], 'type': ['LPLC2', 'LC4', 'LC4'],
                          'superclass': ['visual_projection'] * 3})
    mapped = attach_velocity_circuit(registry, table, b.graph.ids)
    assert mapped['circuits']['looming_velocity']['ids'] == ['2']
    invalid = deepcopy(registry)
    invalid['circuits']['looming']['ids'] = ['999']
    with pytest.raises(ValueError, match='LPLC2 registry'):
        attach_velocity_circuit(invalid, table, b.graph.ids)
    with pytest.raises(ValueError, match='requires MaleCNS'):
        attach_velocity_circuit(dict(registry, snapshot='630'), table, b.graph.ids)


def test_coupled_expiry_uses_pre_expiry_targets_and_stops_body_at_the_same_boundary():
    from nexus.coupled import CoupledSession
    from test_coupled import ClockBody
    from test_descending import brain as descending_brain
    b = descending_brain()
    b.graph.circuits['circuits']['looming'] = {'ids': ['99999']}
    s = CoupledSession(b, ClockBody())
    s.command('loom', 13.7)
    observed = []
    original = s.motor_effects.observe
    def observe(counts, seconds, gains, direct):
        observed.append((b.step, list(direct)))
        original(counts, seconds, gains, direct)
    s.motor_effects.observe = observe
    s.advance(200)
    assert observed == [(100, [8]), (137, [8]), (200, [])]
    assert b.looming_input is None and s.body.time == pytest.approx(.02)
    assert any(at == pytest.approx(.0137) for at, _ in s.body.commands)
