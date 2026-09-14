from copy import deepcopy
import numpy as np
import pandas as pd
import pytest

from nexus.brain.runtime import Brain, Connectome
from nexus.brain.walking import WalkingDecoder
from nexus.coupled import CoupledSession
from nexus.datasets.walking import attach_forward_readout
from test_coupled import ClockBody
from test_descending import brain as descending_brain


def brain():
    b = descending_brain()
    # Test fixture supplies two independent cells with no outgoing edges.
    b.graph.circuits['readouts']['forward_walking'] = ['10763', '11288']
    return b


def test_decoder_tracks_delivered_counts_zero_input_and_disable():
    b = brain()
    d = WalkingDecoder(b)
    assert d.output()['available'] and d.output()['drive'] == 0
    d.observe([10, 0], .1, [1, 1])
    assert d.output()['rate_hz'] == pytest.approx(50 * (1 - np.exp(-1)))
    assert d.output()['drive'] == pytest.approx(d.output()['rate_hz'] / 100)
    assert d.output(0)['drive'] == 0
    d.enabled = False
    assert d.output()['drive'] == 0 and d.output()['rate_hz'] > 0
    d.enabled = True
    d.reset()
    d.observe([10, 10], .1, [0, 0])
    assert d.output()['drive'] == 0


def test_missing_or_partial_readout_never_falls_back_to_baseline_walking():
    b = descending_brain()
    s = CoupledSession(b, ClockBody())
    with pytest.raises(ValueError, match='no mapped'):
        s.command('neural_walking', True)
    assert not s.neural_walking
    b.graph.circuits['readouts']['forward_walking'] = ['99999']
    with pytest.raises(ValueError, match='complete bilateral'):
        WalkingDecoder(b)
    b.graph.snapshot = '630'
    assert not WalkingDecoder(b).output()['available']


def test_neural_mode_has_no_scheduled_walk_turn_or_rest_and_reset_preserves_mode():
    s = CoupledSession(brain(), ClockBody(), autonomous=True)
    s.command('neural_walking', True)
    before = deepcopy(s.behavior.rng.bit_generator.state)
    s.advance(30000)
    assert not s.brain.counts.any()
    assert s.behavior.elapsed == 0 and s.behavior.rng.bit_generator.state == before
    assert all(cmd['drive'] == cmd['turn'] == 0 and cmd['resting'] for _, cmd in s.body.commands)
    s.walking_decoder.observe([10, 10], .1, [1, 1])
    assert s.ground_output()['drive'] > .6 and not s.ground_output()['resting']
    saved = s.walking_decoder.rates.copy()
    s.command('neural_release')
    np.testing.assert_array_equal(s.walking_decoder.rates, saved)
    s.command('reset')
    assert s.neural_walking and s.walking_decoder.output()['drive'] == 0
    s.command('neural_walking', False)
    assert s.ground_output()['drive'] == 1


def test_mode_switch_does_not_change_neural_inputs_counts_rng_or_pause():
    s = CoupledSession(brain(), ClockBody())
    s.command('stimulate', {'ids': ['99999'], 'rate_hz': 100})
    s.advance(100)
    saved = s.brain.v.copy(), s.brain.counts.copy(), deepcopy(s.brain.rng.bit_generator.state)
    s.command('neural_walking', True)
    for before, after in zip(saved[:2], [s.brain.v, s.brain.counts]):
        np.testing.assert_array_equal(before, after)
    assert saved[2] == s.brain.rng.bit_generator.state and not s.running
    assert s.brain.inputs.tolist() == [8]
    s.command('bridge_enabled', False)
    s.walking_decoder.observe([20, 20], .1, [1, 1])
    assert s.ground_output()['drive'] == 0


def test_neural_mode_continues_after_protocol_without_the_authored_scheduler():
    s = CoupledSession(brain(), ClockBody(), autonomous=False)
    s.command('neural_walking', True)
    s.command('resume_after_protocol', True)
    s.command('protocol', {'format': 'nexus-protocol-1', 'duration_ms': 13.7, 'events': []})
    s.advance(1000)
    assert s.running and s.brain.time == .1 and not s.behavior.enabled
    assert s.protocol is None and s.ground_output()['drive'] == 0


def test_forward_mapping_requires_explicit_alias_both_sides_and_same_pack():
    table = pd.DataFrame({'bodyId': [2, 1], 'type': ['DNg100'] * 2, 'somaSide': ['R', 'L'],
                          'superclass': ['descending_neuron'] * 2,
                          'synonyms': ['Sapkal 2024: BDN2'] * 2})
    registry = {'snapshot': 'male-cns:v1.0', 'readouts': {}}
    result = attach_forward_readout(registry, table, [1, 2])
    assert result['readouts']['forward_walking'] == ['1', '2'] and not registry['readouts']
    for mutation in [table.assign(somaSide='L'), table.assign(synonyms=''), table.iloc[:1]]:
        with pytest.raises(ValueError):
            attach_forward_readout(registry, mutation, [1, 2])
    with pytest.raises(ValueError):
        attach_forward_readout(registry, table, [1])


def test_settling_cannot_be_reported_while_neural_walking_drive_remains_active():
    from nexus.recovery import RecoveryMonitor
    monitor = RecoveryMonitor()
    monitor.controls(0, True)
    monitor.controls(0, False)
    for tick in range(100, 10100, 100):
        monitor.observe(tick, 0, 166700, escape=0, disruption=0, steering_hz=0,
                        upright=1, walking_drive=.6)
    assert monitor.status()['phase'] == 'observing' and not monitor.status()['motor_quiet']
    for tick in range(10100, 15100, 100):
        monitor.observe(tick, 0, 166700, escape=0, disruption=0, steering_hz=0,
                        upright=1, walking_drive=0)
    assert monitor.status()['phase'] == 'settled'
