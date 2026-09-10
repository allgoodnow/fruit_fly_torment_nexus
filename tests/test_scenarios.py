from copy import deepcopy
import numpy as np
import pytest

from nexus.brain.circuits import circuit_ids, heat_rate
from nexus.brain.motor import DNA02
from nexus.brain.motor_effects import MotorEffects
from nexus.brain.protocol import Protocol
from nexus.brain.runtime import Brain, Connectome
from nexus.brain.scenarios import scenario_protocol
from nexus.brain.targets import SUGAR


def brain():
    ids = list(dict.fromkeys(SUGAR+DNA02+sum([circuit_ids(n) for n in
               ('looming', 'warmth', 'aversion_proxy', 'giant_fiber')], [])))
    return Brain(Connectome.from_edges(ids, [], [], [], snapshot='630'))


def test_circuit_overlap_release_and_invalid_inputs_are_atomic():
    b = brain()
    b.set_sensory_input(SUGAR, 200)
    b.stimulate([circuit_ids('warmth')[0]], 500)
    b.set_circuit_input('looming', 200)
    b.set_heat(100)
    assert len(b.inputs) == 21+210+7 and b.rates[b.inputs == b.resolve([circuit_ids('warmth')[0]])[0]][0] == 500
    b.advance(.01)
    state = b.v.copy(), b.g.copy(), b.counts.copy(), deepcopy(b.rng.bit_generator.state)
    prior = b.inputs.copy()
    for bad in [19, 101, float('nan'), True, '100']:
        with pytest.raises(ValueError):
            b.set_heat(bad)
        np.testing.assert_array_equal(b.inputs, prior)
        assert b.nominal_temperature == 100
    b.set_circuit_input('looming', 0)
    assert len(b.inputs) == 28 and b.nominal_temperature == 100
    b.release()
    assert not b.circuit_inputs and b.nominal_temperature is None
    assert len(b.inputs) == 21 and b.time == .01
    for before, after in zip(state[:3], [b.v, b.g, b.counts]):
        np.testing.assert_array_equal(before, after)
    assert state[3] == b.rng.bit_generator.state


def test_nominal_heat_saturates_without_inventing_thermal_damage():
    a, b = brain(), brain()
    a.set_heat(40)
    b.set_heat(100)
    for item in (a, b):
        item.advance(.025)
    np.testing.assert_array_equal(a.v, b.v)
    np.testing.assert_array_equal(a.counts, b.counts)
    assert a.inhibition_gain == b.inhibition_gain == 1
    assert heat_rate(25)[1] == 0 and heat_rate(40)[1] == heat_rate(100)[1] == 300
    b.set_heat(None)
    assert not b.inputs.size and b.nominal_temperature is None


@pytest.mark.parametrize('name', ['defensive', 'aversion', 'heat', 'heat_overload', 'seizure'])
def test_scenarios_match_exact_manual_execution_and_release(name):
    a, b = brain(), brain()
    description = scenario_protocol(name, baseline_ms=3.7, stimulus_ms=11.3, recovery_ms=8.1, celsius=100)
    p = Protocol(description, a)
    for _ in range(3):
        p.advance(a, 100)
    previous = 0
    for e in description['events']:
        if e['at_ms'] > previous:
            b.advance((e['at_ms']-previous)/1000)
        {'release': lambda: b.release(),
         'circuit': lambda: b.set_circuit_input(e['name'], e['rate_hz']),
         'heat': lambda: b.set_heat(e['celsius']),
         'inhibition_gain': lambda: b.set_inhibition_gain(e['gain']),
         'stimulate': lambda: b.stimulate(e['ids'], e['rate_hz'])}[e['action']]()
        previous = e['at_ms']
    b.advance((description['duration_ms']-previous)/1000)
    assert p.completed and not a.circuit_inputs and a.inhibition_gain == 1
    assert a.time == b.time == .0231
    for field in ('v', 'g', 'queue', 'queue_size', 'counts'):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    invalid = deepcopy(description)
    invalid['events'].append({'at_ms': 23.1, 'action': 'heat', 'celsius': 101})
    with pytest.raises(ValueError):
        Protocol(invalid, a)
    assert a.time == .0231 and not a.circuit_inputs


def test_motor_proxy_uses_neural_counts_excludes_direct_input_and_supports_ablation():
    b = brain()
    effects = MotorEffects(b)
    counts, gain = np.zeros(len(b.counts)), np.ones(len(b.counts))
    counts[:] = 2
    direct = np.arange(len(counts), dtype=np.int32)
    for _ in range(20):
        effects.observe(counts, .01, gain, direct)
    assert effects.output()['disruption'] == 0
    assert effects.output()['escape'] > 0
    for _ in range(30):
        effects.observe(counts, .01, gain, np.array([], dtype=np.int32))
    assert effects.output()['disruption'] > .9
    remembered = effects.disruption
    effects.enabled = False
    assert effects.output()['disruption'] == effects.output()['escape'] == 0
    assert effects.disruption == remembered
    effects.enabled = True
    for _ in range(100):
        effects.observe(counts, .01, np.zeros_like(gain), direct)
    assert effects.output()['disruption'] < .0001 and effects.output()['escape'] < .0001
    effects.reset()
    assert effects.output()['disruption'] == effects.output()['escape'] == 0


def test_aversion_readout_obeys_spikes_output_gains_ablation_and_decay():
    b = brain()
    effects = MotorEffects(b)
    counts, gains = np.zeros(len(b.counts)), np.ones(len(b.counts))
    counts[effects.aversion] = 1
    for _ in range(30):
        effects.observe(counts, .01, gains, effects.aversion)
    assert effects.output()['avoidance'] > .9
    assert effects.output()['escape'] == effects.output()['disruption'] == 0
    effects.avoidance_enabled = False
    assert effects.output()['avoidance'] == 0
    effects.avoidance_enabled = True
    effects.enabled = False
    assert effects.output()['avoidance'] == 0
    assert effects.output()['aversion_hz'] > 0
    effects.enabled = True
    gains[effects.aversion] = 0
    for _ in range(100):
        effects.observe(counts, .01, gains, effects.aversion)
    assert effects.output()['avoidance'] == 0
    effects.reset()
    assert effects.output()['aversion_hz'] == 0
