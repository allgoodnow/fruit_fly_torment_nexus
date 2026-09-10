import numpy as np
import pytest
from nexus.brain.runtime import Brain, Connectome
from nexus.brain.motor_effects import MotorEffects

MDN = [10763, 11288, 11332, 12348]


def brain():
    g = Connectome.from_edges(MDN+[10360, 523769, 10001, 10010, 99999], [], [], [], snapshot='male-cns:v1.0')
    g.circuits = {'circuits': {'giant_fiber': {'ids': ['10001', '10010']},
                               'aversion_proxy': {'ids': ['99999']}},
                  'readouts': {'steering_left': ['523769'], 'steering_right': ['10360'],
                               'mn9': ['10001', '10010']}}
    return Brain(g)


def test_mdn_spikes_output_gains_motor_ablation_and_decay():
    b = brain()
    effects = MotorEffects(b)
    counts, gains = np.zeros(len(b.counts)), np.ones(len(b.counts))
    counts[effects.mdn] = 1
    for _ in range(30):
        effects.observe(counts, .01, gains, effects.mdn)
    assert effects.output()['retreat'] > .9
    assert effects.output()['escape'] == effects.output()['disruption'] == 0
    effects.descending_enabled = False
    assert effects.output()['retreat'] == 0
    effects.descending_enabled = True
    effects.enabled = False
    assert effects.output()['retreat'] == 0 and effects.output()['mdn_hz'] > 0
    effects.enabled = True
    gains[effects.mdn] = 0
    for _ in range(100):
        effects.observe(counts, .01, gains, effects.mdn)
    assert effects.output()['retreat'] == 0
    effects.reset()
    assert effects.output()['mdn_hz'] == 0


def test_central_aversion_stimulation_alone_no_longer_forces_movement():
    from nexus.coupled import CoupledSession
    from test_coupled import ClockBody
    s = CoupledSession(brain(), ClockBody())
    s.command('circuit', {'name': 'aversion_proxy', 'rate_hz': 100})
    s.advance(3000)
    assert s.brain.counts[-1] > 0
    assert s.motor_effects.output()['retreat'] == 0
    assert all(cmd['drive'] == 1 and cmd['turn'] == 0 for _, cmd in s.body.commands)


def test_mdn_readout_requires_dataset_match_and_complete_group():
    b = brain()
    b.graph.snapshot = '630'
    assert not MotorEffects(b).output()['mdn_available']
    b.graph.snapshot = 'male-cns:v1.0'
    del b.lookup[MDN[0]]
    with pytest.raises(ValueError, match='Incomplete'):
        MotorEffects(b)
