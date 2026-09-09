import numpy as np
import pytest

from nexus.brain.runtime import Brain, Connectome, REFRACTORY_STEPS
from nexus.brain.targets import SUGAR
from nexus.brain.motor import DNA02
from nexus.coupled import CoupledSession
from nexus.environment import FoodEnvironment


def brain():
    return Brain(Connectome.from_edges(np.array(SUGAR+DNA02, dtype=np.int64), [], [], []))


class Body:
    time = 0.

    def __init__(self):
        self.points = []

    def ground_contacts(self):
        return [{'foot': 'lf', 'position_mm': p} for p in self.points]

    def advance(self, seconds, **kwargs):
        self.time += seconds

    def reset(self):
        self.time = 0.


def test_contact_only_drives_sensory_input_and_can_be_disabled_without_removing_food():
    b, body = brain(), Body()
    world = FoodEnvironment(center=(0., 0.), radius=1.)
    session = CoupledSession(b, body, environment=world)
    assert not len(b.inputs)
    body.points = [[2, 0, 0]]
    session.sync_environment()
    assert not world.contact
    body.points = [[.5, 0, 0]]
    session.sync_environment()
    assert world.active and len(b.sensory_inputs) == 21
    session.advance(100)
    assert world.active_seconds == pytest.approx(.01)
    session.command('food_config', {'enabled': False})
    assert world.present and world.contact and not world.active and not len(b.inputs)
    assert b.time == body.time == .01
    session.command('food_config', {'enabled': True})
    session.command('food_config', {'present': False})
    assert not world.contact and not len(b.inputs)


def test_manual_and_sensory_channels_merge_without_duplicate_pulses_and_release_independently():
    b = brain()
    b.stimulate([SUGAR[0], DNA02[0]], 300)
    b.set_sensory_input(SUGAR, 200)
    assert len(b.inputs) == 22 and len(set(b.inputs.tolist())) == 22
    rates = dict(zip(b.inputs.tolist(), b.rates.tolist()))
    assert rates[b.lookup[int(SUGAR[0])]] == 300
    assert rates[b.lookup[int(SUGAR[1])]] == 200
    b.silence([SUGAR[0]])
    b.advance(.01)
    voltage = b.v.copy()
    b.release()
    assert len(b.inputs) == 21 and len(b.manual_inputs) == 0
    assert b.sensory_rate == 200 and b.output_gain.min() == 1
    np.testing.assert_array_equal(b.v, voltage)
    b.stimulate([DNA02[0]], 150)
    b.set_sensory_input([], 0)
    assert b.inputs.tolist() == [b.lookup[int(DNA02[0])]]
    assert b.refractory[b.lookup[int(SUGAR[0])]] == REFRACTORY_STEPS
    b.reset()
    assert not len(b.inputs) and not len(b.manual_inputs) and not len(b.sensory_inputs)


def test_invalid_world_settings_and_sensory_input_preserve_prior_state():
    b = brain()
    b.set_sensory_input(SUGAR, 200)
    for ids, rate in ((SUGAR, float('nan')), (['123'], 200), (SUGAR, 1001)):
        with pytest.raises(ValueError):
            b.set_sensory_input(ids, rate)
        assert b.sensory_rate == 200 and len(b.inputs) == 21
    env = FoodEnvironment()
    before = env.snapshot()
    for value in ({'center_mm': [float('nan'), 0]}, {'radius_mm': -1}, {'enabled': 'false'}, {'rate_hz': 0}):
        with pytest.raises(ValueError):
            env.configure(value, 0)
        assert env.snapshot() == before
