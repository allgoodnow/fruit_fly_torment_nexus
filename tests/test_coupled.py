import numpy as np
import pytest

from nexus.brain.motor import DNA02, SteeringDecoder
from nexus.brain.runtime import Brain, Connectome
from nexus.coupled import CoupledSession


class ClockBody:
    """A clock spy; actual kinematic causality is checked by probe_steering.py."""
    time = 0.

    def __init__(self):
        self.commands = []

    def reset(self):
        self.time = 0.
        self.commands.clear()

    def advance(self, seconds, **command):
        self.time += seconds
        self.commands.append((self.time, command))


def small_brain():
    graph = Connectome.from_edges(np.array(DNA02, dtype=np.int64),
                                  np.array([], dtype=np.int32), np.array([], dtype=np.int32),
                                  np.array([], dtype=np.float64), snapshot='630')
    return Brain(graph)


def test_shared_clock_splits_at_interventions_and_short_sequence_endpoint():
    brain, body = small_brain(), ClockBody()
    session = CoupledSession(brain, body)
    session.command('protocol', {'format': 'nexus-protocol-1', 'duration_ms': 23.1,
                                'events': [{'at_ms': 1.2, 'action': 'stimulate', 'ids': [DNA02[0]], 'rate_hz': 200},
                                           {'at_ms': 11.3, 'action': 'release'}]})
    session.advance(1000)
    assert brain.time == pytest.approx(body.time) == pytest.approx(.0231)
    assert not session.running and session.protocol.completed
    assert [(e['kind'], e['time']) for e in brain.events] == [('stimulate', .0012), ('release', .0113)]
    assert any(abs(at-.0012) < 1e-9 for at, _ in body.commands)
    assert any(abs(at-.0113) < 1e-9 for at, _ in body.commands)
    session.command('step')
    assert brain.time == pytest.approx(.0331) and body.time == pytest.approx(brain.time)
    session.command('reset')
    assert brain.time == body.time == 0
    assert not session.decoder.rates.any() and session.generation == 1


def test_decoder_uses_measured_counts_and_respects_output_silencing_and_ablation():
    brain = small_brain()
    decoder = SteeringDecoder(brain)
    decoder.observe([10, 0], .01, [1, 1])
    active = decoder.output(1)
    assert active['left_drive'] < active['right_drive']
    decoder.enabled = False
    blocked = decoder.output(1)
    assert blocked['left_drive'] == blocked['right_drive'] == 1
    assert decoder.rates[0] > 0
    decoder.reset()
    decoder.observe([10, 0], .01, [0, 1])
    assert decoder.output(1)['left_drive'] == decoder.output(1)['right_drive']
    assert decoder.output(0)['drive'] == 0


def test_neural_release_preserves_clocks_and_baseline_and_rejected_sequence_preserves_state():
    brain, body = small_brain(), ClockBody()
    session = CoupledSession(brain, body)
    session.command('drive', .5)
    session.command('stimulate', {'ids': [DNA02[0]], 'rate_hz': 200})
    session.advance(100)
    saved = brain.v.copy()
    session.command('brain', {'kind': 'release'})
    assert session.baseline == .5 and brain.time == body.time == .01
    np.testing.assert_array_equal(brain.v, saved)
    assert not len(brain.inputs)
    with pytest.raises(ValueError):
        session.command('protocol', {'format': 'invalid'})
    assert session.protocol is None and brain.time == .01


def test_descending_command_interrupts_rest_and_motor_ablation_preserves_neural_state():
    from test_descending import brain as circuit_brain, MDN
    sessions = [CoupledSession(circuit_brain(), ClockBody(), autonomous=True) for _ in range(2)]
    for index, s in enumerate(sessions):
        s.behavior.phase, s.behavior.remaining = 'resting', 2.
        s.command('motor_effects_enabled', index == 0)
        s.command('stimulate', {'ids': MDN, 'rate_hz': 100})
        s.advance(3000)
    active, blocked = sessions
    assert active.behavior.interrupted and not blocked.behavior.interrupted
    assert not active.body.commands[-1][1].get('resting')
    assert blocked.body.commands[-1][1]['resting']
    motor = active.motor_output(active.behavior.output(1.))
    assert motor['drive'] < -.6 and motor['turn'] == 0
    np.testing.assert_array_equal(active.brain.counts, blocked.brain.counts)
    np.testing.assert_array_equal(active.brain.v, blocked.brain.v)
    active.command('neural_release')
    active.advance(10000)
    assert active.motor_effects.output()['retreat'] == 0
    assert not active.behavior.interrupted
    assert active.body.commands[-1][1]['resting']
