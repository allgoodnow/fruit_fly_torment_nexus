import numpy as np
import pytest

from nexus.behavior import GroundBehavior
from nexus.coupled import CoupledSession
from test_coupled import ClockBody, small_brain
from nexus.brain.motor import DNA02


def test_behavior_is_seeded_and_only_uses_uninterrupted_simulated_time():
    a, b = GroundBehavior(True), GroundBehavior(True)
    for _ in range(100):
        a.advance(.01, False)
    b.advance(1., False)
    for _ in range(150):
        a.advance(.01, True)
    assert a.elapsed == pytest.approx(b.elapsed)
    assert a.remaining == pytest.approx(b.remaining)
    assert a.output(1)['state'] == 'interrupted'
    for item in (a, b):
        for _ in range(1000):
            item.advance(.01, False)
    assert a.phase == b.phase and a.turn == b.turn and a.cycles == b.cycles
    a.reset()
    assert a.enabled and a.elapsed == 0 and a.phase == 'exploring'
    assert a.output(0)['drive'] == a.output(0)['turn'] == 0


def test_neural_activity_interrupts_rest_and_release_does_not_erase_response():
    brain, body = small_brain(), ClockBody()
    s = CoupledSession(brain, body, autonomous=True)
    while s.behavior.phase != 'resting':
        s.advance(100)
    assert body.commands[-1][1]['resting']
    motor = s.motor_output(s.behavior.output(1.))
    assert motor['left_drive'] == motor['right_drive'] == motor['drive'] == 0
    state = brain.rng.bit_generator.state
    s.command('autonomous', False)
    s.command('autonomous', True)
    assert brain.rng.bit_generator.state == state
    s.command('stimulate', {'ids': [DNA02[0]], 'rate_hz': 200})
    s.advance(1000)
    assert s.behavior.interrupted and not body.commands[-1][1].get('resting')
    elapsed = s.behavior.elapsed
    s.command('neural_release')
    s.advance(100)
    assert s.behavior.interrupted and s.behavior.elapsed == elapsed
    s.advance(15000)
    assert not s.behavior.interrupted and s.behavior.elapsed > elapsed
    assert brain.time == pytest.approx(body.time)


@pytest.mark.parametrize('resume,enabled', [(True, True), (False, True), (True, False)])
def test_protocol_completion_continues_only_when_free_behavior_and_resume_are_enabled(resume, enabled):
    brain, body = small_brain(), ClockBody()
    s = CoupledSession(brain, body, autonomous=enabled)
    s.command('resume_after_protocol', resume)
    s.command('protocol', {'format': 'nexus-protocol-1', 'duration_ms': 23.1,
                          'events': [{'at_ms': 12.7, 'action': 'release'}]})
    s.advance(1000)
    if resume and enabled:
        assert s.running and s.protocol is None and brain.time == .1
    else:
        assert not s.running and s.protocol.completed and brain.time == .0231
    assert list(brain.events)[-1]['time'] == .0127
    s.command('reset')
    assert brain.time == body.time == s.behavior.elapsed == 0
    assert s.behavior.enabled == enabled and s.resume_after_protocol == resume


def test_single_step_at_experiment_end_never_restarts_paused_simulation():
    brain, body = small_brain(), ClockBody()
    s = CoupledSession(brain, body, autonomous=True)
    s.command('protocol', {'format': 'nexus-protocol-1', 'duration_ms': 12.7, 'events': []})
    s.advance(100)
    s.command('running', False)
    s.command('step')
    assert brain.time == .0127 and s.protocol.completed and not s.running
    s.command('step')
    assert brain.time == .0227 and not s.running and s.protocol is None
