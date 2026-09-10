import json
from unittest.mock import Mock

import numpy as np
import pytest

from nexus.recovery import RecoveryMonitor
from nexus.coupled import CoupledSession
from nexus.brain.motor import DNA02
from test_coupled import ClockBody, small_brain


def observe(monitor, end, *, spikes=0, upright=1., escape=0., disruption=0., steering=0., retreat=0.):
    monitor.observe(end, spikes, 100, escape=escape, disruption=disruption,
                    steering_hz=steering, upright=upright, retreat=retreat)


def released_monitor():
    m = RecoveryMonitor()
    m.controls(0, True)
    observe(m, 100)
    m.controls(100, False)
    return m


def test_release_never_claims_settling_and_paused_wall_time_does_not_count():
    m = released_monitor()
    assert m.status()['phase'] == 'observing'
    for _ in range(50):
        m.snapshot()
        m.controls(100, False)
    assert m.status()['observed_until_ms'] == 10
    observe(m, 5099)
    assert m.status()['phase'] == 'observing'
    observe(m, 5100)
    assert m.status()['settled_at_ms'] == 510
    assert [e['kind'] for e in m.events] == ['release_observation', 'response_settled']
    observe(m, 5200, spikes=10)
    assert m.status()['phase'] == 'observing' and m.status()['settled_at_ms'] is None
    assert m.events[-1]['kind'] == 'response_returned'


@pytest.mark.parametrize('failure', [{'spikes': 10}, {'upright': -1}, {'upright': None},
                                   {'escape': .1}, {'disruption': .1}, {'steering': 20}, {'retreat': .1}])
def test_each_condition_independently_prevents_a_false_recovery(failure):
    m = released_monitor()
    for tick in range(200, 5300, 100):
        observe(m, tick, **failure)
    assert m.status()['phase'] == 'observing'
    assert m.status()['quiet_ticks']['all'] == 0


def test_new_input_closes_observation_and_retained_event_ids_never_repeat():
    m = RecoveryMonitor()
    for i in range(300):
        start = i*200
        m.controls(start, True)
        observe(m, start+100)
        m.controls(start+100, False)
        observe(m, start+200)
    assert len(m.episodes) == 128 and len(m.events) == 256
    assert len({e['id'] for e in m.events}) == 256
    assert m.episodes[-2]['observation_end_reason'] == 'new_input'
    assert m.episodes[-1]['id'] == 300
    saved = m.snapshot()
    saved['episodes'][-1]['quiet_ticks']['all'] = 999999
    assert m.status()['quiet_ticks']['all'] == 100
    m.reset()
    assert m.snapshot()['phase'] == 'no_input_yet' and not m.events


def test_monitor_is_observational_and_uses_exact_protocol_boundaries():
    a, b = [CoupledSession(small_brain(), ClockBody(), autonomous=True) for _ in range(2)]
    b.recovery = Mock()
    protocol = {'format': 'nexus-protocol-1', 'duration_ms': 623.1,
                'events': [{'at_ms': 1.2, 'action': 'stimulate', 'ids': [DNA02[0]], 'rate_hz': 200},
                           {'at_ms': 11.3, 'action': 'release'}]}
    for session in (a, b):
        session.command('resume_after_protocol', False)
        session.command('protocol', protocol)
        while session.running:
            session.advance(100)
    np.testing.assert_array_equal(a.brain.counts, b.brain.counts)
    np.testing.assert_array_equal(a.brain.v, b.brain.v)
    np.testing.assert_array_equal(a.brain.queue, b.brain.queue)
    assert a.brain.rng.bit_generator.state == b.brain.rng.bit_generator.state
    assert a.body.commands == b.body.commands
    row = a.recovery.status()
    assert row['input_start_ms'] == 1.2 and row['release_ms'] == 11.3
    assert row['observed_until_ms'] == 623.1 and row['upright'] is None
    assert row['phase'] != 'settled'  # Clock-only fixtures cannot establish body recovery.
    a.command('reset')
    assert a.recovery.status()['phase'] == 'no_input_yet'
