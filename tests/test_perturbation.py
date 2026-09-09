from copy import deepcopy

import numpy as np
import pytest

from nexus.brain.runtime import Brain, Connectome
from nexus.brain.protocol import Protocol
from test_brain import graph
from test_coupled import ClockBody, small_brain
from nexus.coupled import CoupledSession


@pytest.mark.parametrize('gain', [0., .25, 1.])
def test_overlay_equals_independently_scaled_graph_with_identical_input(gain):
    original = graph()
    saved = original.weights.copy()
    scaled = Connectome(original.ids, original.offsets, original.posts,
                        np.where(original.weights < 0, original.weights*gain, original.weights))
    a, b = Brain(original), Brain(scaled)
    a.set_inhibition_gain(gain)
    for brain in (a, b):
        brain.stimulate([100, 101], 500)
        brain.silence([104])  # The overlay must compose with output silencing.
    for _ in range(7):
        a.advance(.01)
    b.advance(.07)
    for name in ('v', 'g', 'last_spike', 'queue', 'queue_size', 'counts'):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    np.testing.assert_array_equal(original.weights, saved)


def test_paused_restore_and_release_preserve_dynamic_state_and_sensory_channel():
    b = Brain(graph())
    b.set_sensory_input([100], 300)
    b.stimulate([101], 200)
    b.set_inhibition_gain(.25)
    b.advance(.023)
    fields = ('v', 'g', 'last_spike', 'queue', 'queue_size', 'counts')
    state = {name: getattr(b, name).copy() for name in fields}
    rng, history = deepcopy(b.rng.bit_generator.state), list(b.history)
    b.set_inhibition_gain(1.)
    assert len(b.manual_inputs) == 1 and len(b.sensory_inputs) == 1
    b.set_inhibition_gain(0.)
    b.release()
    assert b.inhibition_gain == 1 and b.time == .023 and not len(b.manual_inputs)
    assert b.sensory_inputs.tolist() == [0]
    for name in fields:
        np.testing.assert_array_equal(getattr(b, name), state[name])
    assert rng == b.rng.bit_generator.state and list(b.history) == history
    for invalid in (-.1, 1.01, float('nan'), float('inf'), True, '0.5'):
        with pytest.raises(ValueError):
            b.set_inhibition_gain(invalid)
        assert b.inhibition_gain == 1
    b.set_inhibition_gain(0.)
    b.reset()
    assert b.inhibition_gain == 1 and b.step == 0


def test_gain_applies_when_delayed_inhibitory_spike_arrives():
    b = Brain(Connectome.from_edges([100, 101], [0], [1], [-40.]))
    b.stimulate([100], 200)
    events = np.zeros((10, 1), dtype=bool)
    events[0] = True
    b.advance(.001, input_events=events)
    assert b.queue_size.sum() == 1 and b.g[1] == 0
    b.set_inhibition_gain(.25)
    b.advance(.001, input_events=np.zeros((10, 1), dtype=bool))
    assert b.g[1] == -10.  # Emission at tick 1, delivery at tick 19.


def test_timed_overlay_splits_coupled_clock_and_invalid_sequence_is_atomic():
    a, b = small_brain(), small_brain()
    session = CoupledSession(a, ClockBody())
    events = [{'at_ms': 3.7, 'action': 'inhibition_gain', 'gain': .25},
              {'at_ms': 11.3, 'action': 'release'}]
    description = {'format': 'nexus-protocol-1', 'duration_ms': 23.1, 'events': events}
    session.command('protocol', description)
    session.advance(1000)
    assert session.body.time == pytest.approx(a.time) == pytest.approx(.0231)
    assert [e['time'] for e in a.events] == [.0037, .0113]
    assert a.inhibition_gain == 1 and session.protocol.completed
    p = Protocol(description, b)
    for _ in range(3):
        p.advance(b, 100)
    np.testing.assert_array_equal(a.v, b.v)
    session.command('brain', {'kind': 'inhibition_gain', 'value': .25})
    assert session.monitor.snapshot(a, False, 0)['inhibition_gain'] == .25
    description['events'][0]['gain'] = -1
    with pytest.raises(ValueError):
        session.command('protocol', description)
    assert session.protocol is None and a.inhibition_gain == .25
    session.command('brain', {'kind': 'release'})
    assert a.inhibition_gain == 1 and a.time == .0231
