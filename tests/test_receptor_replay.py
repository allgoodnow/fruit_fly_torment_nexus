from pathlib import Path

import numpy as np

from nexus.brain.receptor_release import ReleaseCurve
from nexus.brain.telemetry import membrane_activity
from test_medulla_relays import model


def replay_model():
    brain = model()
    # The pulse fixture uses -600 source edges, which shut off relay release
    # even at the assumed dark tonic rate. Use -60 in this synthetic assay.
    for root in (13, 14):
        pre = brain.lookup[root]
        a, b = brain.graph.offsets[pre:pre+2]
        brain.graph.weights[a:b] = -60.
    return brain


def test_voltage_release_replay_crosses_relays_without_source_spikes():
    curve = ReleaseCurve.load(Path(__file__).resolve().parents[1] /
                              'experiments/receptor-release-assay-curve.json')
    # 30 ms reference voltage, then ordinary 10 mV depolarization. No pulses.
    dark = np.full((1500, 2), -70.)
    light = dark.copy()
    light[300:] = -60.
    runs = {}
    for name, voltage, blocked in [('dark', dark, False), ('light', light, False),
                                   ('dark_blocked', dark, True), ('light_blocked', light, True)]:
        brain = replay_model()
        brain.configure_receptor_replay(['13', '14'], reference_mv=curve.reference_mv)
        assert 12 not in membrane_activity(brain)['indices']
        if blocked:
            brain.silence(['13', '14'])
        result = brain.advance(.15, trace_ids=['13', '1', '5', '15'],
                               receptor_voltage_mv=voltage, receptor_release_hz=curve.evaluate(voltage))
        assert not brain.counts[brain.receptor_indices].any()
        assert not brain.counts[brain.graded_indices].any()
        np.testing.assert_array_equal(result['voltage_mv'][:, 0], voltage[:, 0])
        runs[name] = (brain, result['voltage_mv'])
    light_brain, light_trace = runs['light']
    _, dark_trace = runs['dark']
    difference = light_trace[:, 1:] - dark_trace[:, 1:]
    np.testing.assert_array_equal(difference[:328], 0.)  # 1 ms sample + 1.8 ms delivery + integration.
    assert np.max(np.abs(difference[:, 0])) > .5
    assert np.max(np.abs(difference[:, 1])) > .1
    assert np.max(np.abs(difference[:, 2])) > .01
    np.testing.assert_array_equal(runs['dark_blocked'][1][:, 1:], runs['light_blocked'][1][:, 1:])
    values = dict(zip(membrane_activity(light_brain)['indices'], membrane_activity(light_brain)['delta_mv']))
    assert values[12] == 10.  # Relative to declared receptor reference, not LIF rest.

    chunked = replay_model()
    chunked.configure_receptor_replay(['13', '14'], reference_mv=curve.reference_mv)
    cursor = 0
    for size in (303, 497, 700):
        part = light[cursor:cursor+size]
        chunked.advance(size*.0001, receptor_voltage_mv=part, receptor_release_hz=curve.evaluate(part))
        cursor += size
    for key in ('v', 'g', 'histamine_g', 'excitation_g', 'counts', 'receptor_queue', 'graded_queue'):
        np.testing.assert_array_equal(getattr(chunked, key), getattr(light_brain, key))
    chunked.reset()
    assert not chunked.receptor_mask.any() and not len(chunked.receptor_indices)
    chunked.advance(.01)
    np.testing.assert_array_equal(chunked.v, -52.)
