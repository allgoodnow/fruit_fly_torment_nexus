"""Show measured subthreshold state without inventing spikes or anatomy."""
import copy
import os
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from nexus.brain.runtime import Brain, Connectome
from nexus.brain.telemetry import NeuralTelemetry
from nexus.brain.anatomy import Anatomy
from nexus.brain.view import BrainView

PACK = Path(__file__).resolve().parents[1] / 'data/brain-v630'


def test_voltage_telemetry_reveals_nonspiking_synaptic_response_without_changing_state():
    graph = Connectome.from_edges([1, 2, 3], [0, 0], [1, 2], [-10., 10.])
    brain = Brain(graph)
    brain.stimulate([1], 100)
    inputs = np.zeros((100, 1), dtype=bool)
    inputs[0] = True
    brain.advance(.01, input_events=inputs)
    assert brain.counts.tolist() == [1, 0, 0]
    before = brain.v.copy(), brain.counts.copy(), copy.deepcopy(brain.rng.bit_generator.state), brain.step
    monitor = NeuralTelemetry(brain)
    packet = monitor.snapshot(brain, False, 0)
    values = dict(zip(packet['membrane_activity']['indices'], packet['membrane_activity']['delta_mv']))
    assert values[1] < -.5 and values[2] > .5
    assert 1 not in packet['active_indices'] and 2 not in packet['active_indices']
    assert packet['membrane_activity'] == monitor.snapshot(brain, False, 0)['membrane_activity']
    brain.release()
    assert monitor.snapshot(brain, False, 0)['membrane_activity'] == packet['membrane_activity']
    np.testing.assert_array_equal(brain.v, before[0])
    np.testing.assert_array_equal(brain.counts, before[1])
    assert brain.rng.bit_generator.state == before[2] and brain.step == before[3]
    brain.reset()
    assert monitor.snapshot(brain, False, 1)['membrane_activity']['indices'] == []


def test_voltage_positions_use_only_known_anchors_and_signed_fixed_colors():
    anatomy = Anatomy(PACK)
    left, right = np.flatnonzero(anatomy.valid)[:2]
    missing = np.flatnonzero(~anatomy.valid)[0]
    positions, colors, sizes, absent = anatomy.voltage_activity([left, right, missing], [-2., 2., -100.])
    np.testing.assert_array_equal(positions, anatomy.positions[[left, right]])
    assert absent == 1 and colors[0, 1] > colors[1, 1]
    assert colors[0, 0] > colors[0, 2] and colors[1, 0] > colors[1, 2]
    assert sizes[0] == sizes[1]
    assert anatomy.voltage_activity([left], [-100.])[2][0] == anatomy.voltage_activity([left], [-5.])[2][0]
    with pytest.raises(ValueError):
        anatomy.voltage_activity([left], [float('nan')])


def test_native_view_mode_pause_reset_and_wrong_neuron_order_clear_voltage():
    app = QApplication.instance() or QApplication([])
    view = BrainView(PACK)
    index = int(np.flatnonzero(view.anatomy.valid)[0])
    packet = {'dataset': '630', 'neuron_order_sha256': view.anatomy.neuron_order_sha256,
              'tick': 100, 'active_indices': [], 'active_steps': [], 'stimulated_ids': [],
              'membrane_activity': {'indices': [index], 'delta_mv': [-3.]}}
    view.update_snapshot(packet)
    assert view.voltage_count == 1 and view.active_count == 0
    colors = view.voltage.color.copy()
    view.update_snapshot(packet)
    np.testing.assert_array_equal(view.voltage.color, colors)
    view.activity_mode.setCurrentIndex(1)
    assert view.voltage_count == 0 and len(view.voltage.pos) == 0
    view.activity_mode.setCurrentIndex(0)
    assert view.voltage_count == 1
    view.update_snapshot(dict(packet, neuron_order_sha256='wrong'))
    assert view.voltage_count == 0 and not len(view.voltage.pos)
    view.update_snapshot(dict(packet, tick=0, membrane_activity={'indices': [], 'delta_mv': []}))
    assert not len(view.voltage.pos)
    view.close()
