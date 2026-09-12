import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PySide6.QtWidgets import QApplication

from nexus.brain.runtime import Brain, Connectome
from nexus.brain.telemetry import NeuralTelemetry
from nexus.neural_activity import NeuralActivityPlot


def brain(size=1):
    return Brain(Connectome.from_edges(list(range(size)), [], [], []))


def test_activity_counts_every_spike_even_when_raw_raster_overflows():
    b = brain(256)
    b.stimulate(list(range(256)), 1000)
    result = b.advance(.1, input_events=np.ones((1000, 256), dtype=bool))
    assert result['unrecorded_spikes'] > 0
    t = NeuralTelemetry(b).snapshot(b, False, 0)
    activity = t['population_activity']
    assert sum(activity['spikes']) == int(b.counts.sum())
    assert len(t['raster_steps']) == 2500
    assert activity['end_steps'] == list(range(100, 1001, 100))
    assert len(activity['spikes']) == 10
    assert all(x > 0 for x in activity['spikes'])
    np.testing.assert_allclose(activity['rates_hz_per_neuron'], np.array(activity['spikes'])/(.01*256))


def test_activity_uses_fixed_bins_across_chunks_and_partial_intervals():
    a, b = brain(), brain()
    for item in [a, b]:
        item.stimulate([0], 1000)
    a.advance(.0237, input_events=np.ones((237, 1), dtype=bool))
    for length in [37, 66, 134]:
        b.advance(length*.0001, input_events=np.ones((length, 1), dtype=bool))
    def activity(item):
        return NeuralTelemetry(item).snapshot(item, False, 0)['population_activity']
    x, y = activity(a), activity(b)
    assert x == y and x['end_steps'] == [100, 200, 237]
    np.testing.assert_allclose(x['durations_ms'], [10, 10, 3.7])
    assert sum(x['spikes']) == a.counts.sum() == b.counts.sum()


def test_silent_bins_scroll_and_pause_does_not_add_or_erase_data():
    b = brain()
    b.stimulate([0], 1000)
    b.advance(.01)
    monitor = NeuralTelemetry(b)
    before = monitor.snapshot(b, False, 0)['population_activity']
    assert monitor.snapshot(b, False, 0)['population_activity'] == before
    b.release()
    for _ in range(6):
        b.advance(1.)
    activity = monitor.snapshot(b, False, 0)['population_activity']
    assert len(activity['spikes']) == 500 and not any(activity['spikes'])
    assert activity['end_steps'][0] == 10200 and activity['end_steps'][-1] == 60100
    b.reset()
    monitor.reset(b)
    assert monitor.snapshot(b, False, 1)['population_activity']['spikes'] == []


def test_native_plot_shows_full_activity_history_holds_on_pause_and_clears_on_reset():
    app = QApplication.instance() or QApplication([])
    plot = NeuralActivityPlot()
    b = brain()
    b.stimulate([0], 1000)
    b.advance(.08)
    monitor = NeuralTelemetry(b)
    packet = monitor.snapshot(b, False, 0)
    plot.update_snapshot(packet)
    x, y = plot.curve.getData()
    np.testing.assert_allclose(x, np.arange(1, 9)*.01)
    assert len(y) == 8 and np.any(y > 0)
    saved = x.copy(), y.copy()
    plot.update_snapshot(monitor.snapshot(b, False, 0))
    for actual, expected in zip(plot.curve.getData(), saved):
        np.testing.assert_array_equal(actual, expected)
    b.reset()
    monitor.reset(b)
    plot.update_snapshot(monitor.snapshot(b, False, 1))
    empty, _ = plot.curve.getData()
    assert empty is None or len(empty) == 0
    assert plot.viewRange()[0] == [0., 1.]
    plot.close()
