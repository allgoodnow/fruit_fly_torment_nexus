import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('ablate_phototransduction_timing')


def test_event_area_ignores_zero_duration_states_and_samples_last_event(audit):
    # A channel count of 27 at 20 ms is superseded at exactly the same time.
    # Only the final count of 3 persists over the following two milliseconds.
    events = np.array([[.1, 0], [20, 27], [20, 3], [22, 0]], dtype=float)
    result = audit.event_metrics(events)
    assert result['event_integrated_channel_ms'] == 6
    assert result['zero_time_transitions'] == 1
    assert result['causal_1ms_grid']['channel_area_channel_ms'] == 6
    assert result['causal_1ms_grid']['peak_open_channels'] == 3
    assert result['causal_1ms_grid']['sampled_latency_ms'] == 0


def test_event_area_preserves_fractional_waits_and_final_interval(audit):
    result = audit.event_metrics(np.array([[.1, 0], [20.25, 2], [299.5, 1]]))
    assert result['event_integrated_channel_ms'] == 559
    assert result['zero_time_transitions'] == 0
    assert result['causal_1ms_grid']['sampled_latency_ms'] == 1


@pytest.mark.parametrize('events', [[], [[2, 1], [1, 2]], [[-1, 0]],
                                  [[301, 0]], [[1, 28]], [[1, float('nan')]]])
def test_event_audit_rejects_invalid_history(audit, events):
    with pytest.raises(ValueError, match='Invalid source event trace'):
        audit.event_metrics(events)
