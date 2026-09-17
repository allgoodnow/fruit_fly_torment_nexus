import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def assay(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('verify_phototransduction_recovery')


def test_recovery_subtracts_first_response_tail_without_clipping(assay):
    result = assay.areas([2, 1, 0], [3, 1, 1], [0, 4, 0], dt_ms=1)
    assert result['incremental_channel_ms'] == -2
    assert result['rested_probe_channel_ms'] == 4


def test_recovery_ratio_keeps_nonresponders_and_uses_group_means(assay):
    rows = [{'incremental_channel_ms': effect, 'rested_probe_channel_ms': fresh,
             'no_probe_channel_ms': 0, 'pre_probe_state': [0]*8}
            for effect, fresh in [(0, 0), (1, 2), (5, 10)]]
    result = assay.summarize(rows)
    assert result['trials'] == 3
    assert result['mean_incremental_channel_ms'] == 2
    assert result['recovery_ratio_of_means'] == .5
    # All-zero bootstrap samples are possible; do not silently drop them.
    assert result['undefined_ratio_resamples'] > 0
    assert result['recovery_ratio_bootstrap_95pct'] is None


def test_recovery_assay_is_reproducible_and_retains_molecular_memory(assay):
    a, traces_a = assay.trial(79100, 50)
    b, traces_b = assay.trial(79100, 50)
    assert a == b
    for key in traces_a:
        np.testing.assert_array_equal(traces_a[key], traces_b[key])
    # Prior activity survives darkness; recovery is not a reset or GUI timer.
    assert a['first_prefix_channel_ms'] > 0
    assert a['pre_probe_state'][6] > 0  # Bound calmodulin retains feedback.
    assert a['pre_probe_state'][7] < 50  # Available G pool has not fully recovered.
