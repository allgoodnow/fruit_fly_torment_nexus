import importlib
from pathlib import Path

import numpy as np
from scipy.io import savemat

from nexus.brain.phototransduction import Phototransduction


def test_common_grid_comparison_does_not_introduce_a_one_ms_phase_shift(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    audit = importlib.import_module('compare_phototransduction_timing')
    monkeypatch.setattr(audit, 'CONDITIONS', {'single': [20]})
    # Same rectangular channel response: 20 <= t < 23 ms.
    source = np.zeros(300)
    source[19:22] = 3
    native = np.zeros(3000, dtype=int)
    native[200:230] = 3
    ll, photons = np.zeros(300), np.zeros(300, dtype=int)
    ll[19], photons[20] = 1, 1
    yy = np.zeros((300, 15))
    yy[19:, 3] = 1
    savemat(tmp_path/'single-1-octave.mat',
            {'channels': source, 'LL': ll, 'seed': 1, 'yy': yy, 'output_rows': 300})
    np.savez(tmp_path/'single-1-python.npz', open_channels=native, absorbed_photons=photons)
    rows = audit.read_comparisons(tmp_path, [1])
    for row in rows:
        assert row['sampled_latency_ms'] == 0
        assert row['channel_area_channel_ms'] == 9
        assert row['peak_open_channels'] == 3
    assert not rows[0]['rhodopsin_exceeds_supplied_photons']
    yy[20, 3] = 2
    savemat(tmp_path/'single-1-octave.mat',
            {'channels': source, 'LL': ll, 'seed': 1, 'yy': yy, 'output_rows': 300})
    assert audit.read_comparisons(tmp_path, [1])[0]['rhodopsin_exceeds_supplied_photons']


def test_audit_preserves_nonresponders_in_denominator(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    audit = importlib.import_module('compare_phototransduction_timing')
    monkeypatch.setattr(audit, 'CONDITIONS', {'single': [20]})
    rows = []
    for model in ('upstream_octave', 'python_continuous'):
        for trace in ([0, 0, 0], [0, 3, 0]):
            rows.append({'condition': 'single', 'model': model,
                         **audit.summarize(trace, np.array([20, 21, 22]), 20)})
    for group in audit.aggregate(rows)['single'].values():
        assert group['trials'] == 2 and group['responded'] == 1
        assert group['peak_open_channels']['mean'] == 1.5
        assert group['sampled_latency_ms']['mean'] == 1


def test_python_delivers_each_photon_once_at_integer_boundaries():
    for arrivals in ([20], [20, 40], [20, 120]):
        receptor = Phototransduction(microvilli=1, seed=77121)
        supplied = 0
        for ms in range(150):
            count = int(ms in arrivals)
            supplied += count
            receptor.advance([count])
            # Rh* can only be created by the supplied photons. Removal may
            # reduce it, but no scheduling path may create a second activation.
            assert receptor.states[0, 3] <= supplied
            assert receptor.photons == supplied
        assert receptor.photons == len(arrivals)
