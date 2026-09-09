import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from nexus.brain.anatomy import Anatomy
from nexus.brain.targets import SUGAR, MN9


def test_release_coordinates_match_model_and_documented_voxel_scale():
    pack = Path(__file__).resolve().parents[1] / 'data/brain-v630'
    anatomy = Anatomy(pack)
    assert len(anatomy.ids) == 127400
    assert anatomy.valid.sum() == 127322
    assert all(anatomy.valid[anatomy.lookup[int(root)]] for root in SUGAR + [MN9])
    index = anatomy.lookup[720575940628857210]
    # Source anchor (109306, 50491, 3960), scaled then rigidly displayed (x,z,-y).
    np.testing.assert_allclose(anatomy.positions[index] + anatomy.centre,
                               [437.224, 158.4, -201.964], atol=.0001)


def test_activity_tracks_simulation_time_and_reports_unlocated_cells():
    pack = Path(__file__).resolve().parents[1] / 'data/brain-v630'
    anatomy = Anatomy(pack)
    known = int(np.flatnonzero(anatomy.valid)[0])
    missing = int(np.flatnonzero(~anatomy.valid)[0])
    positions, colors, sizes, unlocated = anatomy.activity(2000, [known, missing], [1950, 1980])
    assert len(positions) == 1 and unlocated == 1
    np.testing.assert_allclose(positions[0], anatomy.positions[known])
    repeat = anatomy.activity(2000, [known, missing], [1950, 1980])
    np.testing.assert_array_equal(colors, repeat[1])
    assert not len(anatomy.activity(4000, [known], [1950])[0])
    assert not len(anatomy.activity(0, [], [])[0])
    with pytest.raises(ValueError):
        anatomy.activity(2000, [-1], [1950])


def test_rejects_anatomy_with_wrong_index_order(tmp_path):
    np.save(tmp_path / 'ids.npy', np.array([101, 102], dtype=np.int64))
    path = tmp_path / 'anatomy.npz'
    np.savez(path, ids=np.array([102, 101], dtype=np.int64),
             positions_um=np.zeros((2, 3), dtype=np.float32), valid=np.ones(2, dtype=bool))
    (tmp_path / 'anatomy-manifest.json').write_text(json.dumps({
        'snapshot': '630', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))
    with pytest.raises(ValueError, match='neuron order'):
        Anatomy(tmp_path)
