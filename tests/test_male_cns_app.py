"""Native integration boundaries: anatomy, dataset selection and shared-clock inputs."""
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pytest

from nexus.brain.anatomy import Anatomy
from nexus.brain.config import model_config, default_pack
from nexus.brain.protocol import Protocol
from nexus.brain.runtime import Brain, Connectome
from nexus.brain.scenarios import scenario_protocol
from nexus.coupled import CoupledSession
from nexus.environment import FoodEnvironment
from test_coupled import ClockBody
from test_male_cns_runtime import pack

ROOT = Path(__file__).resolve().parents[1]


def test_native_default_is_male_cns_and_old_reference_remains_selectable():
    assert default_pack().name == 'brain-male-cns-v1.0-lif'
    old = model_config(default_pack('flywire-v630'))
    assert old.snapshot == '630' and old.food_available and len(old.first_ids) == 21


def test_ui_configuration_uses_active_specimen_and_bilateral_readouts(pack):
    config = model_config(pack)
    assert config.experimental and not config.food_available
    assert config.first_ids == ('102',)
    assert config.left_ids == ('107',) and config.right_ids == ('108',)
    assert config.mn9_ids == ('109', '110')
    assert '1)' in config.first_label


def test_anchor_source_precedence_units_and_no_fabricated_missing_positions():
    anchor_arrays = runpy.run_path(str(ROOT/'scripts/prepare_male_cns_anatomy.py'))['anchor_arrays']
    frame = pd.DataFrame({'bodyId': [1, 2, 3],
                          'somaLocation': [[100, 200, 300], None, None],
                          'tosomaLocation': [[999, 999, 999], [400, 500, 600], None]})
    xyz, kinds = anchor_arrays(frame, np.array([1, 2, 3]))
    np.testing.assert_allclose(xyz[:2], [[.8, 1.6, 2.4], [3.2, 4., 4.8]])
    assert kinds.tolist() == [1, 2, 0] and np.isnan(xyz[2]).all()
    with pytest.raises(ValueError, match='order'):
        anchor_arrays(frame, np.array([3, 2, 1]))
    frame.at[1, 'tosomaLocation'] = [1, float('nan'), 3]
    with pytest.raises(ValueError, match='Invalid coordinate'):
        anchor_arrays(frame, np.array([1, 2, 3]))


def test_real_male_anatomy_matches_runtime_order_and_counts_unlocated_activity():
    anatomy = Anatomy(default_pack())
    assert anatomy.manifest['snapshot'] == 'male-cns:v1.0'
    assert len(anatomy.ids) == 166700 and anatomy.valid.sum() == 140638
    raw = pd.read_feather(ROOT/'data/raw/male-cns-v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather').set_index('bodyId')
    anchor = np.asarray(raw.loc[10941, 'somaLocation'], dtype=float)*.008
    np.testing.assert_allclose(anatomy.positions[anatomy.lookup[10941]]+anatomy.centre,
                               anchor[[0, 2, 1]]*[1, 1, -1], atol=.0001)
    absent = int(np.flatnonzero(~anatomy.valid)[0])
    assert anatomy.activity(10, [absent], [9])[3] == 1


def test_anatomy_rejects_foreign_snapshot_even_with_the_same_ids(tmp_path):
    ids = np.array([101, 102], dtype=np.int64)
    np.save(tmp_path/'ids.npy', ids)
    np.savez(tmp_path/'anatomy.npz', ids=ids, positions_um=np.ones((2, 3)), valid=np.ones(2, dtype=bool))
    from nexus.datasets.male_cns import digest
    (tmp_path/'anatomy-manifest.json').write_text(json.dumps({'snapshot': '630', 'sha256': digest(tmp_path/'anatomy.npz')}))
    (tmp_path/'manifest.json').write_text(json.dumps({'snapshot': 'male-cns:v1.0'}))
    with pytest.raises(ValueError, match='another dataset'):
        Anatomy(tmp_path)


def test_unmapped_food_remains_physical_and_cannot_inject_foreign_ids(pack):
    brain = Brain(Connectome.load(pack, allow_experimental=True))
    env = FoodEnvironment(center=(0, 0), targets=[])
    env.sample([{'foot': 'lf', 'position_mm': [0, 0, 0]}], brain)
    assert env.present and env.contact and not env.active and not brain.inputs.size
    before = env.snapshot()
    with pytest.raises(ValueError, match='mapping is unresolved'):
        env.configure({'enabled': True, 'present': False}, 0)
    assert env.snapshot() == before


def test_male_overload_protocol_has_mapped_inputs_and_preserves_shared_clock(pack):
    brain = Brain(Connectome.load(pack, allow_experimental=True))
    description = scenario_protocol('seizure', dataset=brain.graph.snapshot,
                                    baseline_ms=3.7, stimulus_ms=11.3, recovery_ms=8.1)
    assert description['events'][1]['action'] == 'circuit'
    body = ClockBody()
    session = CoupledSession(brain, body)
    session.command('protocol', description)
    session.advance(1000)
    assert body.time == pytest.approx(brain.time) == pytest.approx(.0231)
    assert not brain.inputs.size and brain.inhibition_gain == 1
    description['dataset'] = '630'
    before = brain.v.copy()
    with pytest.raises(ValueError, match='another neural dataset'):
        Protocol(description, brain)
    np.testing.assert_array_equal(brain.v, before)
