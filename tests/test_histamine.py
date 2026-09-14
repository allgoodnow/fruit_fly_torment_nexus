"""Receptor scope, causal transmission, and all-or-nothing pack activation."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from nexus.brain.runtime import Brain, Connectome, weights_filename
from nexus.datasets.histamine import POLICY_ID, select_edges, upgrade_pack
from nexus.datasets.male_cns import DATASET, FILES, digest
from nexus.datasets.male_cns_runtime import MODEL_POLICY


@pytest.fixture
def original_pack(tmp_path):
    pack, structural, raw = [tmp_path / name for name in ['pack', 'structural', 'raw']]
    for path in [pack, structural, raw]:
        path.mkdir()
    ids = np.arange(1, 7, dtype=np.int64)
    neurons = pd.DataFrame({'bodyId': ids, 'type': ['R1-R6', 'R1-R6', 'T1', 'L1', 'L2', 'L3']})
    transmitters = pd.DataFrame({'body': ids, 'consensus_nt':
                                ['histamine', 'acetylcholine', 'histamine'] + ['acetylcholine'] * 3})
    neurons.iloc[::-1].to_feather(raw / FILES['annotations'])
    transmitters.iloc[::-1].to_feather(raw / FILES['neurotransmitters'])
    graph = Connectome.from_edges(ids, [0, 0, 0, 1, 2], [3, 4, 5, 3, 4], [0, 0, 0, 2.2, 0])
    arrays = {'ids': ids, 'offsets': graph.offsets, 'posts': graph.posts.astype(np.uint32),
              'contacts': np.array([10, 20, 30, 8, 40], dtype=np.uint32)}
    for name, value in arrays.items():
        np.save(structural / f'{name}.npy', value)
    source = {'format': 'nexus-structural-connectome-1', 'dataset': DATASET,
              'sources': {'files': [{'file': FILES[name], 'sha256': digest(raw / FILES[name])}
                                    for name in ['annotations', 'neurotransmitters']]},
              'files': {f'{name}.npy': {'sha256': digest(structural / f'{name}.npy')} for name in arrays}}
    (structural / 'manifest.json').write_text(json.dumps(source))
    for name in ['ids', 'offsets', 'posts', 'weights']:
        np.save(pack / f'{name}.npy', getattr(graph, name))
    registry = {'format': 'nexus-intervention-circuits-1', 'snapshot': DATASET,
                'circuits': {'retina': {'ids': ['1']}}, 'readouts': {}}
    (pack / 'circuits.json').write_text(json.dumps(registry))
    manifest = {'format': 'nexus-connectome-1', 'snapshot': DATASET, 'experimental': True,
                'model': copy.deepcopy(MODEL_POLICY), 'source_structural_manifest': source,
                'circuit_registry': 'circuits.json',
                'files': {name: digest(pack / name) for name in
                          ['ids.npy', 'offsets.npy', 'posts.npy', 'weights.npy', 'circuits.json']}}
    (pack / 'manifest.json').write_text(json.dumps(manifest))
    return pack, structural, raw


def test_upgrade_changes_only_supported_edges_and_preserves_rollback(original_pack):
    pack, structural, raw = original_pack
    before = (pack / 'manifest.json').read_bytes()
    old = Connectome.load(pack, allow_experimental=True)
    manifest, report = upgrade_pack(pack, structural, raw)
    new = Connectome.load(pack, allow_experimental=True)
    assert new.model['id'] == POLICY_ID
    np.testing.assert_array_equal(new.weights, [-2.75, -5.5, 0, 2.2, 0])
    np.testing.assert_array_equal(old.weights, [0, 0, 0, 2.2, 0])
    assert report['changed_weights'] == 2 and report['rule']['contacts'] == 30
    assert report['rule']['postsynaptic_types'] == ['L1', 'L2']
    assert (pack / report['rollback_manifest']).read_bytes() == before
    assert weights_filename(manifest) != 'weights.npy'
    # Repeat attempts cannot compound the weights or overwrite the original.
    activated = (pack / 'manifest.json').read_bytes()
    with pytest.raises(ValueError, match='already be installed'):
        upgrade_pack(pack, structural, raw)
    assert (pack / 'manifest.json').read_bytes() == activated
    (pack / 'manifest.json').write_bytes(before)
    np.testing.assert_array_equal(Connectome.load(pack, allow_experimental=True).weights, old.weights)


@pytest.mark.parametrize('failure', ['raw', 'candidate'])
def test_failed_upgrade_keeps_active_pack(original_pack, monkeypatch, failure):
    pack, structural, raw = original_pack
    previous = (pack / 'manifest.json').read_bytes()
    old_hash = digest(pack / 'weights.npy')
    if failure == 'raw':
        (raw / FILES['annotations']).write_bytes(b'changed')
    else:
        load = Connectome.load
        def reject_candidate(path, **kwargs):
            if path != pack:
                raise ValueError('candidate rejected')
            return load(path, **kwargs)
        monkeypatch.setattr(Connectome, 'load', reject_candidate)
    with pytest.raises(ValueError, match='checksum|candidate rejected'):
        upgrade_pack(pack, structural, raw)
    assert (pack / 'manifest.json').read_bytes() == previous
    assert digest(pack / 'weights.npy') == old_hash
    assert not list(pack.glob('.histamine-*'))
    assert not list(pack.glob('weights-*.npy'))


def test_selection_requires_aligned_transmitters_and_unique_annotations():
    ids = np.array([1, 2])
    neurons = pd.DataFrame({'bodyId': ids, 'type': ['R1-R6', 'L1']})
    labels = pd.Series(['histamine', 'acetylcholine'], index=ids)
    with pytest.raises(ValueError, match='aligned'):
        select_edges(ids, np.array([0, 1, 1]), np.array([1]), neurons, labels.iloc[::-1])
    with pytest.raises(ValueError, match='Duplicate'):
        select_edges(ids, np.array([0, 1, 1]), np.array([1]), pd.concat([neurons, neurons]), labels)
    with pytest.raises(ValueError, match='Missing annotation'):
        select_edges(ids, np.array([0, 1, 1]), np.array([1]), neurons.iloc[:1], labels)
    neurons.loc[1, 'type'] = None
    assert len(select_edges(ids, np.array([0, 1, 1]), np.array([1]), neurons, labels)) == 0


def test_scanned_inhibition_has_delayed_causal_effect(original_pack):
    pack, structural, raw = original_pack
    old = Connectome.load(pack, allow_experimental=True)
    upgrade_pack(pack, structural, raw)
    new = Connectome.load(pack, allow_experimental=True)
    traces = {}
    for condition, graph in [('old', old), ('new', new), ('blocked', new)]:
        brain = Brain(graph)
        brain.stimulate([1], 100)
        if condition == 'blocked':
            brain.output_gain[0] = 0
        events = np.zeros((1000, 1), dtype=bool)
        events[0] = True
        traces[condition] = brain.advance(.1, input_events=events, trace_ids=[4, 5, 6])['voltage_mv']
    np.testing.assert_array_equal(traces['old'], traces['blocked'])
    np.testing.assert_array_equal(traces['new'][:20], traces['old'][:20])
    assert np.min(traces['new'][:, :2]) < -52.1
    np.testing.assert_array_equal(traces['new'][:, 2], traces['old'][:, 2])
