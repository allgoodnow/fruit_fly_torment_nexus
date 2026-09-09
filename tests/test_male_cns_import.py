"""Regression checks for specimen selection and streamed structural graph import."""
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from nexus.datasets.male_cns import (FILES, audit_and_prepare, classified_neurons,
                                    match_edges, neurotransmitter_audit,
                                    type_crosswalk, verify_sources, write_csr_batch)
from nexus.brain.runtime import Connectome


def annotations():
    return pd.DataFrame({
        'bodyId': [30, 10, 20, 999],
        'superclass': ['ascending_neuron', 'cb_intrinsic', 'cb_intrinsic', None],
        'status': [None, 'Traced', 'Anchor', 'Traced'],
        'type': ['ANtest', 'GNG121', 'DNp01', None],
        'flywireType': [None, 'CB0059', 'DNp01', None],
        'somaSide': ['L', 'L', 'R', None], 'rootSide': [None]*4,
        'mancBodyid': [123, None, 456, None], 'mancType': ['ANtest', None, 'DNp01', None],
        'class': [None]*4, 'subclass': [None]*4, 'supertype': [None]*4, 'synonyms': [None]*4,
        'somaLocation': [[1, 2, 3], None, None, None],
        'tosomaLocation': [None, [4, 5, 6], None, None],
    })


def test_selection_preserves_classified_nontraced_cells_and_excludes_unclassified_traced():
    selected = classified_neurons(annotations())
    assert selected.bodyId.tolist() == [10, 20, 30]
    duplicated = pd.concat([annotations(), annotations().iloc[:1]])
    with pytest.raises(ValueError, match='unique'):
        classified_neurons(duplicated)


def test_endpoint_filter_does_not_snap_unknown_ids_to_neighboring_neurons():
    ids = np.array([10, 20, 30])
    pre = np.array([0, 10, 11, 20, 30, 999, 30])
    post = np.array([10, 30, 20, 21, 10, 10, 999])
    p, q, weights = match_edges(ids, pre, post, np.arange(1, 8))
    assert list(zip(p, q, weights)) == [(0, 2, 2), (2, 0, 5)]


@pytest.mark.parametrize('contacts', [np.array([0]), np.array([-1]), np.array([2**32]), np.array([1.5])])
def test_invalid_contacts_are_not_silently_cast(contacts):
    with pytest.raises(ValueError):
        match_edges(np.array([10]), np.array([10]), np.array([10]), contacts)


def test_csr_scatter_preserves_rows_across_interleaved_batches():
    cursor = np.array([0, 2, 3])
    posts, contacts = np.zeros(6, dtype=np.uint32), np.zeros(6, dtype=np.uint32)
    write_csr_batch(cursor, posts, contacts, np.array([2, 0, 2]), np.array([0, 2, 1]), np.array([8, 3, 9]))
    write_csr_batch(cursor, posts, contacts, np.array([1, 2, 0]), np.array([2, 2, 1]), np.array([7, 5, 4]))
    write_csr_batch(cursor, posts, contacts, np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int))
    assert cursor.tolist() == [2, 3, 6]
    assert posts.tolist() == [2, 1, 2, 0, 1, 2]
    assert contacts.tolist() == [3, 4, 7, 8, 9, 5]


def test_crosswalk_uses_new_body_ids_and_keeps_uncertainty():
    selected = classified_neurons(annotations())
    match = type_crosswalk(selected)['aversion_candidate']
    assert match['ids'] == [10]
    assert match['annotation_matches'][0]['type'] == 'GNG121'
    table = pd.DataFrame({'body': [10, 20, 999], 'consensus_nt': ['GABA', 'unclear', 'ACh']})
    audit = neurotransmitter_audit(selected, table)
    assert audit['consensus_counts'] == {'GABA': 1, 'unclear': 1, 'missing': 1}
    assert audit['missing_table_rows'] == 1
    with pytest.raises(ValueError, match='Duplicate'):
        neurotransmitter_audit(selected, pd.concat([table, table.iloc[:1]]))


def test_two_pass_import_matches_independent_edge_list_and_runtime_rejects_it(tmp_path):
    raw, output = tmp_path / 'raw', tmp_path / 'pack'
    raw.mkdir()
    annotations().to_feather(raw / FILES['annotations'])
    pd.DataFrame({'body': [10, 20, 30], 'consensus_nt': ['GABA', 'ACh', None]}).to_feather(raw / FILES['neurotransmitters'])
    schema = pa.schema([(name, pa.int64()) for name in ['body_pre', 'body_post', 'weight']])
    batches = [([30, 999, 10], [10, 10, 20], [5, 100, 3]),
               ([20, 30, 10], [30, 20, 11], [7, 2, 300])]
    with pa.OSFile(str(raw / FILES['connectivity']), 'wb') as stream, pa.ipc.new_file(stream, schema) as writer:
        for columns in batches:
            writer.write_batch(pa.record_batch(list(columns), schema=schema))
    report = audit_and_prepare(raw, output, {'dataset': 'synthetic test'}, expected_neurons=3)
    ids = np.load(output / 'ids.npy')
    offsets = np.load(output / 'offsets.npy')
    posts = np.load(output / 'posts.npy')
    contacts = np.load(output / 'contacts.npy')
    restored = [(int(ids[i]), int(ids[posts[j]]), int(contacts[j]))
                for i in range(len(ids)) for j in range(offsets[i], offsets[i+1])]
    assert restored == [(10, 20, 3), (20, 30, 7), (30, 10, 5), (30, 20, 2)]
    assert report['raw_connection_rows'] == 6
    assert report['retained_contact_count'] == 17
    assert report['retained_connection_rows'] == 4
    assert report['anatomy']['soma_or_tosoma_present'] == 2
    assert report['neurotransmitters']['missing_consensus'] == 1
    assert report['neurotransmitters']['missing_table_rows'] == 0
    assert len(report['nociception_candidates']['ascending_to_target_edges']) == 2
    assert report['nociception_candidates']['driver_label_matches']['SS01159'] == []
    with pytest.raises(ValueError, match='Unsupported brain pack'):
        Connectome.load(output)
    with pytest.raises(FileExistsError):
        audit_and_prepare(raw, output, {}, expected_neurons=3)
    assert json.loads((output / 'manifest.json').read_text())['runtime_compatible'] is False


def test_source_verification_rejects_foreign_dataset_before_reading_tables(tmp_path):
    with pytest.raises(ValueError, match='another dataset'):
        verify_sources(tmp_path, {'dataset': 'FlyWire v630'})


def test_source_verification_catches_tampering(tmp_path):
    from nexus.datasets.male_cns import BASE, DATASET, digest
    records = []
    for name in FILES.values():
        path = tmp_path / name
        path.write_bytes(b'fixture')
        records.append({'file': name, 'url': BASE + name, 'bytes': path.stat().st_size, 'sha256': digest(path)})
    manifest = {'dataset': DATASET, 'files': records}
    verify_sources(tmp_path, manifest)
    (tmp_path / FILES['annotations']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify_sources(tmp_path, manifest)
