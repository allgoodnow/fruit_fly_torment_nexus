"""Audit MaleCNS v1.0 and preserve unsigned contacts in a separate structural pack.

This is not a fitted dynamical model. No neurotransmitter-to-sign conversion,
voltage scaling, behavioral function assignment, or FlyWire ID reuse occurs here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request

import numpy as np

DATASET = 'male-cns:v1.0'
SOURCE = 'https://male-cns.janelia.org/download/'
BASE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
FILES = {
    'annotations': 'body-annotations-male-cns-v1.0-minconf-0.5.feather',
    'neurotransmitters': 'body-neurotransmitters-male-cns-v1.0.feather',
    'connectivity': 'connectome-weights-male-cns-v1.0-minconf-0.5.feather',
}
SELECTORS = {
    'looming_type': ('type', 'LPLC2'),
    'warmth_type': ('type', 'TRN_VP2'),
    'aversion_candidate': ('flywireType', 'CB0059'),
    'giant_fiber_type': ('type', 'DNp01'),
    'steering_type': ('type', 'DNa02'),
    'walking_type': ('type', 'DNg100'),
    'feeding_motor_type': ('type', 'MN9'),
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_sources(directory, manifest, download=False):
    """Verify pinned hashes; optional downloads never replace existing files."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if manifest['dataset'] != DATASET:
        raise ValueError('Source manifest is for another dataset')
    records = {r['file']: r for r in manifest['files']}
    if set(records) != set(FILES.values()) or len(manifest['files']) != len(FILES):
        raise ValueError('Source manifest must contain exactly the three official tables')
    for name in FILES.values():
        record = records[name]
        path = directory / name
        url = BASE + name
        if record['url'] != url:
            raise ValueError(f'Unexpected source URL: {name}')
        if not path.exists() and download:
            part = path.with_suffix(path.suffix + '.part')
            with urllib.request.urlopen(url, timeout=120) as response, part.open('wb') as out:
                shutil.copyfileobj(response, out)
            if part.stat().st_size != record['bytes'] or digest(part) != record['sha256']:
                raise ValueError(f'Download failed checksum verification: {name}')
            part.replace(path)
        if not path.exists():
            raise FileNotFoundError(f'{path}; pass --download to fetch pinned official data')
        if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError(f'Source checksum mismatch: {name}')


def classified_neurons(annotations):
    neurons = annotations.loc[annotations.superclass.notna()].sort_values('bodyId').copy()
    ids = neurons.bodyId.to_numpy()
    if (not len(ids) or ids.dtype.kind not in 'iu' or (ids <= 0).any()
            or neurons.bodyId.duplicated().any()):
        raise ValueError('Classified body IDs must be unique positive integers')
    return neurons.reset_index(drop=True)


def match_edges(ids, pre, post, contacts):
    """Keep edges whose two endpoints are classified; return local indices."""
    if (pre.dtype.kind not in 'iu' or post.dtype.kind not in 'iu'
            or contacts.dtype.kind not in 'iu' or not (len(pre) == len(post) == len(contacts))):
        raise ValueError('Malformed connectivity columns')
    if len(contacts) and ((contacts <= 0).any() or contacts.max() > np.iinfo(np.uint32).max):
        raise ValueError('Contact counts must be positive uint32-compatible integers')
    pi, qi = np.searchsorted(ids, pre), np.searchsorted(ids, post)
    keep = ((pi < len(ids)) & (qi < len(ids))
            & (ids[np.minimum(pi, len(ids)-1)] == pre)
            & (ids[np.minimum(qi, len(ids)-1)] == post))
    return pi[keep], qi[keep], contacts[keep]


def edge_batches(path, ids):
    import pyarrow as pa
    with pa.memory_map(str(path), 'r') as stream:
        reader = pa.ipc.open_file(stream)
        if reader.schema.names != ['body_pre', 'body_post', 'weight']:
            raise ValueError('Unexpected connectivity schema')
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            columns = [batch.column(i).to_numpy(zero_copy_only=False) for i in range(3)]
            yield len(columns[0]), match_edges(ids, *columns)


def write_csr_batch(cursor, posts, weights, pre, post, contacts):
    """Append a batch to preallocated CSR rows, preserving source row order."""
    if not len(pre):
        return
    order = np.argsort(pre, kind='stable')
    ordered = pre[order]
    unique, starts, counts = np.unique(ordered, return_index=True, return_counts=True)
    destinations = cursor[ordered] + np.arange(len(pre)) - np.repeat(starts, counts)
    posts[destinations] = post[order]
    weights[destinations] = contacts[order]
    cursor[unique] += counts


def type_crosswalk(neurons):
    result = {}
    fields = ['bodyId', 'type', 'flywireType', 'somaSide', 'rootSide', 'mancBodyid', 'mancType']
    for name, (column, value) in SELECTORS.items():
        cells = neurons.loc[neurons[column].eq(value)]
        result[name] = {
            'selector': {column: value}, 'count': len(cells),
            'ids': cells.bodyId.astype(int).tolist(),
            'annotation_matches': json.loads(cells[fields].to_json(orient='records')),
            'status': 'annotation correspondence only; not functional validation',
        }
    return result


def neurotransmitter_audit(neurons, table):
    matched = table.loc[table.body.isin(neurons.bodyId)]
    if matched.body.duplicated().any():
        raise ValueError('Duplicate classified body IDs in neurotransmitter table')
    joined = neurons[['bodyId']].merge(matched[['body', 'consensus_nt']],
                                     left_on='bodyId', right_on='body', how='left', validate='one_to_one')
    counts = {str(k): int(v) for k, v in joined.consensus_nt.fillna('missing').value_counts().items()}
    return {
        'consensus_counts': counts,
        'missing_table_rows': int(joined.body.isna().sum()),
        'missing_consensus': int(joined.consensus_nt.isna().sum()),
        'sign_policy': 'No signs assigned. Identity alone does not specify all postsynaptic effects.',
    }


def audit_and_prepare(raw, output, source_manifest, expected_neurons=166700):
    import pandas as pd
    raw, output = Path(raw), Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite structural pack: {output}')
    annotations = pd.read_feather(raw / FILES['annotations'])
    neurons = classified_neurons(annotations)
    ids = neurons.bodyId.to_numpy(dtype=np.int64)
    if len(ids) != expected_neurons:
        raise ValueError(f'Expected {expected_neurons} classified neurons; got {len(ids)}')
    crosswalk = type_crosswalk(neurons)
    nt = neurotransmitter_audit(neurons, pd.read_feather(raw / FILES['neurotransmitters']))
    targets = {body for name in ('aversion_candidate', 'giant_fiber_type', 'walking_type')
               for body in crosswalk[name]['ids']}
    target_indices = np.flatnonzero(np.isin(ids, list(targets)))
    ascending = neurons.superclass.isin(['ascending_neuron', 'sensory_ascending']).to_numpy()
    degree = np.zeros(len(ids), dtype=np.int64)
    incoming = np.zeros(len(ids), dtype=np.int64)
    raw_edges = edges = contacts_total = self_edges = 0
    candidate_edges = []
    for batch_index, (raw_count, (pre, post, contacts)) in enumerate(edge_batches(raw / FILES['connectivity'], ids)):
        raw_edges += raw_count
        edges += len(pre)
        contacts_total += int(contacts.sum(dtype=np.int64))
        self_edges += int((pre == post).sum())
        degree += np.bincount(pre, minlength=len(ids))
        incoming += np.bincount(post, minlength=len(ids))
        candidate = ascending[pre] & np.isin(post, target_indices)
        candidate_edges.extend(zip(pre[candidate].tolist(), post[candidate].tolist(), contacts[candidate].tolist()))
        if batch_index % 400 == 0:
            print(f'Audit batch {batch_index}: {edges:,} retained connections', flush=True)
    offsets = np.concatenate(([0], degree.cumsum()))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.partial-', dir=output.parent))
    try:
        np.save(staging / 'ids.npy', ids, allow_pickle=False)
        np.save(staging / 'offsets.npy', offsets, allow_pickle=False)
        posts = np.lib.format.open_memmap(staging / 'posts.npy', mode='w+', dtype=np.uint32, shape=(edges,))
        weights = np.lib.format.open_memmap(staging / 'contacts.npy', mode='w+', dtype=np.uint32, shape=(edges,))
        cursor = offsets[:-1].copy()
        for batch_index, (_, (pre, post, contacts)) in enumerate(edge_batches(raw / FILES['connectivity'], ids)):
            write_csr_batch(cursor, posts, weights, pre, post, contacts)
            if batch_index % 400 == 0:
                print(f'Pack batch {batch_index}', flush=True)
        if not np.array_equal(cursor, offsets[1:]):
            raise ValueError('CSR row counts do not match the audit')
        posts.flush()
        weights.flush()
        # Validate the written graph against independently accumulated totals.
        if int(weights.sum(dtype=np.uint64)) != contacts_total or (len(posts) and posts.max() >= len(ids)):
            raise ValueError('Written structural pack failed validation')
        del posts, weights
        files = {name: {'sha256': digest(staging / name), 'bytes': (staging / name).stat().st_size}
                 for name in ('ids.npy', 'offsets.npy', 'posts.npy', 'contacts.npy')}
        manifest = {
            'format': 'nexus-structural-connectome-1', 'dataset': DATASET,
            'runtime_compatible': False, 'neurons': len(ids), 'edges': edges,
            'weights': 'unsigned synaptic contact counts; no voltage or receptor model',
            'ordering': 'bodyId ascending; edges grouped by pre, stable source order within rows',
            'selection': 'superclass is not null; both edge endpoints selected; no weight threshold',
            'files': files, 'sources': source_manifest,
        }
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    fields = ['bodyId', 'type', 'flywireType', 'superclass', 'mancBodyid', 'mancType', 'somaSide']
    lookup = json.loads(neurons[fields].to_json(orient='records'))
    candidates = []
    for pre, post, contacts in sorted(candidate_edges, key=lambda x: (-x[2], int(ids[x[0]]), int(ids[x[1]]))):
        candidates.append({'pre': lookup[pre], 'post_bodyId': int(ids[post]), 'contacts': contacts})
    label_columns = ['type', 'flywireType', 'mancType', 'class', 'subclass', 'supertype', 'synonyms']
    driver_matches = {}
    for label in ('SS01159', 'SS51024', 'IS35283'):
        found = np.zeros(len(neurons), dtype=bool)
        for column in label_columns:
            found |= neurons[column].astype('string').str.contains(label, regex=False, na=False).to_numpy(dtype=bool)
        driver_matches[label] = neurons.loc[found, 'bodyId'].astype(int).tolist()
    return {
        'dataset': DATASET, 'active_application_dataset': 'FlyWire v630 (unchanged)',
        'stage': 'structural pack and annotation crosswalk prepared; runtime migration pending',
        'source_manifest': source_manifest,
        'annotation_rows': len(annotations), 'classified_neurons': len(ids),
        'classification_rule': 'superclass is not null (not status == Traced)',
        'status_counts': {str(k): int(v) for k, v in neurons.status.fillna('missing').value_counts().items()},
        'superclass_counts': {str(k): int(v) for k, v in neurons.superclass.value_counts().items()},
        'raw_connection_rows': raw_edges, 'retained_connection_rows': edges,
        'excluded_connection_rows': raw_edges-edges, 'retained_contact_count': contacts_total,
        'self_connection_rows': self_edges,
        'zero_outdegree': int((degree == 0).sum()), 'zero_indegree': int((incoming == 0).sum()),
        'isolated_classified_neurons': int(((degree == 0) & (incoming == 0)).sum()),
        'anatomy': {'soma_location_present': int(neurons.somaLocation.notna().sum()),
                    'soma_or_tosoma_present': int((neurons.somaLocation.notna() | neurons.tosomaLocation.notna()).sum()),
                    'status': 'presence audit only; coordinate conversion and missing-location fallback pending'},
        'neurotransmitters': nt, 'type_crosswalk': crosswalk,
        'nociception_candidates': {
            'paper': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC12636578/',
            'interpretation': 'Ascending inputs to candidate central targets; structural adjacency does not identify nociceptors or establish pain.',
            'driver_label_search_columns': label_columns, 'driver_label_matches': driver_matches,
            'ascending_to_target_edges': candidates,
            'unresolved': ['Published ascending driver-to-body-ID mapping in this specimen',
                           'Peripheral nociceptor identities and sensory-to-ascending connections',
                           'Functional response and intervention-ablation validation'],
        },
        'pack_manifest': manifest,
        'migration_gates': ['Explicit transmitter/receptor assumptions and numerical validation',
                            'Dataset-specific inputs and motor readouts (including sugar and bilateral MN9)',
                            'MaleCNS anatomy and live-view neuron ordering',
                            'CPU timing, neural regression assays, and coupled body acceptance'],
    }
