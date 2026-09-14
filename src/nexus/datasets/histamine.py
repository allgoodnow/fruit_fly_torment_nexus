"""A narrow receptor-informed sign override; no new anatomical edges.

The evidence supports the sign at R1-R6 -> L1/L2 synapses. The inherited
contact scaling and spike-based dynamics remain unfitted approximations.
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

POLICY_ID = 'male-cns-lamina-histamine-lif-v2'
RULE_ID = 'histamine-R1-R6-to-L1-L2-ort-v1'
SOURCE = 'https://doi.org/10.1074/jbc.M207133200'


def select_edges(ids, offsets, posts, neurons, labels, *,
                 presynaptic_types=('R1-R6',), postsynaptic_types=('L1', 'L2')):
    """Align annotations by ID, then select only scanned edges in the rule."""
    matched = neurons.loc[neurons.bodyId.isin(ids)]
    if matched.bodyId.duplicated().any():
        raise ValueError('Duplicate annotation body IDs')
    if len(matched) != len(ids):
        raise ValueError('Missing annotation rows')
    aligned = matched.set_index('bodyId').reindex(ids)
    if not np.array_equal(labels.index.to_numpy(), ids):
        raise ValueError('Transmitter labels must be aligned to graph IDs')
    # Classified neurons can lack a specific type. They never match this rule.
    types = aligned.type.fillna('')
    eligible = types.isin(presynaptic_types).to_numpy() & labels.eq('histamine').to_numpy()
    targets = types.isin(postsynaptic_types).to_numpy()
    parts = []
    for pre in np.flatnonzero(eligible):
        start, end = int(offsets[pre]), int(offsets[pre + 1])
        parts.append(start + np.flatnonzero(targets[posts[start:end]]))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def apply_override(weights, contacts, edges, model):
    """Only the first omission-only policy may be upgraded, exactly once."""
    if model.get('id') != 'male-cns-fast-transmitter-lif-v1' or model.get('edge_sign_overrides'):
        raise ValueError('Expected the original fast-transmitter policy')
    if model.get('nt_signs', {}).get('histamine') != 0:
        raise ValueError('Expected omitted histamine transmission')
    if np.any(weights[edges] != 0) or np.any(contacts[edges] <= 0):
        raise ValueError('Override requires omitted edges with positive scanned contacts')
    result = copy.deepcopy(model)
    if not len(edges):
        return result
    weights[edges] = -contacts[edges].astype(np.float64) * model['weight_per_contact_mv']
    result['id'] = POLICY_ID
    result['sign_assumptions'] = (
        'Presynaptic fast-transmitter signs with a receptor-informed exception for '
        'histaminergic R1-R6 -> L1/L2. Other histamine and monoamine effects remain omitted. '
        'Glutamate is assumed inhibitory; other receptor-specific effects remain unresolved.')
    result['edge_sign_overrides'] = [{
        'id': RULE_ID, 'presynaptic_type': 'R1-R6', 'presynaptic_consensus_nt': 'histamine',
        'postsynaptic_types': ['L1', 'L2'], 'sign': -1, 'receptor': 'Ort / HCLA',
        'source': SOURCE, 'edges': len(edges), 'contacts': int(contacts[edges].sum()),
        'edge_indices_sha256': hashlib.sha256(
            np.asarray(edges, dtype='<i8').tobytes()).hexdigest(),
        'limitation': 'Cell-type receptor evidence, not measured receptors in this specimen. '
                      'Inherited LIF/contact scaling does not model graded photoreceptor release, '
                      'chloride reversal potential, fitted kinetics, or working vision.'}]
    # The original coverage remains a statement of the default presynaptic sign.
    coverage = result.get('transmitter_coverage', {}).get('histamine')
    if coverage is not None:
        coverage['overridden_inhibitory_edges'] = len(edges)
        coverage['omitted_edges'] = coverage['outgoing_edges'] - len(edges)
    return result


def upgrade_pack(pack, structural, raw, *, extended=False):
    """Validate a complete candidate before atomically activating its manifest.

    Existing arrays are never rewritten. A saved prior manifest and its weights
    allow exact rollback. An already loaded Brain keeps its original mmap.
    """
    import pandas as pd
    from nexus.brain.runtime import Connectome, weights_filename
    from .male_cns import DATASET, FILES, digest
    from .male_cns_runtime import transmitter_signs

    pack, structural, raw = map(Path, (pack, structural, raw))
    previous = (pack / 'manifest.json').read_bytes()
    manifest = json.loads(previous)
    graph = Connectome.load(pack, allow_experimental=True)
    accepted = ['male-cns-fast-transmitter-lif-v1'] + ([POLICY_ID] if extended else [])
    if graph.snapshot != DATASET or graph.model.get('id') not in accepted:
        raise ValueError('Upgrade requires the original MaleCNS model; it may already be installed')
    source = manifest['source_structural_manifest']
    if json.loads((structural / 'manifest.json').read_text()) != source:
        raise ValueError('Structural provenance mismatch')
    for name in ['ids.npy', 'offsets.npy', 'posts.npy', 'contacts.npy']:
        if digest(structural / name) != source['files'][name]['sha256']:
            raise ValueError(f'Structural checksum mismatch: {name}')
    records = {r['file']: r for r in source['sources']['files']}
    for name in [FILES['annotations'], FILES['neurotransmitters']]:
        if digest(raw / name) != records[name]['sha256']:
            raise ValueError(f'Raw checksum mismatch: {name}')
    for name in ['ids', 'offsets', 'posts']:
        if not np.array_equal(np.load(structural / f'{name}.npy', mmap_mode='r'), getattr(graph, name)):
            raise ValueError('Structural/runtime graph ordering mismatch')
    contacts = np.load(structural / 'contacts.npy', mmap_mode='r', allow_pickle=False)
    neurons = pd.read_feather(raw / FILES['annotations'])
    signs, labels = transmitter_signs(graph.ids, pd.read_feather(raw / FILES['neurotransmitters']))
    edges = select_edges(graph.ids, graph.offsets, graph.posts, neurons, labels)
    prior_edges = edges if graph.model['id'] == POLICY_ID else np.empty(0, dtype=np.int64)
    groups = None
    if extended:
        from .receptor_signs import compile_rules, validate_prior_policy, apply_extended
        validate_prior_policy(graph.model, contacts, prior_edges)
        groups = compile_rules(graph.ids, graph.offsets, graph.posts, neurons, labels)
        parts = [e for _, e in groups]
        if graph.model['id'] != POLICY_ID:
            parts.append(edges)
        edges = np.sort(np.concatenate(parts))
    if not len(edges):
        raise ValueError('No scanned edges match the receptor rule')
    for first in range(0, len(graph.ids), 4096):
        last = min(first + 4096, len(graph.ids))
        start, end = graph.offsets[[first, last]]
        expected = (contacts[start:end] * np.repeat(signs[first:last], np.diff(graph.offsets[first:last+1]))
                    .astype(np.float64) * graph.model['weight_per_contact_mv'])
        restored = prior_edges[(prior_edges >= start) & (prior_edges < end)]
        expected[restored - start] = -contacts[restored].astype(np.float64) * graph.model['weight_per_contact_mv']
        if not np.array_equal(graph.weights[start:end], expected):
            raise ValueError('Runtime weights differ from the original sign policy')
    with tempfile.TemporaryDirectory(prefix='.histamine-', dir=pack) as staging_name:
        staging = Path(staging_name)
        # Validate the prospective pack without switching the running installation.
        for name in ['ids.npy', 'offsets.npy', 'posts.npy', manifest['circuit_registry']]:
            (staging / name).hardlink_to(pack / name)
        shutil.copyfile(pack / weights_filename(manifest), staging / 'weights.npy')
        weights = np.load(staging / 'weights.npy', mmap_mode='r+')
        if extended:
            model = apply_extended(weights, contacts, graph.ids, graph.offsets, graph.posts,
                                   neurons, labels, graph.model, groups=groups)
        else:
            model = apply_override(weights, contacts, edges, graph.model)
        weights.flush()
        del weights
        checksum = digest(staging / 'weights.npy')
        name = f'weights-{checksum}.npy'
        (staging / 'weights.npy').rename(staging / name)
        candidate = copy.deepcopy(manifest)
        candidate['model'] = model
        candidate['weights_file'] = name
        candidate['files'][name] = checksum
        contents = (json.dumps(candidate, indent=2) + '\n').encode()
        (staging / 'manifest.json').write_bytes(contents)
        Connectome.load(staging, allow_experimental=True)
        if (pack / 'manifest.json').read_bytes() != previous:
            raise ValueError('Pack changed during preparation; refusing to switch it')
        backup_name = f'manifest-before-histamine-{hashlib.sha256(previous).hexdigest()}.json'
        backup = pack / backup_name
        if backup.exists() and backup.read_bytes() != previous:
            raise ValueError('Backup manifest collision')
        backup.write_bytes(previous)
        if (pack / name).exists():
            if digest(pack / name) != checksum:
                raise ValueError('Weights hash collision')
        else:
            (staging / name).rename(pack / name)
        (staging / 'manifest.json').replace(pack / 'manifest.json')
    report = {'format': 'nexus-histamine-evidence-1', 'dataset': DATASET,
                       'original_manifest_sha256': hashlib.sha256(previous).hexdigest(),
                       'manifest_sha256': digest(pack / 'manifest.json'),
                       'rollback_manifest': backup_name,
                       'source_annotations': records[FILES['annotations']],
                       'source_transmitters': records[FILES['neurotransmitters']],
                       'rule': model['edge_sign_overrides'][0],
                       'neurons': len(graph.ids), 'edges': len(graph.posts),
                       'changed_weights': len(edges), 'topology_unchanged': True,
                       'original_weights_sha256': manifest['files'][weights_filename(manifest)],
                       'weights_sha256': checksum}
    if extended:
        report['format'] = 'nexus-receptor-sign-evidence-1'
        report.pop('rule')
        report['rules'] = model['edge_sign_overrides']
        report['previous_model'] = graph.model['id']
        report['model'] = model['id']
        report['changed_edge_indices_sha256'] = hashlib.sha256(
            np.asarray(edges, dtype='<i8').tobytes()).hexdigest()
    return candidate, report
