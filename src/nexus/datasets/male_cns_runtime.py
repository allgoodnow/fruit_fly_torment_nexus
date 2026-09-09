"""Explicit modeling assumptions for the first MaleCNS runtime experiment."""
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .male_cns import DATASET, FILES, classified_neurons, digest, type_crosswalk

MODEL_POLICY = {
    'id': 'male-cns-fast-transmitter-lif-v1',
    'experimental': True,
    'validation': 'Unfitted MaleCNS adaptation; numerical checks are not biological validation.',
    'nt_signs': {'acetylcholine': 1, 'gaba': -1, 'glutamate': -1,
                 'histamine': 0, 'dopamine': 0, 'octopamine': 0, 'serotonin': 0,
                 'unclear': 0, 'missing': 0},
    'sign_assumptions': 'One sign per presynaptic cell. Glutamate assumed inhibitory; receptor-specific effects unresolved. Histamine and monoamine effects omitted, not biologically absent.',
    'weight_per_contact_mv': 0.275,
    'dynamics': 'Shiu-style LIF: dt 0.1 ms, membrane 20 ms, current 5 ms, delay 1.8 ms, refractory 2.2 ms; directly driven cells have zero refractory.',
    'parameter_source': 'https://doi.org/10.1038/s41586-024-07763-9',
    'limitation': 'No calibrated MaleCNS/VNC dynamics, plasticity, receptor kinetics, nociception validation, or subjective-state model.',
}


def transmitter_signs(ids, table):
    matched = table.loc[table.body.isin(ids)]
    if matched.body.duplicated().any():
        raise ValueError('Duplicate neurotransmitter body IDs')
    labels = matched.set_index('body').consensus_nt.reindex(ids).fillna('missing')
    unknown = set(labels.unique()) - MODEL_POLICY['nt_signs'].keys()
    if unknown:
        raise ValueError(f'Unrecognized neurotransmitter labels: {sorted(unknown)}')
    return labels.map(MODEL_POLICY['nt_signs']).to_numpy(dtype=np.int8), labels


def build_registry(neurons):
    crosswalk = type_crosswalk(neurons)
    entries = {}
    mapping = {'looming': 'looming_type', 'warmth': 'warmth_type',
               'aversion_proxy': 'aversion_candidate', 'giant_fiber': 'giant_fiber_type'}
    for key, name in mapping.items():
        item = crosswalk[name]
        entries[key] = {'ids': [str(i) for i in item['ids']],
                        'selection': item['selector'], 'annotation_matches': item['annotation_matches'],
                        'limitation': item['status']}
        if not entries[key]['ids']:
            raise ValueError(f'Missing circuit: {key}')
    entries['aversion_proxy']['limitation'] = ('GNG121 / FlyWire-type CB0059 correspondence only. '
        'Ascending nociceptor mapping and subjective pain remain unvalidated.')
    sides = {}
    for side in ['L', 'R']:
        rows = neurons.loc[neurons.type.eq('DNa02') & neurons.somaSide.eq(side)]
        if len(rows) != 1:
            raise ValueError(f'Expected one DNa02 on side {side}')
        sides[side] = [str(int(rows.bodyId.iloc[0]))]
    mn9 = crosswalk['feeding_motor_type']['ids']
    if len(mn9) != 2:
        raise ValueError('Expected bilateral MN9 readout')
    return {'format': 'nexus-intervention-circuits-1', 'snapshot': DATASET,
            'annotation_source': 'https://male-cns.janelia.org/download/',
            'circuits': entries,
            'readouts': {'steering_left': sides['L'], 'steering_right': sides['R'],
                         'mn9': [str(i) for i in mn9]},
            'unavailable_inputs': {'sugar': 'No validated MaleCNS sugar-cell mapping yet.'}}


def prepare_runtime(structural, raw, output, *, nociception_cohort=None):
    import pandas as pd
    from nexus.brain.runtime import Connectome
    structural, raw, output = Path(structural), Path(raw), Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    source = json.loads((structural/'manifest.json').read_text())
    if source.get('format') != 'nexus-structural-connectome-1' or source.get('dataset') != DATASET:
        raise ValueError('Expected a MaleCNS structural pack')
    for name in ['ids.npy', 'offsets.npy', 'posts.npy', 'contacts.npy']:
        if digest(structural/name) != source['files'][name]['sha256']:
            raise ValueError(f'Structural checksum mismatch: {name}')
    records = {r['file']: r for r in source['sources']['files']}
    for name in [FILES['annotations'], FILES['neurotransmitters']]:
        if digest(raw/name) != records[name]['sha256']:
            raise ValueError(f'Raw checksum mismatch: {name}')
    neurons = classified_neurons(pd.read_feather(raw/FILES['annotations']))
    ids, offsets, posts, contacts = [np.load(structural/f'{name}.npy', mmap_mode='r', allow_pickle=False)
                                   for name in ['ids', 'offsets', 'posts', 'contacts']]
    if not np.array_equal(ids, neurons.bodyId.to_numpy(dtype=np.int64)):
        raise ValueError('Annotation/graph neuron ordering mismatch')
    if (offsets.ndim != 1 or len(offsets) != len(ids)+1 or offsets[0] != 0
            or offsets[-1] != len(posts) or (np.diff(offsets) < 0).any()
            or len(contacts) != len(posts) or (posts.size and posts.max() >= len(ids))):
        raise ValueError('Invalid structural adjacency')
    signs, labels = transmitter_signs(ids, pd.read_feather(raw/FILES['neurotransmitters']))
    registry = build_registry(neurons)
    if nociception_cohort is not None:
        from .nociception import attach_cohort
        registry = attach_cohort(registry, nociception_cohort, ids, records[FILES['annotations']]['sha256'])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name+'.partial-', dir=output.parent))
    try:
        np.save(staging/'ids.npy', ids, allow_pickle=False)
        np.save(staging/'offsets.npy', offsets, allow_pickle=False)
        np.save(staging/'posts.npy', posts.astype(np.int32), allow_pickle=False)
        weights = np.lib.format.open_memmap(staging/'weights.npy', mode='w+', dtype=np.float64, shape=contacts.shape)
        # Row blocks bound temporary sign expansion regardless of total edge count.
        for first in range(0, len(ids), 4096):
            last = min(first+4096, len(ids))
            a, b = int(offsets[first]), int(offsets[last])
            edge_signs = np.repeat(signs[first:last], np.diff(offsets[first:last+1]))
            weights[a:b] = contacts[a:b]*edge_signs.astype(np.float64)*MODEL_POLICY['weight_per_contact_mv']
        weights.flush()
        degree = np.diff(offsets)
        summary = {label: {'cells': int((labels == label).sum()),
                          'outgoing_edges': int(degree[(labels == label).to_numpy()].sum()),
                          'assigned_sign': MODEL_POLICY['nt_signs'][label]}
                   for label in sorted(labels.unique())}
        del weights
        (staging/'circuits.json').write_text(json.dumps(registry, indent=2)+'\n')
        manifest = {'format': 'nexus-connectome-1', 'snapshot': DATASET,
                    'neurons': len(ids), 'edges': len(posts), 'experimental': True,
                    'model': dict(MODEL_POLICY, transmitter_coverage=summary),
                    'source_structural_manifest': source,
                    'circuit_registry': 'circuits.json',
                    'files': {name: digest(staging/name) for name in
                              ['ids.npy', 'offsets.npy', 'posts.npy', 'weights.npy', 'circuits.json']}}
        (staging/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        Connectome.load(staging, allow_experimental=True)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging)
        raise
