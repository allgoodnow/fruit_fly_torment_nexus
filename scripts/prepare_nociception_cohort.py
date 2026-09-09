"""Resolve published abdominal md candidates using curated cross-dataset matches."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import pandas as pd
from nexus.datasets.nociception import literal_assignment, map_cohort, trace_ascending_paths
from nexus.datasets.male_cns import FILES, digest, type_crosswalk, classified_neurons


def main():
    ref = json.loads((ROOT/'data/manifests/nociception-reference.json').read_text())
    cohorts = []
    for item in ref['notebooks']:
        path = ROOT/'vendor/nociceptive-reference'/item['file']
        if digest(path) != item['sha256']:
            raise ValueError('Published notebook checksum mismatch')
        nb = json.loads(path.read_text())
        if "dataset='manc:v1.2.1'" not in ''.join(nb['cells'][0]['source']):
            raise ValueError('Published neuron list has an unexpected source dataset')
        cohorts.append(literal_assignment(nb, item['cell_index'], item['variable']))
    if any(set(ids) != set(cohorts[0]) for ids in cohorts):
        raise ValueError('Published md cohorts disagree across figures')
    pack = ROOT/'data/brain-male-cns-v1.0-structural'
    manifest = json.loads((pack/'manifest.json').read_text())
    for name, record in manifest['files'].items():
        if digest(pack/name) != record['sha256']:
            raise ValueError('Structural graph checksum mismatch')
    record = next(r for r in manifest['sources']['files'] if r['file'] == FILES['annotations'])
    path = ROOT/'data/raw/male-cns-v1.0'/FILES['annotations']
    if digest(path) != record['sha256']:
        raise ValueError('Annotation checksum mismatch')
    annotations = pd.read_feather(path)
    ids, offsets, posts, contacts = [np.load(pack/f'{name}.npy', mmap_mode='r', allow_pickle=False)
                                   for name in ['ids', 'offsets', 'posts', 'contacts']]
    selected, mapping = map_cohort(cohorts[0], annotations, ids)
    classified = classified_neurons(annotations)
    if not np.array_equal(classified.bodyId.to_numpy(), ids):
        raise ValueError('Annotation and graph ordering differ')
    crosswalk = type_crosswalk(classified)
    targets = {name: crosswalk[key]['ids'] for name, key in
               [('aversion_candidate', 'aversion_candidate'), ('escape', 'giant_fiber_type'), ('walking', 'walking_type')]}
    ascending = classified.loc[classified.superclass.isin(['ascending_neuron','sensory_ascending']), 'bodyId'].astype(int).tolist()
    sensory_edges, paths = trace_ascending_paths(ids, offsets, posts, contacts, selected, ascending, targets)
    relays = sorted({e['post'] for e in sensory_edges})
    status = {key: sum(r['status'] == key for r in mapping) for key in sorted({r['status'] for r in mapping})}
    report = {'format': 'nexus-nociception-cohort-1', 'dataset': 'male-cns:v1.0',
              'source_reference': ref, 'annotation_source': record,
              'structural_manifest_sha256': digest(pack/'manifest.json'),
              'selection': 'Exact mancBodyid matches; require a single match, runtime membership, vnc_sensory superclass and abdomen subclass. No type-name expansion.',
              'status_counts': status, 'source_count': len(cohorts[0]), 'selected_count': len(selected),
              'ids': [str(i) for i in selected], 'mapping': mapping,
              'target_ids': {name: [str(i) for i in roots] for name, roots in targets.items()},
              'path_contact_threshold': 4, 'ascending_relay_ids': [str(i) for i in relays],
              'sensory_to_ascending_edges': sensory_edges, 'two_edge_paths': paths,
              'path_counts': {name: sum(p['target_group'] == name for p in paths) for name in targets},
              'interpretation': 'Published md homolog candidates; curated structural correspondence supports investigation, not proof of nociceptive function in this specimen or subjective pain.',
              'limitations': ['MANC v1.2.1 and MaleCNS v1.0 are different specimens; crosswalk errors remain possible',
                              'Current broad class labels include chemosensory and unknown_sensory; these labels alone do not establish nociception',
                              'Four excluded source cells match unclassified segments and are not added to the 166700-neuron runtime',
                              'SS01159, SS51024 and IS35283 ascending driver assignments remain unresolved',
                              'Optogenetic-like input rates are authored; no calibrated thermal or mechanical transduction']}
    output = ROOT/'experiments/nociception-cohort-v1.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    protocol = {'format': 'nexus-protocol-1', 'dataset': 'male-cns:v1.0',
                'name': 'Abdominal md homolog candidates / experimental nociception input',
                'duration_ms': 700, 'events': [{'at_ms': 0, 'action': 'release'},
                                             {'at_ms': 100, 'action': 'stimulate', 'ids': report['ids'], 'rate_hz': 100},
                                             {'at_ms': 400, 'action': 'release'}]}
    (ROOT/'experiments/nociception-input-v1.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print(json.dumps({'status': status, 'selected': len(selected), 'ascending_relays': len(relays),
                      'sensory_edges': len(sensory_edges), 'path_counts': report['path_counts']}, indent=2))


if __name__ == '__main__':
    main()
