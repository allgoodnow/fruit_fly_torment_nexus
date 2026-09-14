"""Audit LPLC2/LC4 -> GF edges and atomically attach the LC4 input registry."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
from nexus.brain.runtime import Connectome
from nexus.datasets.looming import attach_velocity_circuit
from nexus.datasets.male_cns import FILES, digest
from nexus.sequence_run import write_json


def main():
    import hashlib
    pack = ROOT / 'data/brain-male-cns-v1.0-lif'
    annotation = ROOT / 'data/raw/male-cns-v1.0' / FILES['annotations']
    manifest = json.loads((pack / 'manifest.json').read_text())
    source = next(r for r in manifest['source_structural_manifest']['sources']['files']
                  if r['file'] == annotation.name)
    if digest(annotation) != source['sha256']:
        raise ValueError('Annotation checksum differs from the prepared pack')
    graph = Connectome.load(pack, allow_experimental=True)
    table = pd.read_feather(annotation)
    registry = attach_velocity_circuit(graph.circuits, table, graph.ids)
    lookup = {int(root): i for i, root in enumerate(graph.ids)}
    gf = list(map(int, registry['circuits']['giant_fiber']['ids']))
    expected = table.loc[table.type.eq('DNp01') & table.bodyId.isin(graph.ids)]
    if set(gf) != set(map(int, expected.bodyId)) or len(gf) != 2:
        raise ValueError('Giant-fiber readout does not match the annotations')
    edges, cohorts = [], {}
    for name in ['looming', 'looming_velocity']:
        roots = list(map(int, registry['circuits'][name]['ids']))
        rows = table.loc[table.bodyId.isin(roots)]
        cohorts[name] = {'count': len(roots), 'ids': roots,
                         'sides': {str(k): int(v) for k, v in rows.somaSide.value_counts().items()}}
        for root in roots:
            i = lookup[root]
            start, end = graph.offsets[i:i+2]
            for target in gf:
                for offset in np.flatnonzero(graph.posts[start:end] == lookup[target]):
                    weight = float(graph.weights[start + offset])
                    if weight <= 0:
                        raise ValueError('Expected excitatory visual-to-GF model edges')
                    edges.append({'circuit': name, 'pre': root, 'post': target,
                                  'contacts': round(weight / .275), 'model_weight_mv': weight})
        if {e['post'] for e in edges if e['circuit'] == name} != set(gf):
            raise ValueError(f'{name} does not reach both giant fibers')
    contents = json.dumps(registry, indent=2) + '\n'
    checksum = hashlib.sha256(contents.encode()).hexdigest()
    name = 'circuits-' + checksum[:16] + '.json'
    path = pack / name
    if path.exists() and path.read_text() != contents:
        raise ValueError('Content-addressed registry collision')
    path.write_text(contents)
    manifest['circuit_registry'] = name
    manifest['files'][name] = checksum
    write_json(pack / 'manifest.json', manifest)
    Connectome.load(pack, allow_experimental=True)
    write_json(ROOT / 'data/manifests/male-cns-runtime-v3.json', manifest)
    report = {'format': 'nexus-looming-evidence-1', 'dataset': graph.snapshot,
              'annotation_source': source, 'manifest_sha256': digest(pack / 'manifest.json'),
              'cohorts': cohorts, 'giant_fiber_ids': gf, 'visual_to_gf_edges': edges,
              'source': 'https://doi.org/10.1016/j.cub.2019.01.079',
              'limits': ['Typed connections do not validate firing-rate dynamics.',
                         'Virtual approach envelopes are unfitted engineering choices.',
                         'Grounded motor response is authored; no retinal rendering or flight.']}
    write_json(ROOT / 'experiments/looming-evidence-v1.json', report)
    print(json.dumps({'cohorts': {k: v['count'] for k, v in cohorts.items()},
                      'verified_edges': len(edges),
                      'contacts': sum(e['contacts'] for e in edges)}))


if __name__ == '__main__':
    main()
