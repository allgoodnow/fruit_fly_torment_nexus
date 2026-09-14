"""Audit and attach the official bilateral BDN2 forward-walking readout."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
from nexus.brain.runtime import Connectome
from nexus.datasets.male_cns import FILES, digest
from nexus.datasets.walking import attach_forward_readout
from nexus.sequence_run import write_json


def main():
    pack = ROOT / 'data/brain-male-cns-v1.0-lif'
    annotation = ROOT / 'data/raw/male-cns-v1.0' / FILES['annotations']
    manifest = json.loads((pack / 'manifest.json').read_text())
    source = next(r for r in manifest['source_structural_manifest']['sources']['files']
                  if r['file'] == annotation.name)
    if digest(annotation) != source['sha256']:
        raise ValueError('Annotations differ from the prepared pack')
    graph = Connectome.load(pack, allow_experimental=True)
    table = pd.read_feather(annotation)
    registry = attach_forward_readout(graph.circuits, table, graph.ids)
    selected = table.loc[table.bodyId.isin(map(int, registry['readouts']['forward_walking']))]
    contents = json.dumps(registry, indent=2) + '\n'
    checksum = hashlib.sha256(contents.encode()).hexdigest()
    name = 'circuits-' + checksum[:16] + '.json'
    path = pack / name
    if path.exists() and path.read_text() != contents:
        raise ValueError('Registry hash collision')
    path.write_text(contents)
    manifest['circuit_registry'] = name
    manifest['files'][name] = checksum
    write_json(pack / 'manifest.json', manifest)
    Connectome.load(pack, allow_experimental=True)
    write_json(ROOT / 'data/manifests/male-cns-runtime-v4.json', manifest)
    report = {'format': 'nexus-walking-evidence-1', 'dataset': graph.snapshot,
              'annotation_source': source, 'manifest_sha256': digest(pack / 'manifest.json'),
              'annotations': json.loads(selected[['bodyId', 'type', 'somaSide', 'superclass', 'synonyms']].to_json(orient='records')),
              'functional_source': 'https://doi.org/10.1038/s41586-024-07854-7',
              'readout': registry['readouts']['forward_walking'],
              'limits': ['Forward walking association does not supply a calibrated firing-to-speed function.',
                         'This is a descending readout, not a VNC motor-neuron-to-muscle reconstruction.',
                         'No endogenous excitation or spontaneous exploration is added.']}
    write_json(ROOT / 'experiments/walking-evidence-v1.json', report)
    print(json.dumps(report['annotations']))


if __name__ == '__main__':
    main()
