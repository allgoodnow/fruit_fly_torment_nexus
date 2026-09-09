"""Add a verified sensory candidate registry to an existing MaleCNS runtime pack."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from nexus.datasets.nociception import attach_cohort
from nexus.datasets.male_cns import FILES, digest
from nexus.brain.runtime import Connectome


def main():
    pack = ROOT/'data/brain-male-cns-v1.0-lif'
    graph = Connectome.load(pack, allow_experimental=True)
    manifest = json.loads((pack/'manifest.json').read_text())
    annotations = next(r for r in manifest['source_structural_manifest']['sources']['files'] if r['file'] == FILES['annotations'])
    cohort = json.loads((ROOT/'experiments/nociception-cohort-v1.json').read_text())
    registry = attach_cohort(graph.circuits, cohort, graph.ids, annotations['sha256'])
    # An immutable, content-addressed registry lets the manifest switch atomically.
    contents = json.dumps(registry,indent=2)+'\n'
    import hashlib
    checksum = hashlib.sha256(contents.encode()).hexdigest()
    name = 'circuits-'+checksum[:16]+'.json'
    (pack/name).write_text(contents)
    manifest['circuit_registry'] = name
    manifest['files'][name] = checksum
    temporary = pack/'manifest.json.part'
    temporary.write_text(json.dumps(manifest,indent=2)+'\n')
    temporary.replace(pack/'manifest.json')
    Connectome.load(pack, allow_experimental=True)
    (ROOT/'data/manifests/male-cns-runtime-v2.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Installed {len(cohort["ids"])} sensory candidates in {name}')


if __name__ == '__main__':
    main()
