"""Add provenance-bound, conservative R1–R6 visual-column assignments."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
from nexus.brain.runtime import Connectome, weights_filename
from nexus.datasets.male_cns import FILES, digest
from nexus.datasets.visual_columns import attach_visual_columns
from nexus.retina import VisualColumns


def main():
    pack = ROOT / 'data/brain-male-cns-v1.0-lif'
    structural = ROOT / 'data/brain-male-cns-v1.0-structural'
    raw = ROOT / 'data/raw/male-cns-v1.0' / FILES['annotations']
    original = (pack / 'manifest.json').read_bytes()
    manifest = json.loads(original)
    source = manifest['source_structural_manifest']
    annotation_record = next(r for r in source['sources']['files'] if r['file'] == FILES['annotations'])
    if digest(raw) != annotation_record['sha256']:
        raise ValueError('Annotations differ from this pack')
    for name in ['ids.npy', 'offsets.npy', 'posts.npy', 'contacts.npy']:
        if digest(structural / name) != source['files'][name]['sha256']:
            raise ValueError('Structural graph checksum mismatch')
    graph = Connectome.load(pack, allow_experimental=True)
    for name, current in [('ids.npy', graph.ids), ('offsets.npy', graph.offsets), ('posts.npy', graph.posts)]:
        if not np.array_equal(current, np.load(structural / name, mmap_mode='r')):
            raise ValueError('Structural and runtime graph ordering differ')
    registry = attach_visual_columns(graph.circuits, pd.read_feather(raw), graph.ids,
                                     graph.offsets, graph.posts, np.load(structural / 'contacts.npy', mmap_mode='r'))
    mapping = VisualColumns(registry)
    content = (json.dumps(registry, indent=2) + '\n').encode()
    checksum = hashlib.sha256(content).hexdigest()
    name = 'circuits-' + checksum[:16] + '.json'
    manifest['circuit_registry'] = name
    manifest['files'][name] = checksum
    with tempfile.TemporaryDirectory(prefix='.columns-', dir=pack) as directory:
        staging = Path(directory)
        for array in ['ids.npy', 'offsets.npy', 'posts.npy', weights_filename(manifest)]:
            (staging / array).hardlink_to(pack / array)
        (staging / name).write_bytes(content)
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        VisualColumns(Connectome.load(staging, allow_experimental=True).circuits)
        if (pack / 'manifest.json').read_bytes() != original:
            raise ValueError('Pack changed during preparation')
        if (pack / name).exists() and (pack / name).read_bytes() != content:
            raise ValueError('Registry checksum collision')
        (staging / name).replace(pack / name)
        (staging / 'manifest.json').replace(pack / 'manifest.json')
    report = {'format': 'nexus-visual-column-evidence-1', 'manifest_sha256': digest(pack / 'manifest.json'),
              'annotations': annotation_record, 'mapping': registry['visual_columns'],
              'structural_files': {name: source['files'][name] for name in ['ids.npy', 'offsets.npy', 'posts.npy', 'contacts.npy']}}
    (ROOT / 'experiments/visual-column-evidence-v1.json').write_text(json.dumps(report, indent=2) + '\n')
    (ROOT / 'data/manifests/male-cns-runtime-v8.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({side: {'cells': len(mapping.ids[side]), **{k:v for k,v in registry['visual_columns']['eyes'][side].items() if k != 'cells'}} for side in ['L','R']}))


if __name__ == '__main__':
    main()
