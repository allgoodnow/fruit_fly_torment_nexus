"""Bind left/right eye input cohorts to official MaleCNS annotations."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
from nexus.brain.runtime import Connectome, weights_filename
from nexus.datasets.male_cns import FILES, digest
from nexus.datasets.eyes import attach_eye_inputs


def main():
    pack = ROOT / 'data/brain-male-cns-v1.0-lif'
    raw = ROOT / 'data/raw/male-cns-v1.0'
    original = (pack / 'manifest.json').read_bytes()
    manifest = json.loads(original)
    graph = Connectome.load(pack, allow_experimental=True)
    records = {r['file']: r for r in manifest['source_structural_manifest']['sources']['files']}
    for name in [FILES['annotations'], FILES['neurotransmitters']]:
        if digest(raw / name) != records[name]['sha256']:
            raise ValueError('Official annotations differ from this pack')
    registry = attach_eye_inputs(graph.circuits, pd.read_feather(raw / FILES['annotations']),
                                 pd.read_feather(raw / FILES['neurotransmitters']), graph.ids)
    contents = (json.dumps(registry, indent=2) + '\n').encode()
    checksum = hashlib.sha256(contents).hexdigest()
    name = 'circuits-' + checksum[:16] + '.json'
    manifest['circuit_registry'] = name
    manifest['files'][name] = checksum
    with tempfile.TemporaryDirectory(prefix='.eyes-', dir=pack) as directory:
        staging = Path(directory)
        for array in ['ids.npy', 'offsets.npy', 'posts.npy', weights_filename(manifest)]:
            (staging / array).hardlink_to(pack / array)
        (staging / name).write_bytes(contents)
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        Connectome.load(staging, allow_experimental=True)
        if (pack / 'manifest.json').read_bytes() != original:
            raise ValueError('Pack changed during preparation')
        if (pack / name).exists() and (pack / name).read_bytes() != contents:
            raise ValueError('Registry hash collision')
        (staging / name).replace(pack / name)
        (staging / 'manifest.json').replace(pack / 'manifest.json')
    report = {'format': 'nexus-eye-input-evidence-1', 'snapshot': graph.snapshot,
              'manifest_sha256': digest(pack / 'manifest.json'),
              'annotation_source': records[FILES['annotations']],
              'transmitter_source': records[FILES['neurotransmitters']],
              'cohorts': {key: registry['circuits'][key] for key in ['eye_left', 'eye_right']},
              'limits': ['Unfitted pooled image brightness; 100 Hz at full white.',
                         'Root-side mapping only; no retinotopy or graded phototransduction.',
                         'Cohort counts reflect the classified reconstruction, not matched biological eye sizes.']}
    for path, value in [(ROOT / 'data/manifests/male-cns-runtime-v7.json', manifest),
                         (ROOT / 'experiments/eye-input-evidence-v1.json', report)]:
        path.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({key: len(value['ids']) for key, value in report['cohorts'].items()}))


if __name__ == '__main__':
    main()
