"""Verify the decoder targets against the pinned v630 annotation table."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nexus.brain.motor import DNA02

source = ROOT / 'vendor/flywire-annotations/supplemental_files/Supplemental_file1_annotations.tsv'
model_ids = set(np.load(ROOT / 'data/brain-v630/ids.npy').tolist())
selected = {}
with source.open() as stream:
    for row in csv.DictReader(stream, delimiter='\t'):
        if row['root_id'] in DNA02:
            if row['root_id'] in selected:
                raise ValueError('Duplicate target annotation')
            selected[row['root_id']] = {k: row[k] for k in ['root_id', 'hemibrain_type', 'cell_type', 'side', 'flow', 'super_class']}
for root, side in zip(DNA02, ['left', 'right']):
    row = selected[root]
    assert int(root) in model_ids and row['side'] == side
    assert row['hemibrain_type'] == 'DNa02' and row['super_class'] == 'descending'
registry = {'snapshot': '630', 'targets': [selected[root] for root in DNA02],
            'annotation_source': 'https://github.com/flyconnectome/flywire_annotations/tree/v1.1.0',
            'annotation_commit': 'df6bb136f5b3d91c3992df4e8de2642329e2a384',
            'annotation_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'functional_evidence': 'https://doi.org/10.1016/j.cell.2024.08.033',
            'decoder': {'name': 'DNa02 ipsilateral attenuation v1', 'filter_seconds': .1,
                        'reference_rate_hz': 200, 'maximum_attenuation': .7,
                        'baseline_drive': 1, 'coupling_ms': 10},
            'status': 'Experimental engineering decoder; gains are authored, VNC not simulated, no sensory feedback'}
(ROOT / 'data/brain-v630/motor-registry.json').write_text(json.dumps(registry, indent=2)+'\n')
print(json.dumps(registry, indent=2))
