"""Align official v630 cell anchors to the exact simulation index order."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'vendor/flywire-annotations/supplemental_files/Supplemental_file1_annotations.tsv'
PACK = ROOT / 'data/brain-v630'


def prepare():
    ids = np.load(PACK / 'ids.npy', allow_pickle=False)
    lookup = {int(root): i for i, root in enumerate(ids)}
    positions = np.full((len(ids), 3), np.nan, dtype=np.float32)
    valid = np.zeros(len(ids), dtype=bool)
    with SOURCE.open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            index = lookup.get(int(row['root_id']))
            if index is None:
                continue
            if valid[index]:
                raise ValueError(f"Duplicate annotation for {row['root_id']}")
            xyz = np.array([float(row[f'pos_{axis}']) for axis in 'xyz']) * [.004, .004, .04]
            if not np.isfinite(xyz).all():
                raise ValueError('Non-finite anchor')
            positions[index] = xyz
            valid[index] = True
    target = PACK / 'anatomy.npz'
    np.savez_compressed(target, ids=ids, positions_um=positions, valid=valid)
    manifest = {
        'format': 'nexus-anatomy-1', 'snapshot': '630',
        'source': 'https://github.com/flyconnectome/flywire_annotations/tree/v1.1.0',
        'source_commit': 'df6bb136f5b3d91c3992df4e8de2642329e2a384',
        'source_file': str(SOURCE.relative_to(ROOT / 'vendor/flywire-annotations')),
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'coordinates': 'Cell anchors (typically backbone), not full morphology or guaranteed somas',
        'conversion': 'pos_x/y/z in 4 x 4 x 40 nm voxels multiplied by [0.004, 0.004, 0.04] to micrometres',
        'mapped': int(valid.sum()), 'unmapped': int((~valid).sum()),
        'outliers': 'Two source anchors exceed 1000 um in at least one axis; retained unchanged. Home camera fits central 99.8%; Fit all includes every mapped anchor.',
    }
    (PACK / 'anatomy-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    prepare()
