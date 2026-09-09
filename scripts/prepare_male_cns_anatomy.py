"""Preserve official MaleCNS cell anchors in the runtime neuron order."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import pandas as pd
from nexus.datasets.male_cns import FILES, digest, classified_neurons


def anchor_arrays(neurons, ids):
    if not np.array_equal(neurons.bodyId.to_numpy(dtype=np.int64), ids):
        raise ValueError('Anatomy and runtime neuron order differ')
    positions = np.full((len(ids), 3), np.nan, dtype=np.float32)
    kinds = np.zeros(len(ids), dtype=np.uint8)
    for i, row in enumerate(neurons[['somaLocation', 'tosomaLocation']].itertuples(index=False)):
        for kind, value in enumerate(row, start=1):
            if value is None or (np.isscalar(value) and pd.isna(value)):
                continue
            coordinate = np.asarray(value, dtype=float)
            if coordinate.shape != (3,) or not np.isfinite(coordinate).all():
                raise ValueError(f'Invalid coordinate for body {ids[i]}')
            positions[i] = coordinate*.008  # 8 nm EM voxel coordinates -> micrometres.
            kinds[i] = kind
            break
    return positions, kinds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, default=ROOT/'data/brain-male-cns-v1.0-lif')
    parser.add_argument('--raw', type=Path, default=ROOT/'data/raw/male-cns-v1.0')
    args = parser.parse_args()
    manifest = json.loads((args.pack/'manifest.json').read_text())
    if manifest['snapshot'] != 'male-cns:v1.0':
        raise ValueError('Expected a MaleCNS runtime pack')
    record = next(r for r in manifest['source_structural_manifest']['sources']['files'] if r['file'] == FILES['annotations'])
    source = args.raw/FILES['annotations']
    if digest(source) != record['sha256'] or digest(args.pack/'ids.npy') != manifest['files']['ids.npy']:
        raise ValueError('Anatomy source or neuron IDs failed checksum verification')
    ids = np.load(args.pack/'ids.npy', allow_pickle=False)
    neurons = classified_neurons(pd.read_feather(source))
    positions, kinds = anchor_arrays(neurons, ids)
    path = args.pack/'anatomy.npz'
    np.savez_compressed(path, ids=ids, positions_um=positions, valid=kinds > 0, anchor_kind=kinds)
    report = {'format': 'nexus-anatomy-1', 'snapshot': manifest['snapshot'],
              'sha256': digest(path), 'neuron_order_sha256': hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),
              'source': record, 'coordinate_source': 'https://male-cns.janelia.org/download/',
              'coordinate_space': 'Male CNS EM; x,y,z in 8 nm voxel units', 'scale_to_um': [.008]*3,
              'representation': 'MaleCNS cell anchors (somaLocation, then tosomaLocation); not neuron morphology',
              'mapped': int((kinds > 0).sum()), 'unlocated': int((kinds == 0).sum()),
              'soma_anchors': int((kinds == 1).sum()), 'tosoma_anchors': int((kinds == 2).sum()),
              'missing_policy': 'No invented positions; unlocated activity is counted separately',
              'license': 'CC-BY-4.0'}
    (args.pack/'anatomy-manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
