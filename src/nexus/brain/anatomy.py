"""Coordinate provenance and spike-to-position mapping, independent of Qt."""
import hashlib
import json
from pathlib import Path

import numpy as np

FADE_TICKS = 1500  # 150 ms of simulated time, independent of display frame rate.


class Anatomy:
    def __init__(self, directory):
        directory = Path(directory)
        self.manifest = json.loads((directory / 'anatomy-manifest.json').read_text())
        path = directory / 'anatomy.npz'
        if self.manifest['snapshot'] != '630' or hashlib.sha256(path.read_bytes()).hexdigest() != self.manifest['sha256']:
            raise ValueError('Anatomy snapshot or checksum mismatch')
        with np.load(path, allow_pickle=False) as data:
            self.ids = data['ids']
            raw = data['positions_um']
            self.valid = data['valid']
        if not np.array_equal(self.ids, np.load(directory / 'ids.npy', allow_pickle=False)):
            raise ValueError('Anatomy does not match simulation neuron order')
        self.neuron_order_sha256 = hashlib.sha256(self.ids.astype('<i8').tobytes()).hexdigest()
        if raw.shape != (len(self.ids), 3) or self.valid.shape != self.ids.shape or self.valid.dtype != bool:
            raise ValueError('Invalid anatomy shape')
        if not np.isfinite(raw[self.valid]).all():
            raise ValueError('Invalid mapped coordinates')
        self.lookup = {int(root): i for i, root in enumerate(self.ids)}
        # Rigid display transform: (x, z, -y), then centre. Units remain um.
        transformed = raw[:, [0, 2, 1]].copy()
        transformed[:, 2] *= -1
        low, high = np.percentile(transformed[self.valid], [.1, 99.9], axis=0)
        self.centre = (low + high) / 2
        self.positions = transformed - self.centre
        self.span = float(np.max(high - low))

    def activity(self, tick, indices, steps):
        indices = np.asarray(indices, dtype=np.int64)
        steps = np.asarray(steps, dtype=np.int64)
        if indices.shape != steps.shape or indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(self.ids)):
            raise ValueError('Invalid activity indices')
        age = tick - steps
        recent = (age >= 0) & (age <= FADE_TICKS)
        mapped = recent & self.valid[indices]
        strength = 1 - age[mapped] / FADE_TICKS
        color = np.tile([.72, .025, .04, 1.], (int(mapped.sum()), 1)).astype(np.float32)
        color[:, 3] = .25 + .75 * strength
        size = (3 + 3 * strength).astype(np.float32)
        return self.positions[indices[mapped]], color, size, int((recent & ~self.valid[indices]).sum())
