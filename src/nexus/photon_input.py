"""Explicit, uncalibrated video exposure for the offline molecular eye model.

RGB provides relative intensity; the caller must supply absorbed-photon rates.
No opsin spectrum, optical acceptance field, neurotransmitter or spikes are
inferred here. See docs/video-phototransduction.md for the assumptions.
"""
import copy

import numpy as np

from .retina import VisualColumns, sample_plane
from .video import VIDEO_FPS, VideoSource

FRAME_MS = 1000 // VIDEO_FPS


def relative_intensity(frame, *, transfer):
    """Equal mean RGB, optionally decoded from the specified sRGB transfer."""
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or min(frame.shape[:2]) == 0:
        raise ValueError('Expected an RGB8 video frame')
    values = frame.astype(np.float64)/255.
    if transfer == 'srgb':
        values = np.where(values <= .04045, values/12.92, ((values+.055)/1.055)**2.4)
    elif transfer != 'linear':
        raise ValueError('Choose linear or srgb explicitly')
    return values.mean(axis=2)


class PhotonExposure:
    """Independent Poisson absorbed-photon counts in 1 ms bins.

    Rates are photons/s at normalized black and white, after absorption. This
    is a controlled input assumption, not a calibration recovered from video.
    A seed per receptor keeps input randomness independent of call chunking.
    """
    def __init__(self, seeds, *, white_rate_hz, black_rate_hz=0.):
        self.seeds = tuple(seeds)
        if not self.seeds or any(isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer))
                                 or not 0 <= s < 2**64 for s in self.seeds):
            raise ValueError('Supply an unsigned 64-bit seed per receptor')
        self.white_rate_hz, self.black_rate_hz = float(white_rate_hz), float(black_rate_hz)
        if (not np.isfinite([self.white_rate_hz, self.black_rate_hz]).all()
                or not 0 <= self.black_rate_hz <= self.white_rate_hz):
            raise ValueError('Absorbed-photon rates must be finite with 0 <= black <= white')
        self.reset()

    def reset(self):
        self.rngs = [np.random.default_rng(seed) for seed in self.seeds]
        self.time_ms = 0

    def advance(self, intensity):
        values = np.asarray(intensity, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != len(self.seeds) or len(values) == 0
                or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any()):
            raise ValueError('Expected normalized intensity with shape (ms, receptors)')
        means = (self.black_rate_hz + (self.white_rate_hz-self.black_rate_hz)*values)/1000.
        rngs = copy.deepcopy(self.rngs)
        counts = np.column_stack([rng.poisson(means[:, i]) for i, rng in enumerate(rngs)])
        self.rngs = rngs
        self.time_ms += len(values)
        return counts


class VideoPhotonInput:
    """Local video sampled at selected mapped receptor locations on its clock."""
    def __init__(self, path, registry, ids, seeds, *, white_rate_hz, transfer,
                 black_rate_hz=0.):
        mapping = VisualColumns(registry)
        lookup = {root: (side, uv) for side in ('L', 'R')
                  for root, uv in zip(mapping.ids[side], mapping.uv[side])}
        self.ids = tuple(str(root) for root in ids)
        if (not self.ids or len(set(self.ids)) != len(self.ids)
                or any(root not in lookup for root in self.ids)):
            raise ValueError('Choose unique mapped visual receptor IDs')
        self.sides = tuple(lookup[root][0] for root in self.ids)
        self.uv = np.asarray([lookup[root][1] for root in self.ids])
        self.exposure = PhotonExposure(seeds, white_rate_hz=white_rate_hz,
                                      black_rate_hz=black_rate_hz)
        if len(self.exposure.seeds) != len(self.ids):
            raise ValueError('Supply one exposure seed per receptor')
        if transfer not in ('linear', 'srgb'):
            raise ValueError('Choose linear or srgb explicitly')
        self.transfer = transfer
        self.video = VideoSource(path)
        self.time_ms = 0
        self.ended = False

    def read(self, milliseconds):
        if (isinstance(milliseconds, bool) or not isinstance(milliseconds, (int, np.integer))
                or milliseconds < 1):
            raise ValueError('Read a positive integer number of milliseconds')
        inputs, counts = [], []
        remaining = int(milliseconds)
        while remaining and not self.ended:
            frame = self.video.frame_at(self.time_ms//FRAME_MS)
            if frame is None:
                self.ended = True
                break
            sampled = sample_plane(relative_intensity(frame, transfer=self.transfer), self.uv)
            take = min(remaining, FRAME_MS-self.time_ms % FRAME_MS)
            held = np.broadcast_to(sampled, (take, len(self.ids))).copy()
            counts.append(self.exposure.advance(held))
            inputs.append(held)
            self.time_ms += take
            remaining -= take
        return {'intensity': np.concatenate(inputs) if inputs else np.empty((0, len(self.ids))),
                'absorbed_photons': np.concatenate(counts) if counts else np.empty((0, len(self.ids)), dtype=np.int64)}

    def close(self):
        self.video.close()
