"""Explicit independent receptors with memory-bounded CUDA molecular batches.

SPDX-License-Identifier: GPL-3.0-only
Shares the existing parallel cascade and BG1 membrane. Offline research only.
"""
import copy

import numpy as np

from .phototransduction_parallel import CHUNK_MS, ParallelPhototransduction
from .phototransduction_cuda import CudaWork, require_cuda
from .photoreceptor import PhotoreceptorMembrane
from .photoreceptor_cuda import CudaPhotoreceptorMembrane

ARRAY_NAMES = ('states', 'due', 'reactions', 'reversals', 'streams')


class BatchedPhototransduction:
    """One full molecular population per receptor, never multiplied or shared.

    Input has shape (milliseconds, receptors). Explicit seeds retain each
    receptor's identity across shard sizes and separate calls. Molecular state
    lives on the host between public calls; at most batch_cells receptors reside
    on the GPU together. Channel traces then drive the selected batched membrane.
    """
    def __init__(self, seeds, *, microvilli=30000, batch_cells=8, membrane_backend='cpu'):
        seeds = tuple(seeds)
        if not seeds:
            raise ValueError('Supply one seed per receptor')
        if (isinstance(batch_cells, bool) or not isinstance(batch_cells, (int, np.integer))
                or batch_cells < 1):
            raise ValueError('Batch size must be a positive integer')
        if membrane_backend not in ('cpu', 'cuda'):
            raise ValueError('Choose cpu or cuda for the membrane')
        # Reuse the single-receptor parameter validation and stream initialization.
        # Physical CUDA is required explicitly; there is no automatic fallback.
        require_cuda()
        self.seeds, self.microvilli, self.batch_cells = seeds, microvilli, int(batch_cells)
        self.cells = len(seeds)
        self.membrane_backend = membrane_backend
        membrane_type = PhotoreceptorMembrane if membrane_backend == 'cpu' else CudaPhotoreceptorMembrane
        self.membrane = membrane_type(self.cells)
        self.reset()

    def reset(self):
        models = [ParallelPhototransduction(microvilli=self.microvilli, seed=seed)
                  for seed in self.seeds]
        self.microvilli = models[0].microvilli
        for name in ARRAY_NAMES:
            setattr(self, name, np.concatenate([getattr(m, name) for m in models]))
        self.rngs = [m.rng for m in models]
        self.membrane.reset()
        self.time_ms = 0
        self.events = np.zeros(self.cells, dtype=np.int64)
        self.photons = np.zeros(self.cells, dtype=np.int64)
        self.initialized = False

    @property
    def molecular_state_bytes(self):
        return sum(getattr(self, name).nbytes for name in ARRAY_NAMES)

    def advance(self, absorbed_photons):
        raw = np.asarray(absorbed_photons)
        if (raw.ndim != 2 or raw.shape[1] != self.cells or raw.dtype.kind not in 'iuf'
                or len(raw) == 0 or not np.isfinite(raw).all() or (raw < 0).any()
                or (raw > 1000000).any() or (raw != np.floor(raw)).any()):
            raise ValueError('Expected integer absorbed photon counts with shape (ms, receptors)')
        photons = raw.astype(np.int64)
        arrays = [getattr(self, name).copy() for name in ARRAY_NAMES]
        rngs = copy.deepcopy(self.rngs)
        channels = np.empty((len(photons)*10, self.cells), dtype=np.int64)
        events = np.zeros(self.cells, dtype=np.int64)
        for first in range(0, self.cells, self.batch_cells):
            last = min(first+self.batch_cells, self.cells)
            lo, hi = first*self.microvilli, last*self.microvilli
            work = CudaWork([a[lo:hi] for a in arrays], microvilli=self.microvilli)
            for start in range(0, len(photons), CHUNK_MS):
                stop = min(start+CHUNK_MS, len(photons))
                allocated = np.zeros((stop-start, hi-lo), dtype=np.int32)
                for receptor in range(first, last):
                    column = (receptor-first)*self.microvilli
                    for frame, count in enumerate(photons[start:stop, receptor]):
                        if count:
                            chosen = rngs[receptor].integers(0, self.microvilli, size=int(count))
                            allocated[frame, column:column+self.microvilli] = np.bincount(
                                chosen, minlength=self.microvilli)
                result, n = work.advance_receptors(allocated, self.time_ms+start,
                                                  self.initialized or start > 0)
                channels[start*10:stop*10, first:last] = result
                events[first:last] += n
            for candidate, finished in zip(arrays, work.finish()):
                candidate[lo:hi] = finished
            # Release the previous shard before allocating the next.
            del work
        membrane = copy.deepcopy(self.membrane)
        voltage = membrane.advance_channels(channels)
        for name, candidate in zip(ARRAY_NAMES, arrays):
            setattr(self, name, candidate)
        self.rngs, self.membrane = rngs, membrane
        self.time_ms += len(photons)
        self.events += events
        self.photons += photons.sum(axis=0)
        self.initialized = True
        return {'open_channels': channels, 'voltage_mv': voltage}
