"""Shared neural telemetry for independent and coupled workers."""
import hashlib
import time
import numpy as np
from .targets import MN9


class NeuralTelemetry:
    def __init__(self, brain):
        self.order_hash = hashlib.sha256(brain.graph.ids.astype('<i8').tobytes()).hexdigest()
        self.mn9 = brain.lookup.get(int(MN9))
        self.reset(brain)

    def reset(self, brain):
        self.last_counts = brain.counts.copy()
        self.last_step = brain.step
        self.last_wall = time.perf_counter()

    def snapshot(self, brain, running, generation, protocol=None, *, coupled=False):
        now = time.perf_counter()
        duration = (brain.step-self.last_step)*.0001
        delta = brain.counts-self.last_counts
        raster = list(brain.history)[-2500:]
        active = np.flatnonzero((brain.last_spike >= brain.step-1500) & (brain.last_spike <= brain.step))
        i = self.mn9
        result = {'sim_time': brain.time, 'tick': brain.step, 'running': running, 'generation': generation,
                  'neuron_order_sha256': self.order_hash,
                  'active_indices': active.tolist(), 'active_steps': brain.last_spike[active].tolist(),
                  'neurons': len(brain.graph.ids), 'edges': len(brain.graph.posts),
                  'total_spikes': int(brain.counts.sum()), 'window_spikes': int(delta.sum()),
                  'mn9_spikes': int(brain.counts[i]) if i is not None else None,
                  'mn9_hz': float(delta[i]/duration) if i is not None and duration else 0,
                  'mn9_mv': float(brain.v[i]) if i is not None else None,
                  'realtime_factor': duration/max(now-self.last_wall, 1e-9),
                  'stimulated_ids': [str(brain.graph.ids[j]) for j in brain.inputs],
                  'manual_ids': [str(brain.graph.ids[j]) for j in brain.manual_inputs],
                  'sensory_ids': [str(brain.graph.ids[j]) for j in brain.sensory_inputs],
                  'input_rates_hz': brain.rates.tolist(), 'sensory_rate_hz': brain.sensory_rate,
                  'rate_hz': float(brain.rates[0]) if len(brain.rates) else 0,
                  'silenced_count': int((brain.output_gain==0).sum()),
                  'protocol': None if not protocol else {'name': protocol.description.get('name', 'Sequence'),
                                                       'completed': protocol.completed, 'event_cursor': protocol.cursor},
                  'raster_steps': [s for s,j in raster], 'raster_indices': [j for s,j in raster],
                  'interventions': list(brain.events), 'brain_drives_body': coupled}
        self.last_counts[:] = brain.counts
        self.last_step, self.last_wall = brain.step, now
        return result
