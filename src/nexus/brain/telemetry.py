"""Shared neural telemetry for independent and coupled workers."""
import hashlib
import time
import numpy as np
from .targets import readout_ids


class NeuralTelemetry:
    def __init__(self, brain):
        self.order_hash = hashlib.sha256(brain.graph.ids.astype('<i8').tobytes()).hexdigest()
        ids = readout_ids('mn9', brain.graph) if brain.graph.snapshot in ('630', 'male-cns:v1.0') else []
        self.mn9 = np.array([brain.lookup[int(i)] for i in ids if int(i) in brain.lookup], dtype=np.int32)
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
                  'neuron_order_sha256': self.order_hash, 'dataset': brain.graph.snapshot,
                  'model': brain.graph.model,
                  'active_indices': active.tolist(), 'active_steps': brain.last_spike[active].tolist(),
                  'neurons': len(brain.graph.ids), 'edges': len(brain.graph.posts),
                  'total_spikes': int(brain.counts.sum()), 'window_spikes': int(delta.sum()),
                  'mn9_spikes': int(brain.counts[i].sum()) if len(i) else None,
                  'mn9_hz': float(delta[i].sum()/duration) if len(i) and duration else 0,
                  'mn9_mv': float(brain.v[i].mean()) if len(i) else None,
                  'mn9_cells': [{'id': str(brain.graph.ids[j]), 'spikes': int(brain.counts[j]),
                                 'voltage_mv': float(brain.v[j])} for j in i],
                  'mn9_aggregation': 'summed spikes and rates; mean voltage',
                  'realtime_factor': duration/max(now-self.last_wall, 1e-9),
                  'stimulated_ids': [str(brain.graph.ids[j]) for j in brain.inputs],
                  'manual_ids': [str(brain.graph.ids[j]) for j in brain.manual_inputs],
                  'sensory_ids': [str(brain.graph.ids[j]) for j in brain.sensory_inputs],
                  'input_rates_hz': brain.rates.tolist(), 'sensory_rate_hz': brain.sensory_rate,
                  'rate_hz': float(brain.rates[0]) if len(brain.rates) else 0,
                  'silenced_count': int((brain.output_gain==0).sum()),
                  'inhibition_gain': brain.inhibition_gain,
                  'nominal_temperature_c': brain.nominal_temperature,
                  'circuit_inputs': {name: {'ids': [str(brain.graph.ids[i]) for i in targets], 'rate_hz': rate}
                                     for name, (targets, rate) in brain.circuit_inputs.items()},
                  'population_hz_per_neuron': float(delta.sum()/duration/len(brain.counts)) if duration else 0.,
                  'protocol': None if not protocol else {'name': protocol.description.get('name', 'Sequence'),
                                                       'completed': protocol.completed, 'event_cursor': protocol.cursor},
                  'raster_steps': [s for s,j in raster], 'raster_indices': [j for s,j in raster],
                  'interventions': list(brain.events), 'brain_drives_body': coupled}
        self.last_counts[:] = brain.counts
        self.last_step, self.last_wall = brain.step, now
        return result
