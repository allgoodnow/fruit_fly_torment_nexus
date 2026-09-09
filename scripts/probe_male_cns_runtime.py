"""Headless MaleCNS numerical, intervention, and CPU acceptance experiment."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT/'.runtime/numba'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.runtime/matplotlib'))
import numpy as np
from nexus.brain.runtime import Brain, Connectome
from nexus.brain.motor import SteeringDecoder
from nexus.brain.motor_effects import MotorEffects
from nexus.brain.telemetry import NeuralTelemetry
from nexus.brain.targets import readout_ids


def reference_check(graph):
    """Compare a source-derived subgraph against independent Brian2 equations."""
    import brian2 as b
    probe = Brain(graph)
    target_ids = probe.circuit_ids('warmth')+probe.circuit_ids('aversion_proxy')
    target = probe.resolve(target_ids)
    # Include the 96 strongest output targets, plus all directly driven cells.
    outs, strength = [], []
    for pre in target:
        sl = slice(graph.offsets[pre], graph.offsets[pre+1])
        outs.extend(graph.posts[sl].tolist())
        strength.extend(np.abs(graph.weights[sl]).tolist())
    ranking = np.zeros(len(graph.ids))
    np.add.at(ranking, outs, strength)
    selected = np.unique(np.r_[target, np.argsort(ranking)[-96:]])
    lookup = {int(i): j for j, i in enumerate(selected)}
    pre, post, weights = [], [], []
    for i in selected:
        for edge in range(graph.offsets[i], graph.offsets[i+1]):
            j = int(graph.posts[edge])
            if j in lookup:
                pre.append(lookup[int(i)])
                post.append(lookup[j])
                weights.append(graph.weights[edge])
    subgraph = Connectome.from_edges(graph.ids[selected], pre, post, weights)
    brain = Brain(subgraph)
    brain.stimulate(target_ids, 300)
    events = np.random.default_rng(73100).random((400, len(target_ids))) < .03
    actual = brain.advance(.04, input_events=events, trace_ids=target_ids)
    b.start_scope()
    b.prefs.codegen.target = 'numpy'
    b.defaultclock.dt = .1*b.ms
    neu = b.NeuronGroup(len(selected),
        'dv/dt=(-52*mV-v+g)/(20*ms):volt (unless refractory)\n'
        'dg/dt=-g/(5*ms):volt (unless refractory)\nrfc:second',
        threshold='v > -45*mV', reset='v=-52*mV; g=0*mV', refractory='rfc', method='linear')
    neu.v = -52*b.mV
    neu.rfc = 2.2*b.ms
    driven = brain.resolve(target_ids)
    neu.rfc[driven] = 0*b.ms
    syn = b.Synapses(neu, neu, 'w:volt', on_pre='g+=w', delay=1.8*b.ms)
    syn.connect(i=np.repeat(np.arange(len(selected)), np.diff(subgraph.offsets)), j=subgraph.posts)
    syn.w = subgraph.weights*b.mV
    @b.network_operation(when='synapses', order=0)
    def replay():
        tick = round(float(b.defaultclock.t/b.ms)*10)
        chosen = driven[events[tick]]
        chosen = chosen[np.asarray(neu.not_refractory[:])[chosen]]
        neu.v[chosen] += 68.75*b.mV
    spikes = b.SpikeMonitor(neu)
    b.Network(neu, syn, replay, spikes).run(40*b.ms)
    np.testing.assert_array_equal(actual['indices'], spikes.i[:])
    np.testing.assert_array_equal(actual['steps'], np.rint(spikes.t[:]/b.ms*10).astype(int))
    np.testing.assert_allclose(brain.v, neu.v[:]/b.mV, atol=1e-8, rtol=0)
    np.testing.assert_allclose(brain.g, neu.g[:]/b.mV, atol=1e-8, rtol=0)
    return {'scope': 'source-derived induced subgraph; not full-network Brian comparison',
            'neurons': len(selected), 'edges': len(weights), 'duration_ms': 40,
            'identical_spikes': len(actual['indices']),
            'max_voltage_error_mv': float(np.max(np.abs(brain.v-neu.v[:]/b.mV)))}


def trial(graph, condition, seed, duration=.2):
    brain = Brain(graph, seed=seed)
    decoder, effects = SteeringDecoder(brain), MotorEffects(brain)
    timings = []
    input_hash = hashlib.sha256()
    def advance(seconds):
        for _ in range(round(seconds/.01)):
            before = brain.counts.copy()
            start = time.perf_counter()
            events = brain.rng.random((100, len(brain.inputs))) < brain.rates*.0001
            input_hash.update(np.array([brain.step, len(brain.inputs)], dtype='<i8').tobytes())
            input_hash.update(brain.inputs.astype('<i4').tobytes())
            input_hash.update(events.tobytes())
            brain.advance(.01, input_events=events)
            timings.append(time.perf_counter()-start)
            delta = brain.counts-before
            decoder.observe(delta[decoder.indices], .01, brain.output_gain[decoder.indices])
            effects.observe(delta, .01, brain.output_gain, brain.inputs)
    advance(.05)
    assert not brain.counts.any()
    if condition in ('warmth', 'warmth_aversion', 'warmth_aversion_blocked', 'heat_overload'):
        brain.set_heat(40)
    if 'aversion' in condition:
        brain.set_circuit_input('aversion_proxy', 200)
    if condition == 'warmth_aversion_blocked':
        brain.silence(brain.circuit_ids('aversion_proxy'))
    if condition == 'looming':
        brain.set_circuit_input('looming', 200)
    if condition.startswith('steering_'):
        brain.stimulate(readout_ids(condition, graph), 200)
    if condition == 'heat_overload':
        brain.set_inhibition_gain(.25)
    driven = brain.inputs.copy()
    advance(duration)
    state = {name: getattr(brain, name).copy() for name in ['v', 'g', 'counts', 'queue', 'queue_size']}
    rng = copy.deepcopy(brain.rng.bit_generator.state)
    event_tick = brain.step
    steering, motor = decoder.output(1), effects.output()
    brain.release()
    for name, value in state.items():
        np.testing.assert_array_equal(getattr(brain, name), value)
    assert brain.rng.bit_generator.state == rng and brain.step == event_tick
    assert not len(brain.inputs) and brain.inhibition_gain == 1 and np.all(brain.output_gain == 1)
    advance(.05)
    monitor = NeuralTelemetry(brain)
    packet = monitor.snapshot(brain, False, 0)
    return {'condition': condition, 'seed': seed, 'simulated_seconds': brain.time,
            'sampled_input_sha256': input_hash.hexdigest(),
            'total_spikes': int(brain.counts.sum()),
            'stimulus_spikes': int(state['counts'].sum()),
            'direct_stimulus_spikes': int(state['counts'][driven].sum()),
            'candidate_stimulus_spikes': int(state['counts'][brain.resolve(brain.circuit_ids('aversion_proxy'))].sum()),
            'stimulus_counts_sha256': hashlib.sha256(state['counts'].tobytes()).hexdigest(),
            'active_cells': int(np.count_nonzero(brain.counts)),
            'release_preserved_state': True, 'finite_state': bool(np.isfinite(brain.v).all() and np.isfinite(brain.g).all()),
            'stimulus_end_steering': steering, 'stimulus_end_motor_proxy': motor,
            'mn9_cells': packet['mn9_cells'],
            'advance_wall_seconds': sum(timings), 'advance_realtime_factor': brain.time/sum(timings),
            'median_chunk_wall_ms': float(np.median(timings)*1000),
            'p95_chunk_wall_ms': float(np.percentile(timings, 95)*1000)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, default=ROOT/'data/brain-male-cns-v1.0-lif')
    parser.add_argument('--report', type=Path, default=ROOT/'experiments/male-cns-runtime-v1-results.json')
    args = parser.parse_args()
    started = time.perf_counter()
    graph = Connectome.load(args.pack, allow_experimental=True)
    load_seconds = time.perf_counter()-started
    warmup = Brain(graph)
    warmup.advance(.01)
    del warmup
    comparison = reference_check(graph)
    print('Independent numerical check:', comparison, flush=True)
    rows = []
    conditions = ['baseline', 'warmth', 'looming', 'aversion', 'heat_overload', 'steering_left', 'steering_right']
    trials = [(condition, 73100) for condition in conditions]
    trials += [(condition, seed) for seed in [73100, 73101, 73102]
               for condition in ['warmth_aversion', 'warmth_aversion_blocked']]
    for condition, seed in trials:
        row = trial(graph, condition, seed)
        rows.append(row)
        print(f'{condition} / {seed}: {row["total_spikes"]:,} spikes, {row["advance_realtime_factor"]:.2f}x neural realtime', flush=True)
    pairs = []
    for seed in [73100, 73101, 73102]:
        a, b = [r for r in rows if r['seed'] == seed and r['condition'] in ['warmth_aversion', 'warmth_aversion_blocked']]
        assert a['sampled_input_sha256'] == b['sampled_input_sha256']
        assert min(a['candidate_stimulus_spikes'], b['candidate_stimulus_spikes']) > 0
        pairs.append({'seed': seed, 'matched_external_input_sha256': a['sampled_input_sha256'],
                      'direct_spikes_active_blocked': [a['direct_stimulus_spikes'], b['direct_stimulus_spikes']],
                      'candidate_spikes_active_blocked': [a['candidate_stimulus_spikes'], b['candidate_stimulus_spikes']],
                      'total_stimulus_spike_difference_active_minus_blocked': a['stimulus_spikes']-b['stimulus_spikes'],
                      'population_counts_differ': a['stimulus_counts_sha256'] != b['stimulus_counts_sha256']})
    # An alternative glutamate sign is a sensitivity experiment, not a second fit.
    import pandas as pd
    from nexus.datasets.male_cns import FILES
    from nexus.datasets.male_cns_runtime import transmitter_signs
    _, labels = transmitter_signs(graph.ids, pd.read_feather(ROOT/'data/raw/male-cns-v1.0'/FILES['neurotransmitters']))
    altered = graph.weights.copy()
    for pre in np.flatnonzero((labels == 'glutamate').to_numpy()):
        altered[graph.offsets[pre]:graph.offsets[pre+1]] *= -1
    sensitivity_graph = Connectome(graph.ids, graph.offsets, graph.posts, altered,
                                  graph.snapshot, dict(graph.model, sensitivity='glutamate sign +1'), graph.circuits)
    sensitivity = trial(sensitivity_graph, 'warmth', 73100)
    print('Glutamate-positive sensitivity:', sensitivity['total_spikes'], 'spikes', flush=True)
    report = {'format': 'nexus-male-cns-runtime-assay-1', 'success': True, 'dataset': graph.snapshot,
              'neurons': len(graph.ids), 'edges': len(graph.posts), 'model': graph.model,
              'pack_manifest_sha256': hashlib.sha256((args.pack/'manifest.json').read_bytes()).hexdigest(),
              'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [Path(__file__), ROOT/'src/nexus/brain/runtime.py', ROOT/'src/nexus/datasets/male_cns_runtime.py']},
              'load_seconds': load_seconds, 'max_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              'numerical_reference': comparison, 'trials': rows, 'aversion_output_ablation': pairs,
              'glutamate_positive_sensitivity': sensitivity,
              'limitations': ['Experimental sign and dynamical assumptions; no biological MaleCNS calibration',
                              'Aversion candidate output test does not validate nociception or felt pain',
                              'No coupled 3D body or GUI migration in this assay',
                              'Timing measures neural advance only; rendering and body physics excluded']}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Report saved: {args.report}', flush=True)


if __name__ == '__main__':
    main()
