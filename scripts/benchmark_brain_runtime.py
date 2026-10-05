"""One normal prepared receptor replay, compared with a serial reference.

Checks every public 10 ms boundary, including full neural state and delayed
events. Does not run a stress matrix or require CUDA. An optional saved runtime
module allows comparison against the implementation before an optimization.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT/'.runtime/numba'))
import numba
import numpy as np
from nexus.brain.receptor_release import ReleaseCurve
from nexus.brain.runtime import Brain, Connectome
from run_receptor_brain_replay import file_hash, load_response

STATE_ARRAYS = ('v', 'g', 'histamine_g', 'excitation_g', 'last_spike', 'enabled',
                'counts', 'queue', 'queue_size', 'graded_queue', 'receptor_queue')


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def replay(brain, ids, voltage, rate, reference_mv):
    def configure():
        brain.set_graded_relays(True)
        brain.configure_receptor_replay(ids, reference_mv=reference_mv)

    configure()
    brain.advance(.0001, receptor_voltage_mv=voltage[:1], receptor_release_hz=rate[:1])
    brain.reset()
    configure()
    wall = 0.
    boundaries = []
    for offset in range(0, len(voltage), 100):
        stop = min(offset+100, len(voltage))
        start = time.perf_counter()
        result = brain.advance((stop-offset)*.0001, trace_ids=ids[:8],
                               receptor_voltage_mv=voltage[offset:stop],
                               receptor_release_hz=rate[offset:stop])
        wall += time.perf_counter()-start
        state = {key: array_hash(getattr(brain, key)) for key in STATE_ARRAYS}
        state.update({key: array_hash(result[key]) for key in ('indices', 'steps', 'voltage_mv')})
        state['unrecorded_spikes'] = result['unrecorded_spikes']
        state['history'] = array_hash(np.asarray(brain.history, dtype=np.int64))
        state['activity'] = array_hash(np.asarray(brain.activity, dtype=np.int64))
        state['activity_pending'] = [brain.activity_pending_ticks, brain.activity_pending_spikes]
        state['rng'] = brain.rng.bit_generator.state
        boundaries.append(state)
    return wall, boundaries, {'total_spikes': int(brain.counts.sum()),
                             'source_spikes': int(brain.counts[brain.receptor_indices].sum()),
                             'relay_spikes': int(brain.counts[brain.graded_indices].sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--response-dir', type=Path, required=True)
    parser.add_argument('--release-curve', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--baseline-file', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--pack', type=Path, default=ROOT/'data/brain-male-cns-v1.0-lif')
    args = parser.parse_args()
    graph = Connectome.load(args.pack, allow_experimental=True)
    ids, voltage, _ = load_response(args.response_dir, graph.registry_sha256)
    curve = ReleaseCurve.load(args.release_curve)
    rate = curve.evaluate(voltage)
    if args.baseline_file:
        spec = importlib.util.spec_from_file_location('nexus.brain._runtime_baseline', args.baseline_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        baseline = module.Brain(graph)
    else:
        baseline = Brain(graph, integration_threads=1)
    candidate = Brain(graph, integration_threads=args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    reference_wall, reference, _ = replay(baseline, ids, voltage, rate, curve.reference_mv)
    print(f'Serial reference: {reference_wall:.3f} seconds', flush=True)
    candidate_wall, candidate_states, spikes = replay(candidate, ids, voltage, rate, curve.reference_mv)
    checks = {key: all(a[key] == b[key] for a, b in zip(reference, candidate_states))
              for key in reference[0]}
    report = {
        'format': 'nexus-brain-runtime-benchmark-1', 'success': all(checks.values()),
        'dataset': graph.snapshot, 'neurons': len(graph.ids), 'edges': len(graph.posts),
        'registry_sha256': graph.registry_sha256, 'manifest_sha256': file_hash(args.pack/'manifest.json'),
        'duration_ms': len(voltage)*.1, 'chunk_ms': 10., 'boundaries_checked': len(reference),
        'receptor_ids': ids, 'integration_threads': args.threads, 'synaptic_delivery': 'serial',
        'baseline': 'saved implementation' if args.baseline_file else 'current serial implementation',
        'baseline_sha256': file_hash(args.baseline_file or ROOT/'src/nexus/brain/runtime.py'),
        'runtime_sha256': file_hash(ROOT/'src/nexus/brain/runtime.py'),
        'benchmark_sha256': file_hash(Path(__file__)),
        'input_sha256': {name: file_hash(args.response_dir/name) for name in ('response.npz', 'report.json')},
        'release_curve_sha256': file_hash(args.release_curve),
        'serial_wall_seconds': reference_wall, 'parallel_wall_seconds': candidate_wall,
        'speedup': reference_wall/candidate_wall, 'parallel_realtime_factor': len(voltage)*.0001/candidate_wall,
        'exact_at_every_boundary': checks,
        'final_state_sha256': {key: candidate_states[-1][key] for key in STATE_ARRAYS},
        'spikes': spikes,
        'environment': {'platform': platform.platform(), 'cpu': platform.processor(),
                        'numba': numba.__version__, 'threading_layer': numba.threading_layer()},
        'limits': ['One ordinary replay and one timing per mode, not a stress matrix or cross-machine guarantee.',
                   'Timing includes public advance and returned trace copies, excluding load, warmup, hashing and export.',
                   'The articulated body, rendering and GUI are excluded from this runtime benchmark.',
                   'This is an execution optimization; receptor release and relay laws remain unfitted.',
                   'Default four-thread execution applies to graphs with at least 10,000 neurons; smaller graphs remain serial.'],
    }
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'success': report['success'], 'speedup': report['speedup'],
                      'serial_seconds': reference_wall, 'parallel_seconds': candidate_wall,
                      'exact': checks}), flush=True)
    if not report['success']:
        raise RuntimeError('Runtime states differ; inspect the report')


if __name__ == '__main__':
    main()
