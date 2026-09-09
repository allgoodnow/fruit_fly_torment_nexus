"""Paired full-network assays of source-mapped abdominal md homolog candidates."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT/'.runtime/numba'))
import numpy as np
from nexus.brain.runtime import Brain, Connectome
from nexus.brain.motor import SteeringDecoder
from nexus.brain.motor_effects import MotorEffects


def trial(graph, cohort, seed, condition, rate=100):
    brain = Brain(graph, seed=seed)
    sensory = brain.resolve(cohort['ids'])
    relays = brain.resolve(cohort['ascending_relay_ids'])
    groups = {'sensory': sensory, 'ascending': relays}
    groups.update({name: brain.resolve(ids) for name, ids in cohort['target_ids'].items()})
    effects, steering = MotorEffects(brain), SteeringDecoder(brain)
    hashes = hashlib.sha256()
    records = []
    peak_escape = peak_disruption = 0.
    started = time.perf_counter()
    for phase, chunks in [('baseline', 5), ('stimulation', 30), ('recovery', 15)]:
        if phase == 'stimulation':
            brain.stimulate(cohort['ids'], rate)
            if condition == 'source_outputs_blocked':
                brain.silence(cohort['ids'])
            elif condition == 'ascending_outputs_blocked':
                brain.silence(cohort['ascending_relay_ids'])
        elif phase == 'recovery':
            saved = {k: getattr(brain, k).copy() for k in ['v','g','counts','queue','queue_size']}
            rng = copy.deepcopy(brain.rng.bit_generator.state)
            brain.release()
            for k, array in saved.items():
                np.testing.assert_array_equal(getattr(brain, k), array)
            assert rng == brain.rng.bit_generator.state and not brain.inputs.size
        before = brain.counts.copy()
        for _ in range(chunks):
            old = brain.counts.copy()
            events = brain.rng.random((100, len(brain.inputs))) < brain.rates*.0001
            hashes.update(np.array([brain.step, len(brain.inputs)], dtype='<i8').tobytes())
            hashes.update(brain.inputs.astype('<i4').tobytes())
            hashes.update(events.tobytes())
            brain.advance(.01, input_events=events)
            delta = brain.counts-old
            steering.observe(delta[steering.indices], .01, brain.output_gain[steering.indices])
            effects.observe(delta, .01, brain.output_gain, brain.inputs)
            peak_escape = max(peak_escape, effects.output()['escape'])
            peak_disruption = max(peak_disruption, effects.output()['disruption'])
        counts = brain.counts-before
        direct = int(counts[sensory].sum())
        records.append({'phase': phase, 'total_spikes': int(counts.sum()),
                        'non_source_spikes': int(counts.sum())-direct,
                        'groups': {name: {'spikes': int(counts[indices].sum()),
                                          'active_cells': int(np.count_nonzero(counts[indices]))}
                                   for name, indices in groups.items()},
                        'population_counts_sha256': hashlib.sha256(counts.tobytes()).hexdigest()})
    assert not brain.inputs.size and brain.inhibition_gain == 1 and np.all(brain.output_gain == 1)
    return {'seed': seed, 'condition': condition, 'rate_hz': rate, 'duration_ms': 500,
            'external_input_sha256': hashes.hexdigest(), 'phases': records,
            'release_preserved_state': True,
            'finite_state': bool(np.isfinite(brain.v).all() and np.isfinite(brain.g).all()),
            'peak_escape_proxy': peak_escape, 'peak_motor_disruption_proxy': peak_disruption,
            'final_steering': steering.output(1), 'wall_seconds': time.perf_counter()-started}


def main():
    cohort_path = ROOT/'experiments/nociception-cohort-v1.json'
    cohort = json.loads(cohort_path.read_text())
    directory = ROOT/'data/brain-male-cns-v1.0-lif'
    graph = Connectome.load(directory, allow_experimental=True)
    assert cohort['dataset'] == graph.snapshot
    rows = []
    for seed in [73100, 73101, 73102]:
        matched = []
        for condition in ['intact', 'source_outputs_blocked', 'ascending_outputs_blocked']:
            row = trial(graph, cohort, seed, condition)
            rows.append(row)
            matched.append(row)
            stim = row['phases'][1]
            print(f'{seed} {condition}: {stim["total_spikes"]:,} spikes, targets {stim["groups"]["aversion_candidate"]["spikes"]}, escape {row["peak_escape_proxy"]:.3f}', flush=True)
        assert len({r['external_input_sha256'] for r in matched}) == 1
        assert matched[1]['phases'][1]['non_source_spikes'] == 0
        assert matched[0]['phases'][1]['non_source_spikes'] > 0
    result = {'format': 'nexus-nociception-assay-1', 'success': True, 'dataset': graph.snapshot,
              'cohort_sha256': hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
              'pack_manifest_sha256': hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest(),
              'model': graph.model, 'trials': rows,
              'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__),ROOT/'src/nexus/brain/runtime.py',ROOT/'src/nexus/datasets/nociception.py']},
              'interpretation': 'Tests propagation from source-mapped sensory candidates and sensitivity to relay output blockade. Does not establish subjective pain or identify genetic driver lines.',
              'release': 'Release removes external inputs and restores outputs without resetting neural state; delayed activity may persist.',
              'ui_changed': False}
    (ROOT/'experiments/nociception-v1-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('All paired-input and source-blockade checks passed.', flush=True)


if __name__ == '__main__':
    main()
