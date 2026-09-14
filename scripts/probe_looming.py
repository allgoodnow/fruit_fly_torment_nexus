"""Matched full-network/body trials of dynamic visual input and pathway blockade."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
for name, relative in [('NUMBA_CACHE_DIR', '.runtime/numba'), ('MPLCONFIGDIR', '.runtime/matplotlib'),
                       ('FLYGYM_ASSET_CACHE_DIR', '.runtime/assets')]:
    os.environ.setdefault(name, str(ROOT / relative))
import numpy as np
from nexus.body import FlyBody
from nexus.brain.runtime import Brain, Connectome
from nexus.brain.scenarios import scenario_protocol
from nexus.coupled import CoupledSession
from nexus.sequence_run import record_sequence, write_json


def complete_results(output):
    """Separate stimulation from recovery; release also restores blocked outputs."""
    output = Path(output)
    report = json.loads((output / 'report.json').read_text())
    if not report.get('success') or len(report['trials']) != 15:
        raise ValueError('Expected all fifteen completed paired trials')
    for row in report['trials']:
        trial = output / f"{row['seed']}-{row['condition']}"
        if hashlib.sha256((trial / 'report.json').read_bytes()).hexdigest() != row['report_sha256']:
            raise ValueError('Trial report changed since collection')
        trace = [json.loads(line) for line in (trial / 'trace.jsonl').read_text().splitlines()]
        for phase, start, end in [('during_input', 100, 600), ('after_release', 600, 800)]:
            samples = [r for r in trace if start <= r['from_ms'] and r['to_ms'] <= end]
            row[phase] = {'maximum_escape': max(r['motor_effects']['escape'] for r in samples),
                          'mean_escape': float(np.mean([r['motor_effects']['escape'] for r in samples])),
                          'first_escape_ms': next((r['to_ms'] for r in samples
                                                   if r['motor_effects']['escape'] > .05), None)}
        if row['condition'] == 'both_blocked' and row['during_input']['maximum_escape'] != 0:
            raise RuntimeError('Unexpected escape signal while both visual cohorts are blocked')
    limitation = ('Output blockade ends at release (600 ms). Queued spikes and retained neural state '
                  'can produce a response afterward; phase metrics separate this from stimulation.')
    if limitation not in report['limits']:
        report['limits'].append(limitation)
    write_json(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pack = ROOT / 'data/brain-male-cns-v1.0-lif'
    graph = Connectome.load(pack, allow_experimental=True)
    body = FlyBody(render=False)
    rows, comparisons = [], []
    try:
        for seed in [73100, 73101, 73102]:
            trials = {}
            for condition in ['approach', 'size_blocked', 'velocity_blocked', 'both_blocked', 'motor_blocked']:
                brain = Brain(graph, seed=seed)
                body.reset()
                session = CoupledSession(brain, body, autonomous=True)
                plan = scenario_protocol('defensive', dataset=graph.snapshot,
                                         baseline_ms=100, stimulus_ms=500, recovery_ms=200)
                blocked = {'size_blocked': ['looming'], 'velocity_blocked': ['looming_velocity'],
                           'both_blocked': ['looming', 'looming_velocity']}.get(condition, [])
                for index, name in enumerate(blocked):
                    plan['events'].insert(2 + index, {'at_ms': 100, 'action': 'silence',
                                                     'ids': brain.circuit_ids(name)})
                if condition == 'motor_blocked':
                    session.command('bridge_enabled', False)
                output = args.output_dir / f'{seed}-{condition}'
                report = record_sequence(session, plan, output)
                trace = [json.loads(line) for line in (output / 'trace.jsonl').read_text().splitlines()]
                gf = brain.resolve(brain.circuit_ids('giant_fiber'))
                row = {'seed': seed, 'condition': condition, 'total_spikes': int(brain.counts.sum()),
                       'gf_spikes': brain.counts[gf].tolist(),
                       'maximum_escape': max(r['motor_effects']['escape'] for r in trace),
                       'maximum_disruption': max(r['motor_effects']['disruption'] for r in trace),
                       'minimum_upright': min(r['upright'] for r in trace),
                       'inputs_released': report['inputs_released'] and brain.looming_input is None,
                       'counts_sha256': report['end_counts_sha256'],
                       'report_sha256': hashlib.sha256((output / 'report.json').read_bytes()).hexdigest()}
                if not row['inputs_released']:
                    raise RuntimeError('Approach inputs were not released')
                rows.append(row)
                trials[condition] = (brain.counts.copy(), np.array([r['position_mm'] for r in trace]),
                                     brain.rng.bit_generator.state)
                print(json.dumps(row), flush=True)
            np.testing.assert_array_equal(trials['approach'][0], trials['motor_blocked'][0])
            direct = brain.resolve(brain.circuit_ids('looming') + brain.circuit_ids('looming_velocity'))
            downstream = np.ones(len(brain.counts), dtype=bool)
            downstream[direct] = False
            for condition in ['size_blocked', 'velocity_blocked', 'both_blocked', 'motor_blocked']:
                if trials[condition][2] != trials['approach'][2]:
                    raise RuntimeError('Matched conditions consumed different input draws')
                changed = int(np.count_nonzero(trials['approach'][0][downstream] != trials[condition][0][downstream]))
                difference = float(np.linalg.norm(trials['approach'][1] - trials[condition][1], axis=1).max())
                comparisons.append({'seed': seed, 'comparison': condition,
                                    'identical_input_random_draws': True,
                                    'downstream_cells_with_changed_counts': changed,
                                    'maximum_body_position_difference_mm': difference})
                if condition != 'motor_blocked' and changed == 0:
                    raise RuntimeError(f'{condition} did not change downstream neural counts')
        report = {'format': 'nexus-looming-results-1', 'success': True, 'dataset': graph.snapshot,
                  'neurons': len(graph.ids), 'duration_ms_per_trial': 800,
                  'manifest_sha256': hashlib.sha256((pack / 'manifest.json').read_bytes()).hexdigest(),
                  'trials': rows, 'comparisons': comparisons,
                  'limits': ['Matched pathway blockade checks causal model connectivity, not biological validity.',
                             'Geometry and input envelopes are a virtual stimulus; firing rates are unfitted.',
                             'Body commands remain engineered, with no flight or subjective-state inference.']}
        write_json(args.output_dir / 'report.json', report)
        complete_results(args.output_dir)
    finally:
        body.close()


if __name__ == '__main__':
    main()
