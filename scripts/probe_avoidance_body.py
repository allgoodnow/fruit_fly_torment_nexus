"""Paired physical assay for the authored candidate-aversion motor connection."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
for name, relative in [('NUMBA_CACHE_DIR', '.runtime/numba'), ('MPLCONFIGDIR', '.runtime/matplotlib'),
                       ('FLYGYM_ASSET_CACHE_DIR', '.runtime/assets')]:
    os.environ.setdefault(name, str(ROOT/relative))
import numpy as np
from nexus import __version__
from nexus.body import FlyBody
from nexus.brain.runtime import Brain, Connectome
from nexus.coupled import CoupledSession
from nexus.sequence_run import record_sequence, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = ROOT/'experiments/male-cns-nociception-recovery.json'
    protocol = json.loads(source.read_text())
    pack = ROOT/'data/brain-male-cns-v1.0-lif'
    graph = Connectome.load(pack, allow_experimental=True)
    body = FlyBody(render=False)
    results, comparisons = [], []
    try:
        for seed in [73100, 73101, 73102]:
            trials = {}
            conditions = ['intact', 'avoidance_blocked']
            if seed == 73100:
                conditions += ['source_outputs_blocked', 'candidate_outputs_blocked', 'baseline']
            for condition in conditions:
                brain = Brain(graph, seed=seed)
                body.reset()
                session = CoupledSession(brain, body, autonomous=True)
                plan = copy.deepcopy(protocol)
                if condition == 'avoidance_blocked':
                    session.command('avoidance_enabled', False)
                elif condition in ('source_outputs_blocked', 'candidate_outputs_blocked'):
                    circuit = 'nociception_proxy' if condition == 'source_outputs_blocked' else 'aversion_proxy'
                    plan['events'].insert(2, {'at_ms': 100, 'action': 'silence', 'ids': brain.circuit_ids(circuit)})
                elif condition == 'baseline':
                    plan['events'] = [{'at_ms': 0, 'action': 'release'}]
                output = args.output_dir/f'{seed}-{condition}'
                report = record_sequence(session, plan, output)
                trace = [json.loads(line) for line in (output/'trace.jsonl').read_text().splitlines()]
                stimulus = [r for r in trace if 100 < r['to_ms'] <= 400]
                levels = [r['motor_effects']['avoidance'] for r in stimulus]
                row = {'seed': seed, 'condition': condition, 'spikes': report['total_spikes'],
                       'peak_avoidance_during_input': max(levels),
                       'minimum_drive_during_input': min(r['motor_command']['drive'] for r in stimulus),
                       'peak_avoidance_turn': max(r['motor_command']['avoidance_turn'] for r in trace),
                       'final_avoidance': trace[-1]['motor_effects']['avoidance'],
                       'minimum_upright': min(r['upright'] for r in trace),
                       'final_upright': trace[-1]['upright'],
                       'final_behavior': trace[-1]['behavior']['state'],
                       'recovery': report['recovery']['phase'],
                       'final_position_mm': report['final_position_mm'],
                       'report_sha256': hashlib.sha256((output/'report.json').read_bytes()).hexdigest()}
                results.append(row)
                trials[condition] = (brain.counts.copy(), np.array([r['position_mm'] for r in trace]), row)
                print(json.dumps(row), flush=True)
            intact, blocked = trials['intact'], trials['avoidance_blocked']
            np.testing.assert_array_equal(intact[0], blocked[0])
            deviation = float(np.linalg.norm(intact[1]-blocked[1], axis=1).max())
            assert intact[2]['peak_avoidance_during_input'] > .2
            assert intact[2]['minimum_drive_during_input'] < .8
            assert blocked[2]['peak_avoidance_during_input'] == 0
            assert deviation > 1., deviation
            assert intact[2]['final_avoidance'] < .05
            assert intact[2]['final_upright'] > .8
            comparisons.append({'seed': seed, 'neural_counts_identical_with_avoidance_blocked': True,
                                'maximum_position_difference_mm': deviation})
            if seed == 73100:
                for condition in ['source_outputs_blocked', 'candidate_outputs_blocked', 'baseline']:
                    assert trials[condition][2]['peak_avoidance_during_input'] == 0
                old = ROOT/'runs/arena_v014_sequence/counts.npz'
                if old.exists():
                    with np.load(old) as prior:
                        np.testing.assert_array_equal(intact[0], prior['end_counts'])
                    comparisons[-1]['matches_v014_neural_counts'] = True
        report = {'format': 'nexus-avoidance-body-assay-1', 'version': __version__, 'success': True,
                  'dataset': graph.snapshot, 'neurons': len(graph.ids), 'duration_ms_per_trial': 2400,
                  'protocol_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'manifest_sha256': hashlib.sha256((pack/'manifest.json').read_bytes()).hexdigest(),
                  'trials': results, 'comparisons': comparisons,
                  'interpretation': 'Authored slowdown and turn driven by measured candidate spikes; not a validated biological avoidance gait or subjective-state measurement.'}
        write_json(args.output_dir/'summary.json', report)
    finally:
        body.close()


if __name__ == '__main__':
    main()
