"""Full-network/body checks of BDN2 walking drive without scheduled exploration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
for name, subdir in [('NUMBA_CACHE_DIR', 'numba'), ('MPLCONFIGDIR', 'matplotlib'), ('FLYGYM_ASSET_CACHE_DIR', 'assets')]:
    os.environ.setdefault(name, str(ROOT / '.runtime' / subdir))
import numpy as np
from nexus.brain.runtime import Brain, Connectome
from nexus.body import FlyBody
from nexus.coupled import CoupledSession
from nexus.sequence_run import record_sequence, write_json


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
            for condition in ['quiet', 'bdn2_input', 'bdn2_blocked', 'motor_blocked']:
                brain = Brain(graph, seed=seed)
                body.reset()
                session = CoupledSession(brain, body, autonomous=True)
                session.command('neural_walking', True)
                ids = session.walking_decoder.ids
                plan = {'format': 'nexus-protocol-1', 'dataset': graph.snapshot, 'duration_ms': 700,
                        'events': [{'at_ms': 0, 'action': 'release'},
                                   {'at_ms': 100, 'action': 'stimulate', 'ids': ids, 'rate_hz': 200},
                                   {'at_ms': 400, 'action': 'release'}]}
                if condition == 'quiet':
                    plan['events'].pop(1)
                elif condition == 'bdn2_blocked':
                    plan['events'].insert(2, {'at_ms': 100, 'action': 'silence', 'ids': ids})
                elif condition == 'motor_blocked':
                    session.command('bridge_enabled', False)
                output = args.output_dir / f'{seed}-{condition}'
                report = record_sequence(session, plan, output)
                trace = [json.loads(line) for line in (output / 'trace.jsonl').read_text().splitlines()]
                stimulus = [r for r in trace if 100 <= r['from_ms'] and r['to_ms'] <= 400]
                row = {'seed': seed, 'condition': condition, 'total_spikes': int(brain.counts.sum()),
                       'bdn2_spikes': brain.counts[session.walking_decoder.indices].tolist(),
                       'maximum_forward_drive_during_input': max(r['walking_decoder']['drive'] for r in stimulus),
                       'longitudinal_displacement_during_input_mm': sum(r['longitudinal_delta_mm'] for r in stimulus),
                       'minimum_upright': min(r['upright'] for r in trace),
                       'authored_behavior_elapsed_s': session.behavior.elapsed,
                       'inputs_released': report['inputs_released']}
                if condition in ['quiet', 'bdn2_blocked', 'motor_blocked']:
                    assert row['maximum_forward_drive_during_input'] == 0
                else:
                    assert row['maximum_forward_drive_during_input'] > 0
                assert session.behavior.elapsed == 0 and report['inputs_released']
                rows.append(row)
                trials[condition] = (brain.counts.copy(), np.array([r['position_mm'] for r in trace]),
                                     brain.rng.bit_generator.state)
                print(json.dumps(row), flush=True)
            np.testing.assert_array_equal(trials['bdn2_input'][0], trials['motor_blocked'][0])
            for condition in ['bdn2_blocked', 'motor_blocked']:
                assert trials[condition][2] == trials['bdn2_input'][2]
                difference = float(np.linalg.norm(trials[condition][1] - trials['bdn2_input'][1], axis=1).max())
                assert difference > 0
                comparisons.append({'seed': seed, 'comparison': condition,
                                    'identical_input_random_draws': True, 'maximum_body_difference_mm': difference})
        write_json(args.output_dir / 'report.json', {
            'format': 'nexus-neural-walking-results-1', 'success': True, 'dataset': graph.snapshot,
            'manifest_sha256': hashlib.sha256((pack / 'manifest.json').read_bytes()).hexdigest(),
            'duration_ms_per_trial': 700, 'trials': rows, 'comparisons': comparisons,
            'limits': ['The experiment supplies BDN2 input; it does not demonstrate spontaneous exploration.',
                       'Direct descending readout and FlyGym coordination bypass a reconstructed VNC-to-muscle pathway.',
                       'Rate filter and motor gains are engineered. Output blockade ends at release.']})
    finally:
        body.close()


if __name__ == '__main__':
    main()
