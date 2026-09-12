"""Paired full-network/body checks of the thermal md input and its outgoing effects."""
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
from nexus.brain.scenarios import scenario_protocol
from nexus.coupled import CoupledSession
from nexus.sequence_run import record_sequence, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pack = ROOT/'data/brain-male-cns-v1.0-lif'
    graph = Connectome.load(pack, allow_experimental=True)
    body = FlyBody(render=False)
    results, comparisons = [], []
    try:
        for seed in [73100, 73101, 73102]:
            trials = {}
            for condition in ['heat', 'md_outputs_blocked', 'motor_blocked', 'warmth_only']:
                b = Brain(graph, seed=seed)
                body.reset()
                s = CoupledSession(b, body, autonomous=True)
                plan = scenario_protocol('heat', dataset=graph.snapshot, baseline_ms=100,
                                         stimulus_ms=300, recovery_ms=300, celsius=40)
                if condition == 'md_outputs_blocked':
                    plan['events'].insert(2, {'at_ms':100, 'action':'silence',
                                             'ids':b.circuit_ids('nociception_proxy')})
                elif condition == 'motor_blocked':
                    s.command('bridge_enabled', False)
                elif condition == 'warmth_only':
                    plan['events'][1] = {'at_ms':100, 'action':'circuit', 'name':'warmth', 'rate_hz':300}
                out = args.output_dir/f'{seed}-{condition}'
                run = record_sequence(s, plan, out)
                trace = [json.loads(line) for line in (out/'trace.jsonl').read_text().splitlines()]
                md = b.resolve(b.circuit_ids('nociception_proxy'))
                row = {'seed':seed, 'condition':condition, 'total_spikes':int(b.counts.sum()),
                       'md_spikes':int(b.counts[md].sum()),
                       'mdn_spikes':int(b.counts[s.motor_effects.mdn].sum()),
                       'maximum_retreat':max(r['motor_effects']['retreat'] for r in trace),
                       'maximum_disruption':max(r['motor_effects']['disruption'] for r in trace),
                       'minimum_upright':min(r['upright'] for r in trace),
                       'final_upright':trace[-1]['upright'],
                       'final_recovery_phase':run['recovery']['phase'],
                       'inputs_released':run['inputs_released'],
                       'thermal_channel_released':not b.thermal_nociception_inputs.size,
                       'report_sha256':hashlib.sha256((out/'report.json').read_bytes()).hexdigest()}
                assert row['inputs_released'] and row['thermal_channel_released']
                results.append(row)
                trials[condition] = (b.counts.copy(), np.array([r['position_mm'] for r in trace]))
                print(json.dumps(row), flush=True)
            np.testing.assert_array_equal(trials['heat'][0], trials['motor_blocked'][0])
            direct = b.resolve(b.circuit_ids('warmth')+b.circuit_ids('nociception_proxy'))
            downstream = np.ones(len(b.counts), dtype=bool)
            downstream[direct] = False
            changed = int(np.count_nonzero(trials['heat'][0][downstream] != trials['md_outputs_blocked'][0][downstream]))
            assert changed > 0, 'Thermal md input did not influence any downstream counts'
            difference = float(np.linalg.norm(trials['heat'][1]-trials['md_outputs_blocked'][1], axis=1).max())
            comparisons.append({'seed':seed, 'motor_blockade_preserves_exact_neural_counts':True,
                                'downstream_cells_with_changed_counts_on_md_output_blockade':changed,
                                'maximum_body_position_difference_on_md_output_blockade_mm':difference})
        report = {'version':__version__, 'success':True, 'dataset':graph.snapshot,
                  'neurons':len(graph.ids), 'duration_ms_per_trial':700, 'trials':results,
                  'comparisons':comparisons, 'stimulus':{'nominal_celsius':40,'warmth_rate_hz':300,
                                                       'md_rate_hz':100,'md_cells':len(md)},
                  'manifest_sha256':hashlib.sha256((pack/'manifest.json').read_bytes()).hexdigest(),
                  'source': 'https://doi.org/10.1101/2025.10.28.684868',
                  'limits':['40 C is a scenario gate based on a reported experimental condition, not a measured universal threshold.',
                            'Input rates are engineering choices, not fitted physiology or subjective pain.',
                            'Warmth-only comparison changes random draw allocation; matched md-output blockade is the causal control.',
                            'Body control remains engineered; 300 ms after release is not a recovery certification.']}
        write_json(args.output_dir/'report.json', report)
    finally:
        body.close()


if __name__ == '__main__':
    main()
