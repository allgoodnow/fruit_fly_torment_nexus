"""Matched full-network sign assay and regression across four interventions."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / '.runtime/numba'))
import numpy as np
from nexus.brain.runtime import Brain, Connectome, weights_filename
from nexus.brain.protocol import Protocol
from nexus.brain.scenarios import scenario_protocol
from nexus.datasets.histamine import POLICY_ID
from nexus.datasets.male_cns import digest


def probe_pathway(old, new, changed, output, *, rule_id=None):
    pres = np.searchsorted(new.offsets, changed, side='right') - 1
    # Deterministic assay location: strongest total repaired input from one cell.
    pre = int(np.argmax(np.bincount(pres, weights=-new.weights[changed], minlength=len(new.ids))))
    targets = np.unique(new.posts[changed[pres == pre]])[:32]
    rows = []
    for seed in [73100, 73101, 73102]:
        trials = {}
        for condition, graph in [('original', old), ('repaired', new), ('blocked', new)]:
            brain = Brain(graph, seed=seed)
            trace_ids = graph.ids[targets]
            segments = [brain.advance(.05, trace_ids=trace_ids)['voltage_mv']]
            brain.stimulate([int(graph.ids[pre])], 100)
            if condition == 'blocked':
                brain.silence([int(graph.ids[pre])])
            segments.append(brain.advance(.2, trace_ids=trace_ids)['voltage_mv'])
            brain.release()
            segments.append(brain.advance(.1, trace_ids=trace_ids)['voltage_mv'])
            trace = np.concatenate(segments)
            np.savez_compressed(output / f'{seed}-{condition}.npz', voltage_mv=trace,
                                counts=brain.counts, trace_ids=trace_ids)
            row = {'seed': seed, 'condition': condition, 'photoreceptor_id': str(graph.ids[pre]),
                   'target_ids': [str(i) for i in trace_ids], 'presynaptic_spikes': int(brain.counts[pre]),
                   'minimum_target_voltage_during_input_mv': float(segments[1].min()),
                   'total_spikes': int(brain.counts.sum()), 'inputs_released': len(brain.inputs) == 0}
            rows.append(row)
            if rule_id is not None:
                row['rule_id'] = rule_id
            trials[condition] = (trace, brain.counts.copy(), brain.rng.bit_generator.state)
            print(json.dumps(row), flush=True)
        np.testing.assert_array_equal(trials['original'][0], trials['blocked'][0])
        np.testing.assert_array_equal(trials['original'][0][:500], trials['repaired'][0][:500])
        assert np.min(trials['repaired'][0][500:2500] - trials['original'][0][500:2500]) < -.1
        assert all(trials[c][2] == trials['original'][2] for c in trials)
    return rows


def main(*, extended=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pack = ROOT / 'data/brain-male-cns-v1.0-lif'
    evidence_name = 'receptor-sign-evidence-v1.json' if extended else 'histamine-evidence-v1.json'
    evidence = json.loads((ROOT / 'experiments' / evidence_name).read_text())
    if digest(pack / 'manifest.json') != evidence['manifest_sha256']:
        raise ValueError('Active pack differs from the installation audit')
    new = Connectome.load(pack, allow_experimental=True)
    expected_policy = POLICY_ID
    if extended:
        from nexus.datasets.receptor_signs import POLICY_ID as expected_policy
    if new.model['id'] != expected_policy:
        raise ValueError('Install the corresponding receptor policy first')
    previous_path = pack / evidence['rollback_manifest']
    if digest(previous_path) != evidence['original_manifest_sha256']:
        raise ValueError('Original manifest changed')
    previous = json.loads(previous_path.read_text())
    path = pack / weights_filename(previous)
    if digest(path) != evidence['original_weights_sha256']:
        raise ValueError('Original weights changed')
    old = replace(new, weights=np.load(path, mmap_mode='r'), model=previous['model'])
    changed = np.flatnonzero(new.weights != old.weights)
    expected_hash = (evidence['changed_edge_indices_sha256'] if extended
                     else evidence['rule']['edge_indices_sha256'])
    if (len(changed) != evidence['changed_weights']
            or hashlib.sha256(np.asarray(changed, dtype='<i8').tobytes()).hexdigest() != expected_hash):
        raise ValueError('Weight differences do not match the audited edge set')
    assert np.all(old.weights[changed] == 0) and np.all(new.weights[changed] < 0)
    if extended:
        import pandas as pd
        from nexus.datasets.male_cns_runtime import transmitter_signs
        from nexus.datasets.receptor_signs import compile_rules, edge_digest
        raw = ROOT / 'data/raw/male-cns-v1.0'
        for key in ['source_annotations', 'source_transmitters']:
            if digest(raw / evidence[key]['file']) != evidence[key]['sha256']:
                raise ValueError('Raw data changed since the receptor audit')
        neurons = pd.read_feather(raw / evidence['source_annotations']['file'])
        _, labels = transmitter_signs(new.ids, pd.read_feather(raw / evidence['source_transmitters']['file']))
        groups = compile_rules(new.ids, new.offsets, new.posts, neurons, labels)
        records = {r['id']: r for r in evidence['rules']}
        rows = []
        for rule, edges in groups:
            if not len(edges):
                continue
            assert edge_digest(edges) == records[rule['id']]['edge_indices_sha256']
            output = args.output_dir / rule['id']
            output.mkdir()
            rows.extend(probe_pathway(old, new, edges, output, rule_id=rule['id']))
    else:
        rows = probe_pathway(old, new, changed, args.output_dir)
    comparisons = []
    for seed in [73100, 73101, 73102]:
        for name in ['defensive', 'aversion', 'seizure', 'heat_overload']:
            trials = {}
            for condition, graph in [('original', old), ('repaired', new)]:
                brain = Brain(graph, seed=seed)
                plan = scenario_protocol(name, dataset=graph.snapshot, pain_circuit='nociception_proxy',
                                         celsius=100, baseline_ms=100, stimulus_ms=300, recovery_ms=200)
                protocol = Protocol(plan, brain)
                while not protocol.completed:
                    protocol.advance(brain, 100)
                assert not len(brain.inputs) and np.isfinite(brain.v).all() and np.isfinite(brain.g).all()
                trials[condition] = (brain.counts.copy(), brain.rng.bit_generator.state)
                np.savez_compressed(args.output_dir / f'{seed}-{name}-{condition}.npz', counts=brain.counts)
            assert trials['original'][1] == trials['repaired'][1]
            row = {'seed': seed, 'scenario': name,
                   'original_spikes': int(trials['original'][0].sum()),
                   'repaired_spikes': int(trials['repaired'][0].sum()),
                   'neurons_with_changed_counts': int(np.count_nonzero(trials['original'][0] != trials['repaired'][0])),
                   'identical_input_random_draws': True, 'finite_state_and_inputs_released': True}
            comparisons.append(row)
            print(json.dumps(row), flush=True)
    report = {'format': 'nexus-receptor-sign-results-1' if extended else 'nexus-histamine-results-1', 'success': True,
              'manifest_sha256': evidence['manifest_sha256'], 'neurons': len(new.ids),
              'edges': len(new.posts), 'changed_weights': len(changed), 'topology_unchanged': True,
              'photoreceptor_trials': rows, 'scenario_comparisons': comparisons,
              'limits': ['Direct 100 Hz photoreceptor stimulation is a sign assay, not a light-to-retina model.',
                         'Target hyperpolarization validates implemented inhibition, not measured physiology.',
                         'No graded retinal transmission, fitted receptor dynamics, spontaneous exploration, '
                         'or subjective experience is established.']}
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
