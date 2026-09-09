"""Characterize a negative-edge overlay; spike bins are not EEG or a seizure diagnosis."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / '.runtime/numba'))
import numpy as np
from nexus.brain.runtime import Brain, Connectome
from nexus.brain.anatomy import Anatomy
from nexus.brain.targets import SUGAR


def metrics(bins, regional, counts, excluded):
    downstream = counts.copy()
    downstream[excluded] = 0
    varying = regional[:, regional.std(axis=0) > 0]
    correlations = (np.corrcoef(varying.T)[np.triu_indices(varying.shape[1], 1)]
                    if varying.shape[1] > 1 else np.array([]))
    seconds = len(bins)*.001
    return {'spikes': int(counts.sum()), 'downstream_spikes': int(downstream.sum()),
            'recruited_neurons': int(np.count_nonzero(downstream)),
            'population_hz_per_neuron': float(counts.sum()/seconds/len(counts)),
            'peak_1ms_spikes': int(bins.max()),
            'bin_fano_factor': float(bins.var()/bins.mean()) if bins.mean() else None,
            'mean_octant_correlation': float(correlations.mean()) if len(correlations) else None,
            'nonconstant_octants': int(varying.shape[1]),
            'max_downstream_hz': float(downstream.max()/seconds)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/inhibition_probe')
    parser.add_argument('--seeds', type=int, nargs='+', default=[73100, 73101, 73102])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pack = ROOT / 'data/brain-v630'
    graph, anatomy = Connectome.load(pack), Anatomy(pack)
    group = ((anatomy.positions > np.median(anatomy.positions[anatomy.valid], axis=0)) * [1, 2, 4]).sum(axis=1)
    brain = Brain(graph)
    targets = brain.resolve(SUGAR)
    measured = anatomy.valid.copy()
    measured[targets] = False
    conditions = [('quiet', 1., 0), ('quiet_disinhibited', .25, 0),
                  ('sham', 1., 200), ('reduced_inhibition', .25, 200),
                  ('no_inhibition', 0., 200), ('high_input_control', 1., 1000)]
    rows = []
    report = {'format': 'nexus-inhibition-assay-1', 'overlay': 'negative-edge-gain-v1',
              'interpretation': 'Exploratory activity measurements; no validated seizure classification.',
              'sources': ['https://github.com/philshiu/Drosophila_brain_model',
                          'https://pubmed.ncbi.nlm.nih.gov/34278939/'],
              'brain_manifest_sha256': hashlib.sha256((pack/'manifest.json').read_bytes()).hexdigest(),
              'anatomy_sha256': anatomy.manifest['sha256'], 'bin_ms': 1,
              'windows_ms': {'baseline': [0, 100], 'intervention': [100, 400], 'recovery': [400, 700]},
              'regional_measurement': 'Eight coordinate-median octants; excludes direct input and unlocated cells.',
              'unlocated_cells': int((~anatomy.valid).sum()),
              'release_semantics': 'At 400 ms remove manual input and restore negative-edge gain; preserve dynamic state.',
              'trials': rows}
    brain.advance(.0001)  # Compile before timing.
    for seed in args.seeds:
        for name, gain, rate in conditions:
            brain.seed = seed
            brain.reset()
            bins, regional, summaries = [], [], {}
            window_counts = brain.counts.copy()
            started = time.perf_counter()
            for tick in range(700):
                if tick == 100:
                    brain.set_inhibition_gain(gain)
                    if rate:
                        brain.stimulate(SUGAR, rate)
                if tick == 400:
                    brain.release()
                before = brain.counts.copy()
                brain.advance(.001)
                delta = brain.counts-before
                bins.append(int(delta.sum()))
                regional.append(np.bincount(group[measured], weights=delta[measured], minlength=8).astype(int))
                if tick+1 in (100, 400, 700):
                    label, start = {100: ('baseline', 0), 400: ('intervention', 100), 700: ('recovery', 400)}[tick+1]
                    summaries[label] = metrics(np.array(bins[start:]), np.array(regional[start:]),
                                               brain.counts-window_counts, targets)
                    window_counts = brain.counts.copy()
            assert brain.step == 7000 and brain.inhibition_gain == 1 and not len(brain.inputs)
            assert summaries['baseline']['spikes'] == 0
            if rate == 0:
                assert brain.counts.sum() == 0
            np.savez_compressed(args.output / f'{name}-{seed}.npz',
                                population_bins=np.array(bins), octant_bins=np.array(regional),
                                final_counts=brain.counts)
            row = {'name': name, 'seed': seed, 'inhibition_gain': gain, 'input_hz': rate,
                   'wall_seconds': time.perf_counter()-started, 'windows': summaries,
                   'final_voltage_range_mv': [float(brain.v.min()), float(brain.v.max())],
                   'events': list(brain.events)}
            rows.append(row)
            (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
            print(json.dumps({'name': name, 'seed': seed, 'windows': summaries}), flush=True)
    report['success'] = True
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
