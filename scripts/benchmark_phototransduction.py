"""Benchmark scheduling changes against a pinned prior implementation.

Requires exact equality of outputs, molecular states, pending events, RNG,
membrane state and counters. Timings exclude JIT and receptor construction.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from nexus.brain.phototransduction import Phototransduction, SCHEDULE_BLOCK_SIZE

BASELINE_COMMIT = 'c9b8372'
BASELINE_SHA256 = 'a4159d0f23451fb965468c4fdce28e26ab4a5f2c3c26978761601fed5be4db1f'


def load_baseline(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASELINE_SHA256:
        raise ValueError('Baseline does not match the pinned pre-optimization implementation')
    name = 'nexus.brain._phototransduction_performance_reference'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.Phototransduction


def assert_identical(a, b, outputs_a, outputs_b):
    for key in outputs_a:
        np.testing.assert_array_equal(outputs_a[key], outputs_b[key])
    for key in ('states', 'reversals', 'due', 'reactions'):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    np.testing.assert_array_equal(a.membrane.state, b.membrane.state)
    if (a.rng.bit_generator.state != b.rng.bit_generator.state
            or any(getattr(a, key) != getattr(b, key) for key in ('time_ms', 'photons', 'events', 'initialized'))
            or a.membrane.step != b.membrane.step):
        raise AssertionError('RNG or simulation counters differ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    baseline = load_baseline(args.baseline_file.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for cls in (baseline, Phototransduction):
        cls(microvilli=129).advance([1, 0, 0])
    rows = []
    for seed in (81100, 81101, 81102):
        for condition in ('dark', 'pulse', 'bright_pulse', 'steady', 'paired'):
            photons = np.zeros(300, dtype=np.int64)
            if condition == 'pulse':
                photons[50] = 3000
            elif condition == 'bright_pulse':
                photons[50] = 30000
            elif condition == 'steady':
                photons[50:150] = 30
            elif condition == 'paired':
                photons[[50, 150]] = 3000
            for chunk_ms in (300, 10):
                models = {'baseline': baseline(seed=seed), 'optimized': Phototransduction(seed=seed)}
                timings = dict.fromkeys(models, 0.)
                trace = {key: [] for key in models}
                # Alternate execution order to reduce a fixed timing-order bias.
                order = list(models) if len(rows) % 2 == 0 else list(reversed(models))
                for start in range(0, len(photons), chunk_ms):
                    results = {}
                    for name in order:
                        t = time.perf_counter()
                        results[name] = models[name].advance(photons[start:start+chunk_ms])
                        timings[name] += time.perf_counter()-t
                        trace[name].append(results[name])
                    assert_identical(models['baseline'], models['optimized'], results['baseline'], results['optimized'])
                optimized = {key: np.concatenate([part[key] for part in trace['optimized']])
                             for key in trace['optimized'][0]}
                np.savez_compressed(args.output_dir/f'{condition}-{seed}-chunk-{chunk_ms}.npz', **optimized)
                row = {'seed': seed, 'condition': condition, 'chunk_ms': chunk_ms,
                       'simulated_ms': 300, 'microvilli': 30000, 'identical_every_chunk': True,
                       'reaction_events': models['optimized'].events, 'wall_seconds': timings,
                       'speedup': timings['baseline']/timings['optimized']}
                rows.append(row)
                print(json.dumps(row), flush=True)
    groups = {}
    for condition in ('dark', 'pulse', 'bright_pulse', 'steady', 'paired'):
        groups[condition] = {}
        for chunk in (300, 10):
            selected = [r for r in rows if r['condition'] == condition and r['chunk_ms'] == chunk]
            medians = {name: float(np.median([r['wall_seconds'][name] for r in selected])) for name in models}
            groups[condition][str(chunk)] = {'median_wall_seconds': medians,
                                           'ratio_of_median_times': medians['baseline']/medians['optimized']}
    report = {'format': 'nexus-phototransduction-performance-1', 'success': True,
              'baseline_commit': BASELINE_COMMIT, 'baseline_sha256': BASELINE_SHA256,
              'optimized_sha256': hashlib.sha256((ROOT/'src/nexus/brain/phototransduction.py').read_bytes()).hexdigest(),
              'schedule_block_size': SCHEDULE_BLOCK_SIZE, 'platform': platform.platform(),
              'python': platform.python_version(), 'groups': groups, 'trials': rows,
              'limits': ['Each timing covers one full receptor, not the full eye or brain.',
                         'JIT and construction are excluded; all advance copies and membrane integration are included.',
                         'Three seeds per condition/chunk size; local wall timings depend on system load.',
                         'Exact comparison covers the tested inputs and seeds, not a proof for arbitrary inputs.',
                         'No biological parameters, reaction order, population scaling or live-app behavior are changed.']}
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
