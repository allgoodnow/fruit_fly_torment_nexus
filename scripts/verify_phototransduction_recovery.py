"""Measure intrinsic recovery with a second photon and matched no-probe controls.

This is a model characterization, not a fit to electrophysiological recordings.
All channel areas are unscaled counts integrated over time, not ionic charge.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from nexus.brain.phototransduction import Phototransduction, STATE_NAMES
from nexus.brain.photoreceptor import SOURCE_COMMIT

GAPS_MS = (20, 50, 100, 200, 300, 500, 1000)
FIRST_PHOTON_MS = 20
PROBE_WINDOW_MS = 300


def areas(probe, no_probe, rested_probe, dt_ms=.1):
    """Subtract the first response's tail without clipping signed differences."""
    traces = [np.asarray(x, dtype=float) for x in (probe, no_probe, rested_probe)]
    if (any(x.ndim != 1 or x.size == 0 or not np.isfinite(x).all() or (x < 0).any()
            for x in traces) or any(x.shape != traces[0].shape for x in traces)
            or not np.isfinite(dt_ms) or dt_ms <= 0):
        raise ValueError('Expected matching finite nonnegative channel traces and positive dt')
    values = [float(x.sum()*dt_ms) for x in traces]
    return dict(zip(('probe_channel_ms', 'no_probe_channel_ms', 'rested_probe_channel_ms'), values),
                incremental_channel_ms=values[0]-values[1])


def summarize(rows, *, bootstrap_seed=79100, resamples=2000):
    """Jointly resample seed rows; ratios are ratios of means, never trial ratios."""
    effect = np.array([r['incremental_channel_ms'] for r in rows])
    fresh = np.array([r['rested_probe_channel_ms'] for r in rows])
    if len(rows) < 2 or not np.isfinite(effect).all() or not np.isfinite(fresh).all() or (fresh < 0).any():
        raise ValueError('At least two finite trials are required')
    indices = np.random.default_rng(bootstrap_seed).integers(len(rows), size=(resamples, len(rows)))
    effect_means, fresh_means = effect[indices].mean(axis=1), fresh[indices].mean(axis=1)
    # No division by tiny per-trial responses or removal of failed responses.
    valid = fresh_means > 0
    ratio = float(effect.mean()/fresh.mean()) if fresh.mean() > 0 else None
    return {
        'trials': len(rows), 'mean_incremental_channel_ms': float(effect.mean()),
        'mean_rested_probe_channel_ms': float(fresh.mean()),
        'mean_no_probe_channel_ms': float(np.mean([r['no_probe_channel_ms'] for r in rows])),
        'recovery_ratio_of_means': ratio,
        'incremental_mean_bootstrap_95pct': np.quantile(effect_means, [.025, .975]).tolist(),
        'recovery_ratio_bootstrap_95pct': (np.quantile(effect_means[valid]/fresh_means[valid], [.025, .975]).tolist()
                                        if valid.all() else None),
        'undefined_ratio_resamples': int((~valid).sum()),
        'mean_pre_probe_state': np.mean([r['pre_probe_state'] for r in rows], axis=0).tolist(),
    }


def trial(seed, gap_ms, *, microvilli=1, photons_per_pulse=1):
    if gap_ms < 1 or int(gap_ms) != gap_ms:
        raise ValueError('Photon gap must be a positive integer in ms')
    if (isinstance(photons_per_pulse, bool) or not isinstance(photons_per_pulse, (int, np.integer))
            or not 1 <= photons_per_pulse <= 1000000):
        raise ValueError('Choose 1..1000000 absorbed photons per pulse')
    conditioned = Phototransduction(microvilli=microvilli, seed=seed)
    prefix = np.zeros(FIRST_PHOTON_MS+gap_ms, dtype=np.int64)
    prefix[FIRST_PHOTON_MS] = photons_per_pulse
    first = conditioned.advance(prefix)
    before = conditioned.states.mean(axis=0)
    quiet = copy.deepcopy(conditioned)
    probe_input = np.zeros(PROBE_WINDOW_MS, dtype=np.int64)
    probe_input[0] = photons_per_pulse
    probe = conditioned.advance(probe_input)
    no_probe = quiet.advance(probe_input*0)
    rested = Phototransduction(microvilli=microvilli, seed=seed)
    rested.advance(prefix*0)  # Same dark aging and absolute time, no reset at probe.
    rested_probe = rested.advance(probe_input)
    if (conditioned.photons, quiet.photons, rested.photons) != (2*photons_per_pulse, photons_per_pulse, photons_per_pulse):
        raise RuntimeError('Unexpected photon delivery count')
    expected_time = FIRST_PHOTON_MS+gap_ms+PROBE_WINDOW_MS
    if any(r.time_ms != expected_time or r.membrane.step != expected_time*10
           for r in (conditioned, quiet, rested)):
        raise RuntimeError('Molecular and membrane clocks diverged')
    row = {'seed': seed, 'gap_ms': gap_ms, 'pre_probe_state': before.tolist(),
           'first_prefix_channel_ms': float(first['open_channels'].sum()*.1),
           **areas(probe['open_channels'], no_probe['open_channels'], rested_probe['open_channels'])}
    traces = {f'{name}_{key}': value for name, result in
              [('probe', probe), ('no_probe', no_probe), ('rested_probe', rested_probe)]
              for key, value in result.items()}
    return row, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=128)
    parser.add_argument('--microvilli', type=int, default=1)
    parser.add_argument('--photons-per-pulse', type=int, default=1)
    parser.add_argument('--gaps-ms', type=int, nargs='+', default=list(GAPS_MS))
    args = parser.parse_args()
    if not 2 <= args.repeats <= 4096:
        parser.error('repeats must be 2..4096')
    if not 1 <= args.microvilli <= 30000 or not 1 <= args.photons_per_pulse <= 1000000:
        parser.error('microvilli must be 1..30000 and photons-per-pulse 1..1000000')
    if any(g < 1 for g in args.gaps_ms) or len(set(args.gaps_ms)) != len(args.gaps_ms):
        parser.error('gaps must be distinct positive integers')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    groups, rows = {}, []
    for gap in args.gaps_ms:
        samples = []
        for seed in range(79100, 79100+args.repeats):
            row, traces = trial(seed, gap, microvilli=args.microvilli, photons_per_pulse=args.photons_per_pulse)
            samples.append(row)
            np.savez_compressed(args.output_dir/f'gap-{gap}-seed-{seed}.npz', **traces)
        groups[str(gap)] = summarize(samples)
        rows.extend(samples)
        print(json.dumps({'gap_ms': gap, **groups[str(gap)]}), flush=True)
    report = {'format': 'nexus-phototransduction-recovery-1', 'execution_success': True,
              'source_commit': SOURCE_COMMIT,
              'python_sha256': {name: hashlib.sha256((ROOT/'src/nexus/brain'/name).read_bytes()).hexdigest()
                                for name in ('phototransduction.py', 'photoreceptor.py')},
              'first_photon_ms': FIRST_PHOTON_MS, 'probe_window_ms': PROBE_WINDOW_MS,
              'gaps_ms': args.gaps_ms, 'microvilli': args.microvilli,
              'absorbed_photons_per_pulse': args.photons_per_pulse, 'channel_sample_ms': .1,
              'pre_probe_state_summary': 'mean across the explicit microvillus population',
              'state_names': STATE_NAMES, 'bootstrap_resamples': 2000, 'bootstrap_seed': 79100,
              'groups': groups, 'trials': rows,
              'limits': ['Photons are allocated across the explicit microvilli without a population multiplier; this is not whole-eye recovery.',
                         'Probe and no-probe branches share the complete pre-probe state and RNG; subsequent random paths can diverge.',
                         'Incremental area subtracts the no-probe tail; signed differences are retained and are not individual bump detections.',
                         'Rested controls age in darkness until the same absolute probe time.',
                         'Ratios use group means with all trials; values can exceed one or fall below zero.',
                         'Bootstrap intervals describe Monte Carlo seed uncertainty only, not biological uncertainty or simultaneous confidence bands.',
                         'No direct quantitative fit to biological recordings, new refractory timer, rate tuning or live-application change.']}
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
