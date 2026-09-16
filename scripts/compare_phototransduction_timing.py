"""Compare unmodified upstream MATLAB cascade in Octave with the Python variant.

Both models receive a single microvillus's specified photon-arrival schedule.
This isolates cascade timing from photon allocation, membrane integration and
population scaling. Equal seed labels do not imply matched random draws across
Octave and NumPy. The audit does not certify physiological accuracy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from verify_phototransduction import HASHES, ROOT
import numpy as np
from scipy.io import loadmat
from nexus.brain.phototransduction import Phototransduction
from nexus.brain.photoreceptor import SOURCE_COMMIT

CONDITIONS = {'dark': [], 'single': [20], 'paired_20ms': [20, 40], 'paired_100ms': [20, 120]}
DURATION_MS = 300
AUDIT_HASHES = {**HASHES, 'NaCaPump.m': 'b69814e82750c96f4f9d1b3098bb341263b2a0b6cdf20449e10f7d683080a223'}


def matlab_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def summarize(channels, times_ms, first_photon_ms):
    """Describe the returned sampling grid; area is a rectangular sum, not charge."""
    channels = np.asarray(channels, dtype=float)
    if not np.isfinite(channels).all() or (channels < 0).any() or (channels > 27).any():
        raise ValueError('Invalid channel trace')
    first = np.flatnonzero(channels > 0)
    return {'responded': bool(len(first)), 'peak_open_channels': float(channels.max()),
            'channel_area_channel_ms': float(channels.sum()*(times_ms[1]-times_ms[0])),
            'first_response_sample_ms': float(times_ms[first[0]]) if len(first) else None,
            'sampled_latency_ms': (float(times_ms[first[0]]-first_photon_ms)
                                   if len(first) and first_photon_ms is not None else None)}


def aggregate(rows):
    result = {}
    for condition in CONDITIONS:
        result[condition] = {}
        for model in ('upstream_octave', 'python_continuous'):
            trials = [r for r in rows if r['condition'] == condition and r['model'] == model]
            group = {'trials': len(trials), 'responded': sum(r['responded'] for r in trials)}
            for field in ('peak_open_channels', 'channel_area_channel_ms', 'sampled_latency_ms'):
                values = [r[field] for r in trials if r[field] is not None]
                group[field] = {'mean': float(np.mean(values)), 'median': float(np.median(values)),
                                'q25': float(np.quantile(values, .25)), 'q75': float(np.quantile(values, .75))} if values else None
            result[condition][model] = group
    return result


def read_comparisons(output, seeds):
    """Compare saved traces on a common 1 ms grid; preserve native-grid metrics."""
    rows = []
    common_times = np.arange(1, DURATION_MS)
    for condition, arrivals in CONDITIONS.items():
        for seed in seeds:
            recorded = loadmat(output/f'{condition}-{seed}-octave.mat')
            expected = np.zeros(DURATION_MS)
            expected[np.asarray(arrivals, dtype=int)-1] = 1
            np.testing.assert_array_equal(recorded['LL'].ravel(), expected)
            assert int(recorded['seed'].item()) == seed
            source_channels = recorded['channels'].ravel()
            with np.load(output/f'{condition}-{seed}-python.npz', allow_pickle=False) as trial:
                python_channels = trial['open_channels'].copy()
                photons = np.zeros(DURATION_MS, dtype=np.int64)
                photons[arrivals] = 1
                np.testing.assert_array_equal(trial['absorbed_photons'], photons)
            assert source_channels.shape == (DURATION_MS,)
            assert python_channels.shape == (DURATION_MS*10,)
            first = arrivals[0] if arrivals else None
            for model, native, native_times, aligned in (
                ('upstream_octave', source_channels, np.arange(1, DURATION_MS+1), source_channels[:-1]),
                ('python_continuous', python_channels, np.arange(DURATION_MS*10)*.1, python_channels[10::10]),
            ):
                row = {'condition': condition, 'seed_label': seed, 'model': model,
                       **summarize(aligned, common_times, first),
                       'native_grid': summarize(native, native_times, first)}
                if condition == 'dark' and row['responded']:
                    raise RuntimeError('Unexpected channel opening in darkness')
                if model == 'upstream_octave':
                    row['returned_rows'] = int(recorded['output_rows'].item())
                    row['maximum_active_rhodopsin'] = float(recorded['yy'][:, 3].max())
                    row['rhodopsin_exceeds_supplied_photons'] = row['maximum_active_rhodopsin'] > len(arrivals)
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--octave', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=32)
    parser.add_argument('--reuse-recordings', action='store_true',
                        help='Re-analyze a completed audit with matching source hashes and trial count')
    args = parser.parse_args()
    if not 1 <= args.repeats <= 1000:
        parser.error('repeats must be between 1 and 1000')
    source = args.source_dir.resolve()
    for filename, expected in AUDIT_HASHES.items():
        if hashlib.sha256((source/filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Unexpected source revision: {filename}')
    octave = args.octave.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=args.reuse_recordings)
    output = args.output_dir.resolve()
    previous = None
    if args.reuse_recordings:
        previous = json.loads((output/'report.json').read_text())
        if (previous['source_sha256'] != AUDIT_HASHES or previous['repeats_per_condition_per_model'] != args.repeats
                or previous['duration_ms'] != DURATION_MS or previous['arrival_times_ms'] != CONDITIONS
                or previous['python_model'] != 'source-reaction-network-continuous-events-v1'
                or not previous['execution_success']):
            raise ValueError('Saved audit configuration does not match')
        version = previous['octave_version']
    else:
        version = subprocess.run([str(octave), '--no-gui', '--quiet', '--no-init-file', '--eval',
                                  "fprintf('%s', version);"], check=True, capture_output=True, text=True).stdout.strip()
    rows = []
    seeds = list(range(77100, 77100+args.repeats))
    start = time.perf_counter()
    for condition, arrivals in CONDITIONS.items():
        if args.reuse_recordings:
            continue
        driver = output / f'{condition}.m'
        # Script loads original parameters and calls original cascade without
        # rewriting event logic, recording every result independently.
        driver.write_text(f"""addpath({matlab_string(source)});
output_dir = {matlab_string(output)};
condition = {matlab_string(condition)};
arrivals = [{' '.join(str(x) for x in arrivals)}];
for seed = [{' '.join(str(x) for x in seeds)}]
  rng(seed, 'twister');
  Initial_PR_Goodaa1new;
  LL = zeros({DURATION_MS}, 1);
  LL(arrivals) = 1;
  tic;
  [tstep, yy] = RandomBumpModel_GillespieD_Continue_TRPLa({DURATION_MS*10}, yini_online, LL, para, 50, 0.2);
  elapsed = toc;
  if rows(yy) < {DURATION_MS}, error('Missing source output samples'); end
  channels = yy(1:{DURATION_MS}, 1);
  output_rows = rows(yy);
  filename = fullfile(output_dir, sprintf('%s-%d-octave.mat', condition, seed));
  save('-mat7-binary', filename, 'channels', 'yy', 'tstep', 'seed', 'condition', 'LL', 'elapsed', 'output_rows');
  fprintf('source %s seed %d done in %.3f s\\n', condition, seed, elapsed);
  fflush(stdout);
end
""")
        with (output/f'{condition}-octave.log').open('w') as log:
            subprocess.run([str(octave), '--no-gui', '--quiet', '--no-init-file', str(driver)],
                           stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=120*args.repeats, env={**os.environ, 'OCTAVE_HISTFILE': os.devnull})
        print(f'Original source completed {condition}: {len(seeds)} trials', flush=True)
        for seed in seeds:
            recorded = loadmat(output/f'{condition}-{seed}-octave.mat')
            source_channels = recorded['channels'].ravel()
            # MATLAB LL(k) is delivered at t=k ms, not (k-1) ms.
            # Our input frame at index k is delivered at t=k ms.
            photons = np.zeros(DURATION_MS, dtype=np.int64)
            photons[arrivals] = 1
            receptor = Phototransduction(microvilli=1, seed=seed)
            trial = receptor.advance(photons)
            np.savez_compressed(output/f'{condition}-{seed}-python.npz', **trial, absorbed_photons=photons)
            for model, channels, times in (
                ('upstream_octave', source_channels, np.arange(1, DURATION_MS+1)),
                ('python_continuous', trial['open_channels'], np.arange(DURATION_MS*10)*.1),
            ):
                row = {'condition': condition, 'seed_label': seed, 'model': model,
                       **summarize(channels, times, arrivals[0] if arrivals else None)}
                if condition == 'dark' and row['responded']:
                    raise RuntimeError('Unexpected channel opening in darkness')
                if model == 'upstream_octave':
                    row['returned_rows'] = int(recorded['output_rows'].item())
                rows.append(row)
        (output/'completed-trials.json').write_text(json.dumps(rows, indent=2)+'\n')
    rows = read_comparisons(output, seeds)
    anomalous_inputs = {(r['condition'], r['seed_label']) for r in rows
                       if r.get('rhodopsin_exceeds_supplied_photons')}
    report = {'format': 'nexus-phototransduction-timing-audit-2',
              'execution_success': True, 'source_commit': SOURCE_COMMIT,
              'source_sha256': AUDIT_HASHES, 'octave_version': version,
              'python_model': 'source-reaction-network-continuous-events-v1',
              'repeats_per_condition_per_model': args.repeats, 'duration_ms': DURATION_MS,
              'arrival_times_ms': CONDITIONS, 'wall_seconds': previous['wall_seconds'] if previous else time.perf_counter()-start,
              'comparison_grid_ms': [1, DURATION_MS-1, 1],
              'source_input_anomalies': [r for r in rows if r.get('rhodopsin_exceeds_supplied_photons')],
              'groups': aggregate(rows), 'trials': rows,
              'groups_excluding_observed_input_anomalies': aggregate([
                  r for r in rows if (r['condition'], r['seed_label']) not in anomalous_inputs]),
              'limits': ['Runs the unmodified cascade in GNU Octave, not proprietary MATLAB.',
                         'Tests one microvillus at a time; no full receptor, membrane, or synaptic pathway is compared.',
                         'Both are compared at 1..299 ms in 1 ms increments. Native-grid measurements are also retained.',
                         'Upstream uses ceil-based row filling; matching the sampling grid does not undo that output convention.',
                         'Sampled onset and area include these output-grid differences; they are not exact continuous-time latency or ionic charge.',
                         'Equal seed labels do not match the random draws across implementations.',
                         'The Rh maximum check can miss duplicate deliveries after Rh removal; the exclusion group is not certified duplicate-free.',
                         'No equivalence tolerance was defined; successful execution does not mean the two models match.',
                         'Neither implementation is validated against biological recordings by this audit.']}
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['groups'], indent=2), flush=True)


if __name__ == '__main__':
    main()
