"""Isolate upstream scheduler rules using guarded copies, never the live model.

SPDX-License-Identifier: GPL-3.0-only
Source attribution: licenses/photoreceptor/NOTICE.md.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess

from check_phototransduction_photon_delivery import FUNCTION, instrument, arrivals_from_log
from compare_phototransduction_timing import AUDIT_HASHES, matlab_string, summarize
import numpy as np
from scipy.io import loadmat
from nexus.brain.photoreceptor import SOURCE_COMMIT

RULES = {
    'unrounded': ('t_update = round(tt*sam)/sam;', 't_update = tt;'),
    'no_offset': ('abs(la+(norm_1+norm_2))', 'abs(norm_1+norm_2)'),
    'no_uniform_floor': ('r2=0.001+(0.999)*rand;', 'r2=max(realmin,rand);'),
    'no_rh_protection': ('if nn==4 && Rh_Up==1', 'if false'),
}
VARIANTS = {'guarded_source': (), **{key: (key,) for key in RULES},
            'combined': tuple(RULES)}
RULES.update({
    'round_0p01ms': ('t_update = round(tt*sam)/sam;', 't_update = round(tt*100)/100;'),
    'round_0p001ms': ('t_update = round(tt*sam)/sam;', 't_update = round(tt*1000)/1000;'),
    # Source computes this Rh-dependent rate before delivering photons. Its
    # Rh deactivation rate is computed afterward, creating asymmetric rates.
    'fresh_input_rate': ('z(4,1)=0;', 'z(3,1)=kap_G*y(8)*y(4);\n    z(4,1)=0;'),
})
VARIANTS.update({key: (key,) for key in ('round_0p01ms', 'round_0p001ms')})
VARIANTS.update({
    'no_rh_protection_fresh_input': ('no_rh_protection', 'fresh_input_rate'),
    'combined_fresh_input': (*VARIANTS['combined'], 'fresh_input_rate'),
})


def replace_once(code, old, new):
    if code.count(old) != 1:
        raise ValueError(f'Missing or ambiguous source statement: {old}')
    return code.replace(old, new)


def variant_source(original, rules):
    code = instrument(original, guard=True)
    code = replace_once(code, '[tstep, yy] =', '[tstep, yy, audit_events] =')
    code = replace_once(code, 'starh = 0;', 'audit_events = zeros(0,2);\nstarh = 0;')
    # Record actual post-reaction state at its source execution time. This
    # observes the existing before-wait rule without changing RNG or states.
    code = replace_once(code, 'r2=0.001+(0.999)*rand;',
                        'audit_events(end+1,:) = [ii,y(1)];\n    r2=0.001+(0.999)*rand;')
    for rule in rules:
        code = replace_once(code, *RULES[rule])
    return code


def event_metrics(events, duration_ms=300):
    """Integrate source post-reaction states; repeated timestamps have zero area."""
    events = np.asarray(events, dtype=float)
    if (events.ndim != 2 or events.shape[1] != 2 or not len(events)
            or not np.isfinite(events).all() or (np.diff(events[:, 0]) < 0).any()
            or (events[:, 0] < 0).any() or (events[:, 0] > duration_ms).any()
            or (events[:, 1] < 0).any() or (events[:, 1] > 27).any()):
        raise ValueError('Invalid source event trace')
    intervals = np.diff(np.r_[events[:, 0], duration_ms])
    indices = np.searchsorted(events[:, 0], np.arange(1, duration_ms), side='right')-1
    sampled = np.zeros(duration_ms-1)
    valid = indices >= 0
    sampled[valid] = events[indices[valid], 1]
    return {'event_count': len(events),
            'zero_time_transitions': int(np.count_nonzero(np.diff(events[:, 0]) == 0)),
            'event_integrated_channel_ms': float(events[:, 1] @ intervals),
            'causal_1ms_grid': summarize(sampled, np.arange(1, duration_ms), 20)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--octave', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=32)
    parser.add_argument('--variants', nargs='+', choices=tuple(VARIANTS), default=list(VARIANTS))
    args = parser.parse_args()
    if not 1 <= args.repeats <= 1000:
        parser.error('repeats must be 1..1000')
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    for name, expected in AUDIT_HASHES.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Unexpected source revision: {name}')
    output.mkdir(parents=True, exist_ok=False)
    octave = str(args.octave.resolve())
    env = {**os.environ, 'OCTAVE_HISTFILE': os.devnull}
    version = subprocess.run([octave, '--no-gui', '--quiet', '--no-init-file', '--eval',
                              "fprintf('%s', version);"], env=env, text=True,
                             capture_output=True, check=True).stdout.strip()
    original = (source/f'{FUNCTION}.m').read_text()
    seeds = list(range(77100, 77100+args.repeats))

    def run_variant(item):
        name, rules = item
        folder = output/name
        folder.mkdir()
        code = variant_source(original, rules)
        (folder/f'{FUNCTION}.m').write_text(code)
        driver = folder/'driver.m'
        driver.write_text(f"""addpath({matlab_string(source)});
addpath({matlab_string(folder)}, '-begin');
for seed = [{' '.join(map(str, seeds))}]
  rng(seed, 'twister'); Initial_PR_Goodaa1new;
  LL = zeros(300,1); LL(20) = 1;
  fprintf('TRIAL %d\\n', seed);
  [tstep, yy, audit_events] = {FUNCTION}(3000, yini_online, LL, para, 50, .2);
  save('-mat7-binary', fullfile({matlab_string(folder)}, sprintf('%d.mat',seed)), 'tstep','yy','audit_events');
end
""")
        result = subprocess.run([octave, '--no-gui', '--quiet', '--no-init-file', str(driver)],
                                env=env, text=True, capture_output=True, timeout=120*len(seeds))
        (folder/'octave.log').write_text(result.stdout+result.stderr)
        result.check_returncode()
        logs = result.stdout.split('TRIAL ')[1:]
        if len(logs) != len(seeds):
            raise RuntimeError('Missing input-delivery logs')
        rows = []
        for seed, log in zip(seeds, logs):
            if int(log.splitlines()[0]) != seed:
                raise RuntimeError('Mismatched trial log')
            deliveries = arrivals_from_log(log)
            if [(r['time_ms'], r['count']) for r in deliveries] != [(20., 1.)]:
                raise RuntimeError('Photon was not delivered exactly once')
            data = loadmat(folder/f'{seed}.mat')
            row = {'seed': seed, 'delivered_photons': 1,
                   'source_1ms_grid': summarize(data['yy'][:299, 0], np.arange(1, 300), 20),
                   **event_metrics(data['audit_events'])}
            rows.append(row)
        print(f'{name}: {len(rows)} trials complete', flush=True)
        return name, {'changed_rules': list(rules), 'generated_source_sha256': hashlib.sha256(code.encode()).hexdigest(),
                      'trials': rows}

    with ThreadPoolExecutor(max_workers=2) as pool:
        variants = dict(pool.map(run_variant, [(name, VARIANTS[name]) for name in dict.fromkeys(args.variants)]))
    groups = {}
    for name, data in variants.items():
        rows = data['trials']
        groups[name] = {
            'trials': len(rows),
            'responded': sum(r['causal_1ms_grid']['responded'] for r in rows),
            'mean_source_grid_channel_ms': float(np.mean([r['source_1ms_grid']['channel_area_channel_ms'] for r in rows])),
            'mean_causal_grid_channel_ms': float(np.mean([r['causal_1ms_grid']['channel_area_channel_ms'] for r in rows])),
            'mean_event_integrated_channel_ms': float(np.mean([r['event_integrated_channel_ms'] for r in rows])),
            'zero_time_transitions': sum(r['zero_time_transitions'] for r in rows),
            'event_count': sum(r['event_count'] for r in rows),
        }
    report = {'format': 'nexus-phototransduction-timing-ablation-1', 'execution_success': True,
              'source_commit': SOURCE_COMMIT, 'source_sha256': AUDIT_HASHES, 'octave_version': version,
              'duration_ms': 300, 'photon_arrivals_ms': [20], 'groups': groups, 'variants': variants,
              'limits': ['Every variant guards duplicate input and verifies exactly one delivered photon per trial.',
                         'Reaction-before-wait and source output filling are retained; fresh_input variants refresh the Rh-dependent G activation rate after input.',
                         'Event integration uses recorded source post-reaction states; it is not ionic charge.',
                         'Same initial seeds do not guarantee matched reaction paths after a rule changes.',
                         'Only single-photon responses are tested; no physiological validation or live-app change.']}
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(groups, indent=2), flush=True)


if __name__ == '__main__':
    main()
