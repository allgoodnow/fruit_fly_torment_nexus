"""Reproduce and isolate duplicate photon delivery in the pinned upstream loop.

Runs original, logging-only and delivery-guarded copies in separate Octave
processes. Copies are written only to the requested results folder. The source
checkout and application's simulator are never modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from compare_phototransduction_timing import AUDIT_HASHES, matlab_string
import numpy as np
from scipy.io import loadmat
from nexus.brain.photoreceptor import SOURCE_COMMIT

FUNCTION = 'RandomBumpModel_GillespieD_Continue_TRPLa'
CASES = [('single', [20], 77121), ('single_control', [20], 77100),
         ('paired_20ms', [20, 40], 77121), ('paired_100ms', [20, 120], 77121)]


def instrument(source, *, guard):
    edits = [('y(4) = LL(rh_in)+y(4);',
              "y(4) = LL(rh_in)+y(4);\n        fprintf('PHOTON_ARRIVAL %.17g %.17g %.17g %.17g\\n', ii, LL(rh_in), y4_pre, y(4));")]
    if guard:
        edits += [('starh = 0;', 'delivered_photons = false(length(LL),1);\nstarh = 0;'),
                  ('if rem(ii,1)==0 && t_rho(T_rhin)==0',
                   'if rem(ii,1)==0 && ~delivered_photons(ii)'),
                  ('y(4) = LL(rh_in)+y(4);',
                   'y(4) = LL(rh_in)+y(4);\n        delivered_photons(rh_in) = true;')]
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError('Expected source statement was missing or ambiguous')
        source = source.replace(old, new)
    return source


def arrivals_from_log(text):
    rows = []
    for line in text.splitlines():
        if line.startswith('PHOTON_ARRIVAL '):
            time, count, before, after = map(float, line.split()[1:])
            if count > 0:
                rows.append({'time_ms': time, 'count': count, 'rhodopsin_before': before,
                             'rhodopsin_after': after})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--octave', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    for name, sha in AUDIT_HASHES.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != sha:
            raise ValueError(f'Unexpected source revision: {name}')
    output.mkdir(parents=True, exist_ok=False)
    original = (source/f'{FUNCTION}.m').read_text()
    versions = {'original': original, 'logging_only': instrument(original, guard=False),
                'delivery_guard': instrument(original, guard=True)}
    records = {}
    for mode, code in versions.items():
        folder = output/mode
        folder.mkdir()
        (folder/f'{FUNCTION}.m').write_text(code)
        for condition, arrivals, seed in CASES:
            prefix = folder/f'{condition}-{seed}'
            driver = prefix.with_suffix('.m')
            driver.write_text(f"""addpath({matlab_string(source)});
addpath({matlab_string(folder)}, '-begin');
rng({seed}, 'twister');
Initial_PR_Goodaa1new;
LL = zeros(300,1); LL([{' '.join(map(str, arrivals))}]) = 1;
[tstep, yy] = {FUNCTION}(3000, yini_online, LL, para, 50, .2);
save('-mat7-binary', {matlab_string(prefix.with_suffix('.mat'))}, 'yy', 'tstep');
""")
            result = subprocess.run([str(args.octave.resolve()), '--no-gui', '--quiet', '--no-init-file', str(driver)],
                                    text=True, capture_output=True, timeout=120,
                                    env={**os.environ, 'OCTAVE_HISTFILE': os.devnull})
            prefix.with_suffix('.log').write_text(result.stdout+result.stderr)
            result.check_returncode()
            records[mode, condition] = (loadmat(prefix.with_suffix('.mat')), arrivals_from_log(result.stdout))
            print(f'{mode} {condition}: complete', flush=True)
    comparisons = []
    for condition, arrivals, seed in CASES:
        raw, logged, guarded = [records[mode, condition] for mode in versions]
        for field in ('yy', 'tstep'):
            np.testing.assert_array_equal(raw[0][field], logged[0][field])
        expected = [(float(t), 1.) for t in arrivals]
        actual_guarded = [(r['time_ms'], r['count']) for r in guarded[1]]
        assert actual_guarded == expected
        assert guarded[0]['yy'][:, 3].max() <= len(arrivals)
        comparison = {'condition': condition, 'seed': seed, 'supplied_arrivals_ms': arrivals,
                      'logging_preserves_complete_source_output': True,
                      'observed_original_deliveries': logged[1],
                      'observed_guarded_deliveries': guarded[1],
                      'guard_delivers_each_supplied_photon_once': True,
                      'original_maximum_active_rhodopsin': float(raw[0]['yy'][:, 3].max()),
                      'guarded_maximum_active_rhodopsin': float(guarded[0]['yy'][:, 3].max()),
                      'guard_changes_channel_trace': not np.array_equal(raw[0]['yy'][:, 0], guarded[0]['yy'][:, 0])}
        comparisons.append(comparison)
    assert len(comparisons[0]['observed_original_deliveries']) == 2
    report = {'format': 'nexus-phototransduction-source-input-audit-1', 'success': True,
              'source_commit': SOURCE_COMMIT, 'source_sha256': AUDIT_HASHES,
              'comparisons': comparisons,
              'limits': ['Logging-only output is checked against original full state and time arrays.',
                         'The guarded source copy fixes observed repeated input handling only; it is not a calibrated replacement model.',
                         'No change is made to the source checkout, Python cascade or live application.',
                         'Only four specified schedules/seeds are checked; this is a reproduction, not an exhaustive upstream audit.']}
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
