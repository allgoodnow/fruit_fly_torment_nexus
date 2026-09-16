"""Source-equation checks and offline photon-to-voltage experiments.

This verifies the reaction kernels, not equivalence of the new event scheduler
to MATLAB or to physiological recordings. No files or controls in the live app
are modified. See docs/phototransduction.md for the numerical differences.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

from verify_photoreceptor_membrane import ROOT, expression
import numpy as np
from nexus.brain.photoreceptor import SOURCE_COMMIT
from nexus.brain.phototransduction import (
    PARAMETERS, Phototransduction, calcium_update, ghk_currents, reaction_rates,
)

HASHES = {
    'Initial_PR_Goodaa1new.m': '0977e2a54feedea0b9c50130bdeae8c5a63dba06b7b66a9a44045ea88924f892',
    'RandomBumpModel_GillespieD_Continue_TRPLa.m': '4bdfda06813c992a24a2c2d6d9b150fb0ca7b6d526b745abc9bc8bbd9eafb987',
    'TRP_Rev_GHK_online.m': '72445ed44e095572dc14e3c2a56981f49696972b259ceaa311055b1d721547cf',
    'Gillespie_sim_b2p3_Skew_speed205.m': '39bc3d350f415be3ac13b955bc97c3622eb4dcd3f90a4ac74d1d6b3e51ca6e9b',
}


def references(directory):
    source = {}
    for name, sha in HASHES.items():
        data = (directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f'Unexpected phototransduction source revision: {name}')
        source[name] = re.sub(r'%[^\n]*', '', data.decode()).replace('...\n', '')
    parameters = np.zeros(27)
    for index, code in re.findall(r'para\((\d+)\)\s*=\s*([^;]+);', source['Initial_PR_Goodaa1new.m']):
        parameters[int(index)-1] = eval(expression(code), {'__builtins__': {}}, {'aa': 1})
    np.testing.assert_array_equal(PARAMETERS, parameters)
    cascade = source['RandomBumpModel_GillespieD_Continue_TRPLa.m']
    aliases = {name: parameters[int(index)-1] for name, index in
               re.findall(r'(\w+)\s*=\s*para\((\d+)\);', cascade)}
    feedback = eval(expression(re.search(r'fnstr\s*=\s*([^;]+);',
                    source['Gillespie_sim_b2p3_Skew_speed205.m'])[1]), {'__builtins__': {}}, {})
    helpers = [(name, expression(re.search(r'\b'+name+r'\s*=\s*([^;]+);', cascade)[1]))
               for name in ('fbp', 'fbn', 'Oc')]
    z_expressions = [(int(i)-1, int(j)-1, expression(code)) for i, j, code in
                     re.findall(r'z\((\d+),(\d+)\)\s*=\s*([^;]+);', cascade)]
    ca_code = expression(re.search(r'y\(2\)\s*=\s*([^;]+);', cascade)[1])
    ghk_text = source['TRP_Rev_GHK_online.m'].split('function ', 1)[1].split('\n', 1)[1]
    ghk_expressions = [(name, expression(code)) for name, code in
                       re.findall(r'\b(\w+)\s*=\s*([^;]+);', ghk_text)]

    def ghk(n, calcium, previous):
        scope = dict(Ntrp=n, Cai=calcium, TRP_Rev_pre=previous,
                     exp=math.exp, log=math.log, pi=math.pi)
        for name, code in ghk_expressions:
            scope[name] = eval(code, {'__builtins__': {}}, scope)
        return np.array([scope[k] for k in ('I_m', 'ICa', 'INa', 'IMg', 'IK', 'Rev')])

    def rates_and_calcium(state, previous):
        scope = dict(aliases, y=lambda i: state[int(i)-1], fnstr=feedback, G_t=50.)
        for name, code in helpers:
            scope[name] = eval(code, {'__builtins__': {}}, scope)
        z = np.zeros((8, 2))
        for i, j, code in z_expressions:
            z[i, j] = eval(code, {'__builtins__': {}}, scope)
        currents = ghk(state[0], state[1]/6.022/3/100, previous)
        scope['ICa'] = currents[1]
        ca = eval(ca_code, {'__builtins__': {}}, scope)
        return z, np.array([ca, currents[-1]])
    return ghk, rates_and_calcium


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    ghk, reference = references(args.source_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(26119)
    states = np.zeros((128, 8))
    states[:, 0] = rng.integers(0, 28, len(states))
    states[:, 1] = rng.uniform(.05, 2000, len(states))
    states[:, 3] = rng.integers(0, 5, len(states))
    states[:, 5] = rng.integers(0, 301, len(states))
    states[:, 6] = rng.integers(0, 701, len(states))
    # Partition a 50-molecule G pool into active/PLC-bound/available/refractory.
    pools = rng.multinomial(50, [.25]*4, size=len(states))
    states[:, 2], states[:, 4], states[:, 7] = pools[:, 0], pools[:, 1], pools[:, 2]
    reversal = rng.uniform(0, .02, len(states))
    expected_rates, expected_ca, expected_ghk = [], [], []
    max_error = 0.
    for state, previous in zip(states, reversal):
        expected_z, expected_c = reference(state, previous)
        expected_g = ghk(state[0], state[1]/6.022/3/100, previous)
        actual_z = reaction_rates(state)
        actual_c = np.array(calcium_update(state, previous))
        actual_g = np.array(ghk_currents(state[0], state[1]/6.022/3/100, previous))
        for actual, expected in ((actual_z, expected_z), (actual_c, expected_c), (actual_g, expected_g)):
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-11)
            max_error = max(max_error, float(np.abs(actual-expected).max()))
        expected_rates.append(expected_z)
        expected_ca.append(expected_c)
        expected_ghk.append(expected_g)
    np.savez_compressed(args.output_dir / 'reference-fixture.npz', source_commit=SOURCE_COMMIT,
                        states=states, previous_reversal_v=reversal, rates=expected_rates,
                        calcium_updates=expected_ca, ghk_currents=expected_ghk)
    # Warm compilation before benchmark timing. No scaled-down population gain.
    Phototransduction(microvilli=1).advance([0])
    rows = []
    for seed in (73100, 73101, 73102):
        for condition in ('dark', 'pulse', 'steady'):
            photons = np.zeros(300, dtype=np.int64)
            if condition == 'pulse':
                photons[50] = 3000
            elif condition == 'steady':
                photons[50:150] = 30  # Same 3000 absorbed photons, spread over 100 ms.
            receptor = Phototransduction(seed=seed)
            start = time.perf_counter()
            result = receptor.advance(photons)
            elapsed = time.perf_counter()-start
            assert receptor.membrane.step == receptor.time_ms*10
            assert receptor.photons == int(photons.sum())
            assert np.all(result['open_channels'] >= 0)
            assert np.all(result['open_channels'] <= receptor.microvilli*27)
            assert np.all(result['open_channels'][:500] == 0)
            assert np.all(receptor.states[:, 2]+receptor.states[:, 4]+receptor.states[:, 7] <= 50)
            if condition == 'dark':
                assert not result['open_channels'].any()
            else:
                assert result['open_channels'].max() > 0
            np.savez_compressed(args.output_dir / f'{seed}-{condition}.npz',
                                **result, absorbed_photons=photons, molecular_state=receptor.states)
            row = {'seed': seed, 'condition': condition, 'microvilli': receptor.microvilli,
                   'absorbed_photons': receptor.photons, 'reaction_events': receptor.events,
                   'peak_open_channels': int(result['open_channels'].max()),
                   'peak_voltage_mv': float(result['voltage_mv'].max()),
                   'end_voltage_mv': float(result['voltage_mv'][-1]),
                   'end_open_channels': int(result['open_channels'][-1]),
                   'simulated_ms': receptor.time_ms, 'wall_seconds': elapsed,
                   'membrane_clock_matches_cascade': True}
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = {'format': 'nexus-phototransduction-results-1', 'success': True,
              'source_commit': SOURCE_COMMIT, 'source_sha256': HASHES,
              'kernel_states_checked': len(states), 'maximum_kernel_absolute_error': max_error,
              'model': 'source-reaction-network-continuous-events-v1',
              'trials': rows,
              'limits': ['Kernels match source expressions; complete stochastic trajectories have not been validated against MATLAB or recordings.',
                         'Unrounded exponential timing omits the source latency offset, uniform floor and protected Rh* removal.',
                         'Fast calcium is refreshed once per event preparation, with a fixed -70 mV cascade, as documented.',
                         'Photons are supplied as absorbed counts at 1 ms boundaries. Channel samples are held at 0.1 ms left edges.',
                         'This tests one complete receptor at a time, not all 3377 receptors or the full brain.',
                         'Video-to-photon calibration and synaptic transmitter release are absent; the live app is unchanged.']}
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
