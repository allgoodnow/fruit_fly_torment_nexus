"""Verify continuous TRP current feedback against pinned source expressions.

The input experiments prescribe channel counts, not photons, images, or
biological recordings. Continuous ODE coupling replaces the source's offline
voltage iterations. Reference trajectories use source equations with DOP853.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from verify_photoreceptor_membrane import ROOT, SOURCE_HASHES, expression, source_reference
import numpy as np
from scipy.integrate import solve_ivp
from nexus.brain.photoreceptor import (
    PhotoreceptorMembrane, SOURCE_COMMIT, channel_derivative, initial_state,
    trp_current_na,
)


def source_channel_current(directory):
    data = (directory / 'Vol_FeedbackCluster.m').read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_HASHES['Vol_FeedbackCluster.m']:
        raise ValueError('Unexpected voltage-feedback source revision')
    text = re.sub(r'%[^\n]*', '', data.decode())
    # Both source branches initialize the same fixed TRP reversal potential.
    reversal = re.findall(r'TRPrev\s*=\s*(.*?)\*ones\(', text)
    if len(reversal) != 2 or reversal[0] != reversal[1]:
        raise ValueError('Ambiguous source reversal potential')
    statements = [('TRPrev', expression(reversal[0]))]
    for name in ('df', 'drive', 'Iextra'):
        code = re.search(r'^\s*' + name + r'\s*=\s*([^;]+);', text, re.M)[1]
        statements.append((name, expression(code)))
    # Source passes LICsum*0.001 from pA to the nA membrane input.
    if 'wt_cc_model_pump(LICsum*0.001,param,samprate)' not in text:
        raise ValueError('Unexpected source current units')

    def current(channels, voltage):
        scope = {'max': max, 'BumpS': channels, 'Vol': voltage}
        for name, code in statements:
            scope[name] = eval(code, {'__builtins__': {}}, scope)
        return scope['Iextra'] * .001
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    membrane_rhs = source_reference(args.source_dir)
    channel_current = source_channel_current(args.source_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(26118)
    states = np.tile(initial_state()[0], (64, 1))
    states[:, 0] = rng.uniform(-80, 25, len(states))
    states[:, 1:6] = rng.uniform(.01, .99, (len(states), 5))
    states[:, 6] = rng.uniform(7, 12, len(states))
    states[:, 7] = rng.uniform(130, 150, len(states))
    states[:, 8] = rng.uniform(.0001, .001, len(states))
    counts = rng.integers(0, 6001, len(states))
    currents = np.array([channel_current(n, y[0]) for n, y in zip(counts, states)])
    derivatives = np.array([membrane_rhs(y, i) for y, i in zip(states, currents)])
    np.testing.assert_allclose([trp_current_na(n, y[0]) for n, y in zip(counts, states)],
                               currents, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose([channel_derivative(y, n) for y, n in zip(states, counts)],
                               derivatives, rtol=1e-11, atol=1e-11)
    amplitudes = np.array([0, 27, 270, 2700, 5400])
    indices = np.array([0, 9, 99, 499, 999, 1000, 1009, 1049, 1099,
                        1499, 1999, 2000, 2009, 2099, 2499, 2999])
    rows, pulse_references = [], []
    for amplitude in amplitudes:
        opening = np.zeros((3000, 1), dtype=np.int64)
        opening[1000:2000] = amplitude
        model = PhotoreceptorMembrane()
        actual = model.advance_channels(opening)[:, 0]
        y = initial_state()[0]
        reference_states = []
        for channels in (0, int(amplitude), 0):
            solve = solve_ivp(lambda t, state: membrane_rhs(state, channel_current(channels, state[0])),
                              (0, 100), y, method='DOP853', rtol=1e-10, atol=1e-12,
                              max_step=.5, t_eval=np.arange(1, 1001)*.1)
            assert solve.success
            y = solve.y[:, -1]
            reference_states.append(solve.y.T)
        reference = np.concatenate(reference_states)
        pulse_references.append(reference[indices])
        error = float(np.max(np.abs(actual-reference[:, 0])))
        assert error < .002
        np.testing.assert_allclose(model.state[0], reference[-1], rtol=1e-5, atol=2e-5)
        np.savez_compressed(args.output_dir / f'pulse-{amplitude}-channels.npz',
                            open_channels=opening[:, 0], voltage_mv=actual, reference_states=reference)
        # Same prescribed channel train, but freeze driving force at -70 mV.
        # This diagnostic isolates the implemented voltage feedback mechanism.
        fixed = PhotoreceptorMembrane()
        fixed_v = fixed.advance(opening * channel_current(1, -70.))[:, 0]
        if amplitude > 0:
            assert actual[1999] < fixed_v[1999]
            assert channel_current(amplitude, actual[1999]) < channel_current(amplitude, -70.)
        row = {'open_channels_during_pulse': int(amplitude),
               'max_voltage_error_mv': error, 'pulse_end_voltage_mv': float(actual[1999]),
               'fixed_driving_force_pulse_end_voltage_mv': float(fixed_v[1999]),
               'pulse_end_current_na': float(channel_current(amplitude, actual[1999])),
               'current_at_minus_70_mv_na': float(channel_current(amplitude, -70.)),
               'recovery_end_voltage_mv': float(actual[-1])}
        rows.append(row)
        print(json.dumps(row), flush=True)
    np.savez_compressed(args.output_dir / 'reference-fixture.npz', source_commit=SOURCE_COMMIT,
                        states=states, open_channels=counts, currents_na=currents, derivatives=derivatives,
                        pulse_amplitudes=amplitudes, sample_indices=indices,
                        pulse_reference_states=np.array(pulse_references))
    batch = PhotoreceptorMembrane(3377)
    stimulus = np.full((100, 3377), 270, dtype=np.int64)
    start = time.perf_counter()
    batch.advance_channels(stimulus)
    elapsed = time.perf_counter()-start
    report = {'format': 'nexus-photoreceptor-channel-results-1', 'success': True,
              'source_commit': SOURCE_COMMIT, 'source_sha256': SOURCE_HASHES,
              'parameter_set': 'BG1', 'dt_ms': .1, 'conductance_ps_per_channel': 8,
              'trp_reversal_mv': 20, 'derivative_states_checked': len(states),
              'current_and_derivative_source_checks': True,
              'coupling': 'TRP current recomputed at every RK4 substage',
              'trials': rows,
              'cpu_benchmark': {'cells': 3377, 'simulated_ms': 10, 'wall_seconds': elapsed,
                                'includes_compilation': False},
              'limits': ['Channel counts are prescribed experiments, not simulated photon absorption.',
                         'Uses the fixed-reversal inward-current rule of Vol_FeedbackCluster.m, not its GHK microvillus model.',
                         'Continuous coupling replaces the source offline iteration and is checked against source equations with DOP853, not a MATLAB run.',
                         'No image calibration, biochemical cascade, synaptic release, or live brain integration is included.',
                         'The CPU benchmark covers only this subsystem. No real-time whole-fly claim is made.']}
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
