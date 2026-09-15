"""Check the Python membrane against expressions read from pinned MATLAB source.

No MATLAB installation is needed. This is a scalar-expression translation of
the upstream RHS, followed by SciPy's adaptive DOP853 solver. It checks equation
and numerical agreement, not reproduction of a MATLAB run or biological data.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / '.runtime/numba'))

import numpy as np
from scipy.integrate import solve_ivp
from nexus.brain.photoreceptor import PhotoreceptorMembrane, derivative, initial_state, SOURCE_COMMIT

SOURCE_HASHES = {
    'wt_cc_model_pump.m': '5bd6a223f52a2fcd8c86b38c7ccc065fef12c6d111a9d5de0df768ab74cc3767',
    'NaCaPump_Body.m': 'fde20f79cf84fff8fc9bceab3ff7b5e599eae8250dcf4a2a382694a17355f495',
    'NaKPump.m': 'e2b0ddc3e93ca7bbdb1db50260f3b1d8157c2c9152404deacffa379150358622',
    'Vol_FeedbackCluster.m': '47eac23a5295081ac933feffcc9ef46809b1cafad97e3d69ce92163b033dbfa6',
}


def expression(text):
    text = text.replace('.^', '**').replace('^', '**').replace('.*', '*').replace('./', '/')
    tree = ast.parse(text.strip(), mode='eval')
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
               ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError('Unsupported source expression')
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name)
                or node.func.id not in ('exp', 'sqrt', 'y', 'param')):
            raise ValueError('Unsupported source call')
    return compile(tree, '<pinned MATLAB expression>', 'eval')


def source_reference(directory):
    texts = {}
    for name, expected in SOURCE_HASHES.items():
        data = (directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Unexpected source revision: {name}')
        text = re.sub(r'%[^\n]*', '', data.decode()).replace('...\n', '')
        texts[name] = text
    param_text = re.search(r'^param\s*=\s*\[([^]]+)\]', texts['Vol_FeedbackCluster.m'], re.M)[1]
    params = np.array([eval(expression(p), {'__builtins__': {}}, {}) for p in param_text.split()])
    functions = {}
    for name in ('NaCaPump_Body.m', 'NaKPump.m'):
        functions[name] = [(m[1], expression(m[2])) for m in
                           re.finditer(r'^\s*(\w+)\s*=\s*([^;]+);', texts[name], re.M)]
    body = texts['wt_cc_model_pump.m'].split('function dy=cur_clamp', 1)[1]
    # These are the nonzero-Shaker/nonzero-novel-K branches selected by BG1.
    names = ('Cm Vl gl VK gl2 Vlic glic Vsyn gsyn gKsm gKAm gnewm '
             'ah bh an bn amKA bmKA ahKA bhKA anew bnew Sm Sb gclm v05 A gcl '
             'gKs gKA gnew Inew Ishaker Ishab Ileak').split()
    assignments = []
    for name in names:
        matches = list(re.finditer(r'^\s*' + name + r'\s*=\s*([^;]+);', body, re.M))
        if len(matches) != 1:
            raise ValueError(f'Ambiguous MATLAB assignment: {name}')
        assignments.append((name, expression(matches[0][1])))
    rhs = body.split('dy=[', 1)[1].split('];', 1)[0]
    equations = [expression(line) for line in rhs.splitlines() if line.strip()]
    if len(equations) != 9:
        raise ValueError('Expected nine membrane equations')

    def reference(y, current):
        scope = {'exp': math.exp, 'sqrt': math.sqrt, 'pi': math.pi,
                 'y': lambda i: y[int(i)-1], 'param': lambda i: params[int(i)-1],
                 'Vol': float(y[0]), 'Cai': float(y[8]), 'Nai': float(y[6]), 'I': current}
        for statements in functions.values():
            for name, code in statements:
                scope[name] = eval(code, {'__builtins__': {}}, scope)
        scope['gKAmf'] = .087e-3
        for name, code in assignments:
            scope[name] = eval(code, {'__builtins__': {}}, scope)
        return np.array([eval(code, {'__builtins__': {}}, scope) for code in equations])
    return reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True,
                        help='BiophysicalPhotoreceptorModel directory at the pinned revision')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    reference = source_reference(args.source_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # Non-equilibrium states exercise each gate and the ion/pump terms.
    rng = np.random.default_rng(26117)
    states = np.tile(initial_state()[0], (64, 1))
    states[:, 0] = rng.uniform(-80, -10, len(states))
    states[:, 1:6] = rng.uniform(.01, .99, (len(states), 5))
    states[:, 6] = rng.uniform(7, 12, len(states))
    states[:, 7] = rng.uniform(130, 150, len(states))
    states[:, 8] = rng.uniform(.0001, .001, len(states))
    currents = rng.uniform(-.1, 3, len(states))
    expected = np.array([reference(y, i) for y, i in zip(states, currents)])
    actual = np.array([derivative(y, i) for y, i in zip(states, currents)])
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    np.savez_compressed(args.output_dir / 'reference-derivatives.npz',
                        states=states, currents_na=currents, derivatives=expected)
    results, pulse_references = [], []
    sample_indices = np.array([0, 9, 99, 499, 999, 1000, 1009, 1049, 1099,
                               1499, 1999, 2000, 2009, 2099, 2499, 2999])
    for amplitude in (0., .05, .2, 1., 3.):
        stimulus = np.zeros((3000, 1))
        stimulus[1000:2000] = amplitude
        model = PhotoreceptorMembrane()
        actual_v = model.advance(stimulus)[:, 0]
        y = initial_state()[0]
        reference_states = []
        # Split exactly at current discontinuities; no solver step spans them.
        for current in (0., amplitude, 0.):
            solution = solve_ivp(lambda t, state: reference(state, current), (0, 100), y,
                                 method='DOP853', rtol=1e-10, atol=1e-12,
                                 max_step=.5, t_eval=np.arange(1, 1001)*.1)
            assert solution.success
            y = solution.y[:, -1]
            reference_states.append(solution.y.T)
        expected_states = np.concatenate(reference_states)
        pulse_references.append(expected_states[sample_indices])
        error = float(np.max(np.abs(actual_v-expected_states[:, 0])))
        assert error < .002
        np.testing.assert_allclose(model.state[0], expected_states[-1], rtol=1e-5, atol=2e-5)
        np.savez_compressed(args.output_dir / f'pulse-{amplitude:g}-na.npz',
                            current_na=stimulus[:, 0], voltage_mv=actual_v,
                            reference_states=expected_states)
        result = {'current_na': amplitude, 'max_voltage_error_mv': error,
                  'minimum_mv': float(actual_v.min()), 'maximum_mv': float(actual_v.max())}
        results.append(result)
        print(json.dumps(result), flush=True)
    np.savez_compressed(args.output_dir / 'reference-fixture.npz',
                        source_commit=SOURCE_COMMIT, states=states,
                        currents_na=currents, derivatives=expected,
                        pulse_amplitudes_na=np.array([r['current_na'] for r in results]),
                        pulse_sample_indices=sample_indices,
                        pulse_reference_states=np.array(pulse_references))
    # Warmed CPU cost for the number of R1–R6 cells in the current eye registry.
    batch = PhotoreceptorMembrane(3377)
    start = time.perf_counter()
    batch.advance(np.full((100, 3377), .2))
    elapsed = time.perf_counter()-start
    report = {'format': 'nexus-photoreceptor-membrane-results-1', 'success': True,
              'source_commit': SOURCE_COMMIT, 'source_sha256': SOURCE_HASHES,
              'parameter_set': 'BG1', 'integration_dt_ms': .1,
              'reference': 'Pinned MATLAB scalar expressions evaluated in Python; adaptive SciPy DOP853',
              'derivative_states_checked': len(states),
              'maximum_absolute_derivative_difference': float(np.abs(actual-expected).max()),
              'current_pulse_trials': results,
              'cpu_benchmark': {'cells': 3377, 'simulated_ms': 10, 'wall_seconds': elapsed,
                                'includes_compilation': False},
              'limits': ['Membrane subsystem only; no photon-absorption or quantum-bump model.',
                         'No RGB-to-current calibration or voltage-to-histamine transfer is supplied.',
                         'Numerical agreement is not validation against biological recordings or a MATLAB execution.',
                         'Not enabled in the application; does not change brain state, video processing or movement.']}
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
