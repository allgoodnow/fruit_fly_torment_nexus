"""Validate and time the optional CUDA membrane stage, including host transfers.

This is neither the photon-reaction cascade nor a whole-fly GPU benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import numba
from numba import cuda
from nexus.brain.photoreceptor import PhotoreceptorMembrane, DT_MS, SOURCE_COMMIT
from nexus.brain.photoreceptor_cuda import CudaPhotoreceptorMembrane


def inputs(ticks, cells, condition):
    channels = np.zeros((ticks, cells), dtype=np.int64)
    amplitudes = np.resize([0, 27, 270, 2700, 5400], cells)
    if condition == 'pulse':
        channels[ticks//3:2*ticks//3] = amplitudes
    elif condition == 'steady':
        channels[:] = amplitudes
    elif condition != 'dark':
        raise ValueError('Unknown channel experiment')
    return channels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if cuda.config.ENABLE_CUDASIM or not cuda.is_available():
        raise RuntimeError('Benchmark requires a physical CUDA GPU, not the CUDA simulator')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = cuda.get_current_device()
    # Warm both compilers and synchronize before recording operation times.
    for cls in (PhotoreceptorMembrane, CudaPhotoreceptorMembrane):
        cls(129).advance_channels(np.zeros((2, 129), dtype=np.int64))
    rows = []
    for cells in (1, 129, 3377):
        for condition in ('dark', 'pulse', 'steady'):
            for ticks, chunk_ticks in ((100, 100), (1000, 1000), (1000, 100)):
                channels = inputs(ticks, cells, condition)
                for repeat in range(3):
                    models = {'cpu': PhotoreceptorMembrane(cells), 'gpu': CudaPhotoreceptorMembrane(cells)}
                    elapsed = dict.fromkeys(models, 0.)
                    errors = []
                    for start in range(0, ticks, chunk_ticks):
                        results = {}
                        order = list(models) if repeat % 2 == 0 else list(reversed(models))
                        for name in order:
                            begin = time.perf_counter()
                            results[name] = models[name].advance_channels(channels[start:start+chunk_ticks])
                            elapsed[name] += time.perf_counter()-begin
                        error = float(np.abs(results['cpu']-results['gpu']).max())
                        np.testing.assert_allclose(results['gpu'], results['cpu'], rtol=0, atol=1e-7)
                        np.testing.assert_allclose(models['gpu'].state, models['cpu'].state, rtol=1e-9, atol=1e-10)
                        if models['gpu'].step != models['cpu'].step:
                            raise AssertionError('CPU and GPU clocks differ')
                        errors.append(error)
                    row = {'cells': cells, 'condition': condition, 'simulated_ms': ticks*DT_MS,
                           'chunk_ms': chunk_ticks*DT_MS, 'repeat': repeat,
                           'wall_seconds': elapsed, 'speedup': elapsed['cpu']/elapsed['gpu'],
                           'maximum_voltage_error_mv': max(errors),
                           'maximum_final_state_absolute_error': float(np.abs(models['cpu'].state-models['gpu'].state).max())}
                    rows.append(row)
                print(f'{cells} cells, {condition}, {ticks*DT_MS:g} ms in {chunk_ticks*DT_MS:g} ms chunks: passed', flush=True)
    groups = []
    for row in rows[::3]:
        selected = [r for r in rows if all(r[k] == row[k] for k in ('cells', 'condition', 'simulated_ms', 'chunk_ms'))]
        medians = {name: float(np.median([r['wall_seconds'][name] for r in selected])) for name in ('cpu', 'gpu')}
        groups.append({**{k: row[k] for k in ('cells', 'condition', 'simulated_ms', 'chunk_ms')},
                       'median_wall_seconds': medians, 'speedup': medians['cpu']/medians['gpu']})
    report = {'format': 'nexus-photoreceptor-cuda-1', 'success': True,
              'source_commit': SOURCE_COMMIT, 'dtype': 'float64', 'fastmath': False, 'dt_ms': DT_MS,
              'device': device.name.decode() if isinstance(device.name, bytes) else str(device.name),
              'compute_capability': list(device.compute_capability), 'numba': numba.__version__,
              'python': platform.python_version(), 'platform': platform.platform(),
              'implementation_sha256': {name: hashlib.sha256((ROOT/'src/nexus/brain'/name).read_bytes()).hexdigest()
                                         for name in ('photoreceptor.py', 'photoreceptor_cuda.py')},
              'voltage_tolerance_mv': 1e-7, 'state_rtol': 1e-9, 'state_atol': 1e-10,
              'maximum_voltage_error_mv': max(r['maximum_voltage_error_mv'] for r in rows),
              'groups': groups, 'trials': rows,
              'limits': ['Tests prescribed channel counts and membrane integration only.',
                         'Warm timings include device allocations, host/device transfers, synchronization, validation and host state commit.',
                         'Compilation and model construction are excluded; three repeats per condition.',
                         'GPU and CPU floating-point results are tolerance-checked, not claimed bitwise identical.',
                         'No stochastic molecular events, optical calibration, transmitter release or live GUI behavior are changed.']}
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
