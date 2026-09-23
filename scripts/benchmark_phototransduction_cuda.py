"""Compare the per-unit-stream CPU/GPU cascades and report whole-call timing."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import numba
from numba import cuda
from nexus.brain.phototransduction import Phototransduction
from nexus.brain.phototransduction_parallel import ParallelPhototransduction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if not cuda.is_available() or cuda.config.ENABLE_CUDASIM:
        raise RuntimeError('Requires physical CUDA hardware')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for backend in ('cpu', 'cuda'):
        ParallelPhototransduction(microvilli=129, backend=backend).advance([1, 0])
    Phototransduction(microvilli=1).advance([0])
    rows = []
    for seed in (84100, 84101, 84102):
        for condition in ('dark', 'pulse', 'bright', 'steady', 'paired'):
            photons = np.zeros(300, dtype=np.int64)
            if condition == 'pulse': photons[50] = 3000
            if condition == 'bright': photons[50] = 30000
            if condition == 'steady': photons[50:150] = 30
            if condition == 'paired': photons[[50, 150]] = 3000
            cpu = ParallelPhototransduction(seed=seed)
            gpu = ParallelPhototransduction(seed=seed, backend='cuda')
            old = Phototransduction(seed=seed)
            models = {'cpu_parallel': cpu, 'cuda_parallel': gpu, 'legacy_cpu': old}
            times, outputs = {}, {}
            order = list(models) if len(rows)%2 == 0 else list(reversed(models))
            for name in order:
                t = time.perf_counter()
                outputs[name] = models[name].advance(photons)
                times[name] = time.perf_counter()-t
            a, b = outputs['cpu_parallel'], outputs['cuda_parallel']
            np.testing.assert_array_equal(a['open_channels'], b['open_channels'])
            np.testing.assert_allclose(a['voltage_mv'], b['voltage_mv'], rtol=0, atol=1e-7)
            np.testing.assert_array_equal(cpu.streams, gpu.streams)
            np.testing.assert_array_equal(cpu.reactions, gpu.reactions)
            np.testing.assert_allclose(cpu.states, gpu.states, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(cpu.reversals, gpu.reversals, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(cpu.due, gpu.due, rtol=1e-10, atol=1e-8)
            assert cpu.events == gpu.events and cpu.rng.bit_generator.state == gpu.rng.bit_generator.state
            row = {'seed': seed, 'condition': condition, 'microvilli': 30000, 'simulated_ms': 300,
                   'wall_seconds': times, 'events_parallel': cpu.events,
                   'exact_channel_trace_and_rng': True,
                   'maximum_state_difference': float(np.abs(cpu.states-gpu.states).max()),
                   'channel_area_channel_ms': {name: float(o['open_channels'].sum()*.1) for name,o in outputs.items()}}
            rows.append(row)
            print(json.dumps(row), flush=True)
    groups = {}
    for condition in ('dark','pulse','bright','steady','paired'):
        group = [r for r in rows if r['condition']==condition]
        medians = {name: float(np.median([r['wall_seconds'][name] for r in group])) for name in models}
        groups[condition] = {'median_wall_seconds': medians,
            'speedup_vs_matching_cpu': medians['cpu_parallel']/medians['cuda_parallel'],
            'speedup_vs_legacy_cpu': medians['legacy_cpu']/medians['cuda_parallel']}
    # Distribution characterization, not matching seeds or an equivalence test.
    ensemble = []
    for seed in range(85100, 85228):
        for condition in ('single', 'paired_50ms', 'paired_500ms'):
            gap = {'single':0, 'paired_50ms':50, 'paired_500ms':500}[condition]
            photons = np.zeros(20+gap+300, dtype=np.int64); photons[20]=1
            if gap: photons[20+gap]=1
            old, new = Phototransduction(microvilli=1, seed=seed), ParallelPhototransduction(microvilli=1, seed=seed)
            traces = [model.advance(photons)['open_channels'] for model in (old,new)]
            ensemble.append({'seed':seed,'condition':condition,
                             'legacy_area':float(traces[0].sum()*.1),'parallel_area':float(traces[1].sum()*.1)})
    summaries = {}
    for condition in ('single','paired_50ms','paired_500ms'):
        group = [r for r in ensemble if r['condition']==condition]
        summaries[condition] = {key: {'mean':float(np.mean([r[key] for r in group])),
                                      'std':float(np.std([r[key] for r in group],ddof=1))}
                                for key in ('legacy_area','parallel_area')}
    report = {'format':'nexus-cuda-cascade-1','success':True,
              'device':cuda.get_current_device().name.decode(),'numba':numba.__version__,
              'dt_ms':.1,'dtype':'float64','fastmath':False,'internal_chunk_ms':20,
              'implementation_sha256':{name:hashlib.sha256((ROOT/'src/nexus/brain'/name).read_bytes()).hexdigest()
                  for name in ('phototransduction.py','phototransduction_parallel.py','phototransduction_cuda.py')},
              'groups':groups,'trials':rows,'ensemble_groups':summaries,'ensemble_trials':ensemble,
              'limits':['One complete receptor at a time, not all mapped receptors or the full brain.',
                        'Whole-call timings include photon allocation, transfers, reduction, membrane integration and host commit; exclude initialization and JIT.',
                        'Parallel backend uses separate xoroshiro128+ event streams and inverse-CDF waits; legacy uses a shared NumPy generator.',
                        'Legacy seed trajectories are intentionally different. Ensemble summaries are descriptive, not a physiological fit or predefined equivalence test.',
                        'Photon allocation and the one-cell membrane remain on CPU. Live application is unchanged.']}
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__': main()
