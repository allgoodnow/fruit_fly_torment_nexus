"""Measure eight explicit receptors with distinct ordinary light protocols."""
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
from nexus.brain.phototransduction_batch import BatchedPhototransduction
from nexus.brain.phototransduction_parallel import ParallelPhototransduction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    seeds = list(range(87200, 87208))
    photons = np.zeros((300, 8), dtype=np.int64)
    photons[50, [1, 5]] = 3000
    photons[50, [2, 6]] = 30000
    photons[50:150, [3, 7]] = 30
    # Two receptors each: dark, moderate pulse, bright pulse and steady input.
    for membrane_backend in ('cpu', 'cuda'):
        BatchedPhototransduction([0, 1], microvilli=129,
                                membrane_backend=membrane_backend).advance([[1, 0]])
    ParallelPhototransduction(microvilli=129, backend='cuda').advance([1])
    references = [ParallelPhototransduction(seed=seed) for seed in seeds]
    t = time.perf_counter()
    expected = [r.advance(photons[:, i]) for i, r in enumerate(references)]
    reference_seconds = time.perf_counter()-t
    rows = []
    configurations = [(1, 'cpu'), (4, 'cpu'), (8, 'cpu'), (8, 'cuda')]
    for repeat in range(3):
        models = [ParallelPhototransduction(seed=s, backend='cuda') for s in seeds]
        t = time.perf_counter()
        single = [m.advance(photons[:, i]) for i, m in enumerate(models)]
        single_seconds = time.perf_counter()-t
        for i in range(8):
            np.testing.assert_array_equal(single[i]['open_channels'], expected[i]['open_channels'])
        for batch_cells, membrane_backend in (configurations if repeat % 2 == 0 else configurations[::-1]):
            model = BatchedPhototransduction(seeds, batch_cells=batch_cells,
                                            membrane_backend=membrane_backend)
            t = time.perf_counter()
            actual = model.advance(photons)
            seconds = time.perf_counter()-t
            max_voltage_difference = 0.
            for i, ref in enumerate(references):
                region = slice(i*30000, (i+1)*30000)
                np.testing.assert_array_equal(actual['open_channels'][:, i], expected[i]['open_channels'])
                np.testing.assert_allclose(actual['voltage_mv'][:, i], expected[i]['voltage_mv'], atol=1e-7, rtol=0)
                np.testing.assert_array_equal(model.streams[region], ref.streams)
                np.testing.assert_array_equal(model.reactions[region], ref.reactions)
                np.testing.assert_allclose(model.states[region], ref.states, rtol=1e-9, atol=1e-9)
                np.testing.assert_allclose(model.due[region], ref.due, rtol=1e-10, atol=1e-8)
                np.testing.assert_allclose(model.reversals[region], ref.reversals, rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(model.membrane.state[i], ref.membrane.state[0], rtol=1e-9, atol=1e-10)
                assert model.rngs[i].bit_generator.state == ref.rng.bit_generator.state
                assert model.events[i] == ref.events and model.photons[i] == ref.photons
                max_voltage_difference = max(max_voltage_difference, float(
                    np.abs(actual['voltage_mv'][:, i]-expected[i]['voltage_mv']).max()))
            row = {'repeat': repeat, 'batch_cells': batch_cells, 'membrane_backend': membrane_backend,
                   'wall_seconds': seconds, 'independent_cuda_seconds': single_seconds,
                   'maximum_voltage_difference_mv': max_voltage_difference,
                   'exact_channel_trace_and_rng': True,
                   'events_per_receptor': model.events.tolist(),
                   'molecular_host_state_bytes': model.molecular_state_bytes}
            print(json.dumps(row), flush=True)
            rows.append(row)
    medians = []
    for batch_cells, membrane_backend in configurations:
        group = [r for r in rows if r['batch_cells']==batch_cells and r['membrane_backend']==membrane_backend]
        median = float(np.median([r['wall_seconds'] for r in group]))
        sequential = float(np.median([r['independent_cuda_seconds'] for r in group]))
        medians.append({'batch_cells':batch_cells,'membrane_backend':membrane_backend,
                        'wall_seconds':median,'independent_cuda_seconds':sequential,
                        'speedup_vs_independent_cuda':sequential/median})
    names = ('phototransduction.py', 'phototransduction_parallel.py', 'phototransduction_cuda.py',
             'phototransduction_batch.py', 'photoreceptor.py', 'photoreceptor_cuda.py')
    report = {'format':'nexus-cuda-receptor-batch-1','success':True,
              'device':cuda.get_current_device().name.decode(),'numba':numba.__version__,
              'receptors':8,'microvilli_per_receptor':30000,'simulated_ms':300,
              'seeds':seeds,'protocols':['dark','pulse_3000','pulse_30000','steady_30_per_ms']*2,
              'dt_ms':.1,'internal_chunk_ms':20,'dtype':'float64','fastmath':False,
              'matching_cpu_reference_seconds':reference_seconds,
              'medians':medians,'trials':rows,
              'implementation_sha256':{name:hashlib.sha256((ROOT/'src/nexus/brain'/name).read_bytes()).hexdigest()
                                       for name in names},
              'limits':['One eight-receptor workload on one GPU, not a whole-eye performance claim.',
                        'Timings include whole advance calls, but exclude construction, stream initialization and compilation.',
                        'Matching CPU reference uses per-unit streams, not the older optimized scheduler.',
                        'All receptors retain full independent molecular states; no population multiplier or shared response.',
                        'Photon allocation and between-call storage remain on CPU; live vision is not connected.']}
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
