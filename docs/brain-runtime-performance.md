# Brain runtime execution

The source build advances each neuron's membrane state independently on up to
four CPU threads for graphs with at least 10,000 neurons. Smaller graphs use one
thread by default. This applies to both the coupled native app and unattended
brain runs; it does not require CUDA or change the interface.

Only membrane integration runs in parallel. Threshold detection, spike queues,
synaptic deliveries, graded release, receptor clamps and manual input retain
their existing sequence. Each membrane task writes only its own cell's state.
The implementation keeps float64, disables fast math and uses the same 0.1 ms
step, equations, weights, refractory periods and delays.

Diagnostics expose `runtime.backend`, `runtime.integration_threads`,
`runtime.synaptic_delivery` and `runtime.dt_ms`. Pausing, releasing and resetting
retain the selected execution policy. The calling thread's previous Numba thread
mask is restored after each advance.

## Choosing a thread count

`Brain(graph, integration_threads=1)` selects serial execution explicitly.
For the app or scripts, set `NEXUS_BRAIN_THREADS=1` before launching. A positive
integer selects another count within Numba's configured worker limit. The usual
default is four, capped by that limit. This is an execution setting, not a new
biological parameter. On small models, launch overhead can outweigh any gain.

The packaging configuration includes Numba's TBB, OpenMP and workqueue components.
This change has been checked in the Linux source app; the updated packaged app
and macOS performance have not been validated. The released 1.1.1 archive still
uses its previous runtime.

## Reproduce the ordinary replay comparison

On the Ryzen 7 7735HS laptop, one 500 ms prepared replay of the 166,700-neuron,
25,582,938-edge MaleCNS graph took **11.89 s** with the previous serial runtime
and **5.82 s** with four threads: **2.04× faster**. All checked state matched
exactly at all 50 chunk boundaries, including spike times and pending synaptic
events. The run used Numba 0.67.0 with its TBB threading layer. This is one
ordinary timing per implementation; workload, CPU state and threading backend
affect performance. The compact result is
[`brain-runtime-parallel-v1-results.json`](../experiments/brain-runtime-parallel-v1-results.json).

The brain alone still advances slower than real time. A short GPU integration
probe also measured substantial per-tick host/device transfer overhead when
retaining serial CPU synaptic delivery. It is not a benchmark of an entirely
GPU-resident brain, which has not been implemented or validated here. CUDA
preparation of molecular receptor responses remains available independently.

Use an existing prepared receptor response and its explicit release curve:

```sh
python scripts/benchmark_brain_runtime.py \
  --response-dir runs/video_phototransduction_v1_final \
  --release-curve experiments/receptor-release-assay-curve.json \
  --threads 4 \
  --output-dir runs/new-brain-runtime-comparison
```

This performs one serial and one parallel replay, checks their complete neural
state, pending events, returned traces, raster, activity counts and random state
at every 10 ms boundary, and saves a compact report. It excludes loading, JIT
warmup, hashing, rendering and the articulated body from the measured interval.
There is no stress matrix.

To compare against an earlier implementation instead of today's serial path,
save that commit's `src/nexus/brain/runtime.py` and pass its path with
`--baseline-file`. The saved module is executable Python: use a trusted copy
from this repository.

The benchmark is an execution check. It does not calibrate spontaneous activity,
receptor histamine release or motor behavior. CPU speedups do not establish
real-time performance for the complete app or the whole molecular eye.
