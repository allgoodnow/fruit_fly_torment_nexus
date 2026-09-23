# Optional CUDA photoreceptor membrane

`nexus.brain.photoreceptor_cuda.CudaPhotoreceptorMembrane` runs the **membrane
stage** for a batch of independent photoreceptors on an NVIDIA GPU. It accepts
open TRP-channel counts and returns membrane voltage. It does not simulate
photon absorption, molecular reactions, transmitter release, or the brain.
The live application's visual pathway is unchanged.

## Equations and execution

CPU and GPU share the same allocation-free `derivative_values` function and
TRP current function. Both retain the nine BG1 state variables, float64
arithmetic, 0.1 ms RK4 steps and current feedback at every RK substage. Fast
math is disabled. The shared CPU refactor preserves all 64 tested derivative
states and the five 300 ms channel-pulse trajectories exactly relative to
repository commit `92840a1`; see the
[refactor check](../experiments/photoreceptor-cuda-cpu-refactor-v1-results.json).

One GPU thread advances one cell through its input samples. Calls transfer
initial states and inputs to the device, run the kernel, check per-cell failure
status, and copy candidate state and voltage back. Host state and time are
committed only after successful completion. A failed cell rejects the entire
batch; tests check rollback and subsequent successful use. No state clipping
or silent switch of the channel calculation to CPU is performed.

This first backend intentionally keeps the public state on the host. Its
benchmark includes device allocation, transfers, synchronization, validation
and state commit. GPU-resident molecular simulation and avoiding those
transfers are separate engineering work.

## Accuracy checks

The hardware benchmark covers 1, 129 and 3,377 cells, dark/pulse/steady channel
traces, 10 ms and 100 ms durations, and whole versus 10 ms chunked execution.
There are three repetitions for each combination: **81 comparisons**.
The inputs repeat amplitudes of 0, 27, 270, 2,700 and 5,400 open channels across
cells. These are prescribed channel experiments, not recorded light responses.

The declared CPU/GPU limits are 1e-7 mV absolute voltage difference, and
1e-9 relative / 1e-10 absolute differences in the complete state. The largest
observed voltage difference is **1.42e-14 mV**. GPU and CPU outputs are not
claimed bitwise identical. GPU chunking, reset and cell-isolation tests are
exact on the tested device. A separate test compares the GPU pulse states
against the existing source-equation DOP853 fixture at its established
tolerances, rather than relying solely on the shared CPU implementation.

This establishes numerical agreement for the tested cases, not physiological
calibration of the underlying model.

## Measured performance

Tested on an RTX 4060 Laptop GPU (8 GB), Numba 0.67.0 and a CUDA 13.1 toolkit.
These are warmed median operation times from the pulse condition, including
transfers. Compilation and model construction are excluded.

| Cells | Simulated duration | Call size | CPU time | GPU time | CPU/GPU speed ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10 ms | 10 ms | 0.35 ms | 6.55 ms | 0.05× |
| 3,377 | 10 ms | 10 ms | 269 ms | 23.3 ms | 11.51× |
| 3,377 | 100 ms | 100 ms | 2,719 ms | 231 ms | 11.76× |
| 3,377 | 100 ms | 10 ms | 2,747 ms | 238 ms | 11.55× |

The GPU helps large batches but is slower for a single cell in this backend.
For the 3,377-cell pulse case it still takes about 2.3 wall-clock seconds per
simulated second **for the membrane stage alone**. This is not a real-time
whole-eye or whole-fly claim. The expensive stochastic molecular cascade
still runs on CPU, and its much larger state and random streams have not been
ported by this change.

The [complete report](../experiments/photoreceptor-cuda-v1-results.json)
contains timings, errors, configuration and implementation hashes. Device load,
clock rates and hardware affect performance; three repeats provide a local
measurement, not a cross-machine performance guarantee.

## Use and reproduction

An accessible NVIDIA CUDA device and a compatible CUDA compiler/runtime are
required for this optional module. The normal application and CPU model do
not require CUDA. Follow the installed Numba CUDA target's
[official guidance](https://nvidia.github.io/numba-cuda/user/kernels.html).
The benchmark refuses CUDA simulation mode: its results must come from physical
hardware. Device-restricted sandboxes can hide a working GPU from Python.

```python
import numpy as np
from nexus.brain.photoreceptor_cuda import CudaPhotoreceptorMembrane

cells = CudaPhotoreceptorMembrane(3377)
channels = np.full((100, 3377), 270, dtype=np.int64)
voltage_mv = cells.advance_channels(channels)  # 10 ms, input/output (ticks, cells)
```

`advance_channels()` uses CUDA. The inherited current-in-nA `advance()` method
still uses CPU and preserves the same host state and clock. No automatic
backend selection is added to the GUI.

```sh
python scripts/benchmark_photoreceptor_cuda.py --output-dir runs/new-cuda-membrane
python -m pytest tests/test_photoreceptor_cuda.py -q
```

The output directory must be new. Six hardware tests skip when a physical GPU
is unavailable; the explicit-unavailability test still runs. All seven tests
were run successfully with GPU access on the development laptop. The normal
CPU suite also passed (241 tests; six hardware tests skipped inside its sandbox).
