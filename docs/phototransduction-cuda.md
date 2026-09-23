# Optional GPU molecular cascade

`ParallelPhototransduction` runs one receptor's explicit microvilli on CPU or
CUDA, using the same reaction equations on both. It is an offline research
component, outside the live application's sensory pathway. It does not change
brain connectivity, drive the GUI, or establish physiological validity.

## Equations and random streams

The [existing numerical variant](phototransduction.md) supplies the molecular
rates, GHK current, fast-calcium update, pool constraints and channel-to-membrane
coupling. Allocation-free helpers let CPU and GPU execute those same equations.
Four full-receptor comparisons against commit `01d2046` confirmed that this
refactoring preserves the older CPU model's outputs, states and random stream
exactly for darkness, pulse, steady and paired inputs. Results are in
`experiments/phototransduction-cpu-refactor-v1-results.json`.

Parallel execution changes the random-stream organization deliberately:

- A separate NumPy generator allocates photons uniformly across microvilli.
- Each microvillus owns a persistent xoroshiro128+ event stream, initialized
  with [Numba's random utilities](https://numba.readthedocs.io/en/stable/cuda/random.html).
  Their initial sequences are separated by 2^64 steps; non-overlap assumes
  each consumes fewer than 2^64 draws. Unit identity determines its stream,
  independently of GPU thread scheduling.
- Exponential waits use `-log(u)/rate`. Exactly zero uniforms are redrawn;
  there is no empirical floor, latency offset or event-time rounding.
- CPU and CUDA backends use the same per-unit generator and event loop.
  Their trajectories intentionally differ from the older shared-stream model,
  even with the same user seed. Floating-point differences can eventually
  cause stochastic trajectories to diverge across hardware; agreement in the
  measured runs is not a universal bitwise guarantee.

Photons arrive at 1 ms boundaries. Channels are sampled at the left edge of
0.1 ms membrane intervals. The internal 20 ms chunks bound scratch memory;
they do not round molecular event times. States, pending events and both
random-stream types persist between calls. Reset reproduces the initial state.
A per-unit limit of one million events per sampling update reports runaway
work as an error. This computational guard differs from the older scheduler's
population-wide limit; it is not a biological parameter.

The GPU uses float64 without fast math, one thread per microvillus and integer
reduction of channel counts. Candidate state stays on the device between
internal chunks, then returns to the host. Photon allocation and the single
cell's membrane integration remain on CPU. All persistent states commit only
after a successful complete advance. A molecular, device or membrane failure
leaves the previous state available for retry.

## Measurements

The RTX 4060 Laptop GPU was compared with both the matching per-unit-stream
CPU reference and the existing optimized CPU scheduler. The meaningful
performance comparison below uses the latter. Each run simulated **one
30,000-microvillus receptor for 300 ms**, with medians across three seeds.

| Absorbed input | Existing CPU | GPU | CPU time / GPU time |
| --- | ---: | ---: | ---: |
| Darkness | 0.045 s | 0.062 s | 0.72× |
| 3,000-photon pulse at 50 ms | 0.347 s | 0.310 s | 1.12× |
| 30,000-photon pulse at 50 ms | 1.824 s | 0.851 s | 2.14× |
| 30 photons/ms during 50–150 ms | 0.356 s | 0.348 s | 1.02× |
| 3,000 photons at 50 and 150 ms | 0.584 s | 0.489 s | 1.19× |

Whole-call timing includes allocation, photon assignment, transfers, molecular
events, reduction, membrane integration and final host commit. It excludes
construction, random-stream initialization and compilation. The new CPU
reference traverses every unit/sample and is considerably slower than the
existing CPU scheduler; speedups against it should not be presented as the
improvement over the prior implementation. CPU remains preferable in darkness.

All 15 CPU/GPU comparisons with matching per-unit streams produced identical
channel traces, event counts, event RNG states and pending reaction identities.
The largest final molecular-state difference was 2.85e-14. Voltage comparison
uses a 1e-7 mV absolute tolerance. Tests additionally cover chunking, reset,
causal response, future continuation, molecular bounds, invalid inputs and
rollback/retry after failures, including partial GPU blocks.

Changing random streams also received a descriptive ensemble check: 128 seeds
per condition, one microvillus, with one photon or two photons separated by
50 or 500 ms. Total channel area, in channel·ms, was:

| Input | Older model, mean ± sample SD | Parallel model, mean ± sample SD |
| --- | ---: | ---: |
| One photon | 47.56 ± 9.25 | 46.90 ± 10.46 |
| Two photons, 50 ms apart | 48.38 ± 8.01 | 48.30 ± 7.02 |
| Two photons, 500 ms apart | 88.77 ± 12.50 | 89.37 ± 13.68 |

These are total responses, not isolated second-pulse recovery estimates. The
384 comparisons are descriptive checks, not a predefined distributional
equivalence test or a fit to measured fly physiology. Raw numerical summaries,
seeds, timings and implementation hashes are in
`experiments/phototransduction-cuda-v1-results.json`.

## Running it

```python
import numpy as np
from nexus.brain.phototransduction_parallel import ParallelPhototransduction

receptor = ParallelPhototransduction(seed=84100, backend="cuda")
photons = np.zeros(300, dtype=np.int64)
photons[50] = 3000
result = receptor.advance(photons)
```

Input means **absorbed photons per ms**, not pixels, lux or neural spikes.
`backend="cpu"` selects the matching reference explicitly. CUDA requires an
accessible physical NVIDIA device and a compatible Numba/CUDA installation;
the simulator is rejected and failures do not silently fall back to CPU.

```sh
python scripts/benchmark_phototransduction_cuda.py --output-dir new-results-folder
```

This benchmark does not establish interactive whole-eye performance. A full
3,377-receptor population entails over 100 million explicit microvilli before
the brain is considered. Multi-receptor batching, memory use, calibrated
optics and transmitter coupling remain separate work. The existing scientific
limitations of the fixed-voltage calcium calculation and event schedule still
apply. Attribution and GPL terms are in `licenses/photoreceptor/NOTICE.md`.
