# Offline photon-to-voltage model

`nexus.brain.phototransduction.Phototransduction` connects absorbed photon counts
to molecular reactions, open TRP channels, and the BG1 membrane subsystem.
It is a **research numerical variant**, outside the application's import graph.
It is not available in the live GUI and does not alter the loaded connectome.

## Source and implemented physics

The reaction network and parameters are adapted from Zhuoyi Song's June 2017
code accompanying [Juusola et al., eLife 6:e26117](https://doi.org/10.7554/eLife.26117).
The [source revision](https://github.com/JuusolaLab/Microsaccadic_Sampling_Paper/tree/a4453f7e47abf2ea2a924c376d4c1c157c6011e2/BiophysicalPhotoreceptorModel)
is pinned. The component retains:

- The 27 reaction parameters from `Initial_PR_Goodaa1new.m` and the selected
  negative-feedback strength of 50 from the simulation driver.
- Activation/deactivation rates for rhodopsin, G proteins, PLC, the source's
  intermediate messenger A, TRP channels, and calcium-bound calmodulin.
- The source's GHK current calculation and fast-calcium algebraic update at
  a fixed −70 mV microvillus voltage.
- A 50-molecule G-protein pool per microvillus, with transfers into active,
  PLC-bound, available and refractory pools. Each microvillus has at most
  27 open TRP channels.
- Uniform random photon assignment across an explicit population of 30,000
  microvilli by default. There is no multiplier applied to smaller populations.
- Summation of open channels into the previously verified 8 pS/+20 mV current
  feedback rule and the nine-variable BG1 membrane. See
  [the membrane component](photoreceptor-membrane.md).

The microvillus calcium variable is distinct from the membrane model's bulk
intracellular calcium. As in the source's two-stage calculation, the cascade
uses its fixed-voltage calculation; whole-cell voltage feedback affects the
current produced by the summed channels. This is not a fully bidirectional
calcium/voltage simulation across all compartments.

## Numerical variant and limits

**Source-equation agreement does not mean full-simulator equivalence.** The
MATLAB event loop selects and updates a reaction before its subsequent waiting
time, uses a bounded uniform draw, adds an empirical latency rate, rounds times
to 0.1 ms, and can protect a first rhodopsin deactivation after light input.
This implementation uses a different, explicitly documented schedule:

1. Calculate reaction propensities from the current molecular state, then make
   the source's one-pass fast-calcium update.
2. Schedule an unrounded exponential wait from the summed propensities and
   choose a reaction proportional to those propensities. The event executes at
   the scheduled time. The source's empirical waiting-time offset, uniform
   floor and special rhodopsin protection are not applied.
3. Incoming photons activate rhodopsin at 1 ms sample boundaries. They cause
   pending waits/reactions to be resampled using the changed propensities.
4. Sample the summed open-channel count at the **left** edge of each 0.1 ms
   membrane interval. The membrane returns voltage at the interval's end.

Fast calcium is refreshed once per event preparation, including preparation
after photon arrival; it is not continuously integrated or solved to a fixed
point. Because its update is mixed with stochastic reactions, this is not
claimed to be an exact Gillespie solution of a fully specified continuous
biochemical system. A direct timing comparison against the original cascade
running in GNU Octave found substantially larger integrated channel responses
in this variant, plus a reproducible duplicate-input defect in the upstream
loop. See the [timing audit](phototransduction-timing-audit.md). The variant is
not a calibrated reproduction; population responses and physiological accuracy
remain unvalidated.

The event generator is NumPy's seeded default generator, not MATLAB's original
MT19937 stream. Random draws, molecular state, pending events and membrane
state persist between calls. Chunking a given input trace gives exactly the
same result. A failed advance commits none of those states. Invalid or negative
propensities cause an error; they are not clipped. An explicit work limit of
one million reaction events in a sampling update prevents runaway integration;
this is a computational guard, not a biological parameter.

The input is **absorbed photons per 1 ms sample**, not photons per second,
incident illumination, lux, RGB, or neural spikes. Counts occur at each sample
boundary; their sub-millisecond arrival times are not reconstructed. No optical
calibration, pixel-to-photon mapping, transmitter-release model, or live brain
connection is supplied. Quantum bumps are molecular channel responses and are
never represented as brain action potentials.

## Use and verification

```python
import numpy as np
from nexus.brain.phototransduction import Phototransduction

receptor = Phototransduction(seed=73100)  # 30,000 explicit microvilli
photons = np.zeros(300, dtype=np.int64)   # 300 ms
photons[50] = 3000                      # absorbed photon pulse at 50 ms
result = receptor.advance(photons)
# 3,000 samples each: result['open_channels'], result['voltage_mv']
```

`scripts/verify_phototransduction.py` checks source hashes, extracts parameters
and scalar expressions from the MATLAB files, and compares reaction rates, GHK
currents and calcium updates at 128 molecular states. This independent
expression evaluation does not execute the original event scheduler.

The verifier then runs three conditions across three seeds, each with a full
30,000-microvillus receptor for 300 ms: darkness, a 3,000-photon pulse at 50 ms,
and 30 photons per ms during 50–150 ms. The latter two contain the same total
absorbed photons. Numerical results, source hashes and limitations are in
`experiments/phototransduction-v1-results.json`; they are not physiological
measurements. The regression fixture is `tests/fixtures/phototransduction-v1.npz`.

```sh
python scripts/verify_phototransduction.py \
  --source-dir /path/to/Microsaccadic_Sampling_Paper/BiophysicalPhotoreceptorModel \
  --output-dir new-results-folder
```

Tests also check dark behavior, causal input timing, the unscaled channel-to-
membrane handoff, molecular-pool limits, exact chunking, reset, and rollback
after a failed membrane update.

The observed photon-driven runs take about 0.45 seconds per 300 simulated ms
for **one receptor** after compilation on the development CPU. That does not
establish usable performance for all 3,377 mapped receptors. Each full receptor
also maintains its own 30,000 molecular states. Scaling to the whole eye is a
separate unresolved engineering problem; this implementation supplies a
mechanistic offline reference.

Attribution, the GPL-3.0 license, and changes from upstream are recorded in
`licenses/photoreceptor/NOTICE.md`.
