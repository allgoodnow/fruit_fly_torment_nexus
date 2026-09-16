# Published photoreceptor membrane component

The research module `nexus.brain.photoreceptor` implements the membrane subsystem
from [Juusola et al. (2017)](https://doi.org/10.7554/eLife.26117), using the
[authors' source](https://github.com/JuusolaLab/Microsaccadic_Sampling_Paper/tree/a4453f7e47abf2ea2a924c376d4c1c157c6011e2/BiophysicalPhotoreceptorModel).
It is **not yet connected to the application's visual input**. The existing
video pathway and brain behavior remain unchanged.

## What is implemented

Each R1–R6 cell has nine continuous state variables: membrane voltage, five
potassium-channel gating variables, and intracellular sodium, potassium and
calcium concentrations. The equations include Shab, Shaker and novel potassium
conductances, chloride conductance, leaks, and the upstream pump/exchanger terms.
They use the BG1 parameters selected in `Vol_FeedbackCluster.m`.

The input is a current trace in **nanoamperes**, held constant within each 0.1 ms
interval. The output is voltage in **millivolts at the end of each interval**.
There is no spike threshold or voltage reset. Reset restores the published
−70 mV initial condition and initial gates/concentrations; that initial condition
is not claimed to be an equilibrated dark state. The fixed-step RK4 solver
preserves state across calls and rejects nonfinite or invalid gate/ion states
without committing a partial update. Removable singularities in gate-rate
formulas are evaluated using their analytic limits; voltage is not clipped.

```python
import numpy as np
from nexus.brain.photoreceptor import PhotoreceptorMembrane

cell = PhotoreceptorMembrane()
current_na = np.zeros((3000, 1))  # 300 ms, one cell
current_na[1000:2000] = .2       # 100 ms current pulse
voltage_mv = cell.advance(current_na)
```

## Light-sensitive channel current

`advance_channels(open_channels)` accepts a `(ticks, cells)` array of
nonnegative integer counts of open TRP channels. Each count is held over a
0.1 ms interval. It uses the effective current rule in the authors'
`Vol_FeedbackCluster.m`:

```text
I_TRP [nA] = open_channels × 8 [pS] × max(20 − voltage [mV], 0) × 10⁻⁶
```

The 8 pS channel conductance and +20 mV reversal come from that source, as does
the inward-only rectification. This is the source's fixed-reversal approximation,
not a full GHK ion-permeability calculation. For example, one open channel at
−70 mV supplies 0.72 pA, or 0.00072 nA.

The current is recalculated at every RK4 substage using the changing membrane
voltage. As voltage rises toward the reversal potential, the driving force
decreases. The resulting current also enters the source's sodium, potassium and
calcium equations. No spike reset or voltage clamp is applied. Closing channels
removes their current while membrane and ion states recover continuously.

This is a change in numerical coupling: the source performs 100 offline
iterations between a voltage trace and a reconstructed current trace; this
implementation solves their continuous coupled equations. Agreement with those
equations is checked separately and is not described as reproducing the
upstream iteration procedure.

Channel counts must be supplied explicitly. They are **not photon counts**:
producing them from absorbed photons still requires the biochemical cascade.
There is no automatic conversion from pixels or scale-up from fewer microvilli.
The current-input API remains available and retains its previous results.

## Verification and scope

`scripts/verify_photoreceptor_membrane.py` checks source hashes, extracts scalar
expressions directly from the MATLAB equations, and evaluates them in Python.
It compares derivatives at 64 non-equilibrium states, then integrates five
current-pulse protocols with SciPy DOP853 at tighter tolerances. Integration is
split at every current discontinuity. This provides a separate reference for
the fixed-step solver; **it is not a MATLAB execution or validation against
biological recordings**.

The protocols have 100 ms at zero current, 100 ms at 0, 0.05, 0.2, 1 or 3 nA,
then 100 ms at zero. Results are in
`experiments/photoreceptor-membrane-v1-results.json`. The checked source revision
is `a4453f7e47abf2ea2a924c376d4c1c157c6011e2`. To reproduce the report after
checking out that revision:

```sh
python scripts/verify_photoreceptor_membrane.py \
  --source-dir /path/to/Microsaccadic_Sampling_Paper/BiophysicalPhotoreceptorModel \
  --output-dir new-results-folder
```

The generated `reference-fixture.npz` supplies the committed regression fixture
in `tests/fixtures/photoreceptor-membrane-v1.npz`. Additional tests cover
continuity at rate singularities, current removal, independent cells, exact
chunking, reset, and failed-update recovery. No original recordings are included.

`scripts/verify_photoreceptor_channels.py` additionally extracts the TRP current
expressions from the same pinned source. It verifies currents and derivatives
at 64 states, then compares five prescribed channel-pulse experiments against
source equations integrated with DOP853. Each experiment contains 100 ms with
closed channels, 100 ms with 0, 27, 270, 2,700 or 5,400 open channels, then 100 ms
closed. These are numerical experiments, not recordings or simulated light
responses. The largest observed voltage error is below 0.00024 mV.

The same channel trains are also run with their driving force fixed at −70 mV
to isolate the implemented voltage feedback. With 5,400 open channels, the
feedback case ends its pulse at −41.14 mV and 2.64 nA, compared with a fixed
3.89 nA drive. Results and source hashes are in
`experiments/photoreceptor-channels-v1-results.json`. The corresponding reference
fixture is `tests/fixtures/photoreceptor-channels-v1.npz`. Run the channel verifier
with the same `--source-dir` and a new `--output-dir` to reproduce it.

The recorded warmed CPU benchmark takes about 0.25 wall-clock seconds for 10 ms
across 3,377 independent cells. This is roughly 25 times slower than real time
for this subsystem alone; it does not include the biochemical cascade, brain,
or body. It supports short offline experiments, not a real-time performance claim.

## What this does not establish

The authors' complete light-response model also includes photon absorption and
stochastic biochemical events within microvilli. A separate
[offline photon-to-voltage component](phototransduction.md) now implements the
source's reaction network with an explicitly different event scheduler. It has
not been validated as a reproduction of the complete published simulator.
The membrane and channel-current feedback alone do not reproduce the full
phototransduction/adaptation model.

An MP4 supplies pixel values, not measured photon flux or current. Connecting
this subsystem to video requires an explicit optical/light-to-current model.
Connecting its voltage to the existing synaptic weights also requires a
voltage-to-transmitter-release model. Neither conversion has been supplied or
silently fitted to make the brain display more activity. The app's current
generic spiking photoreceptors remain an acknowledged limitation until those
interfaces are implemented and checked.

This component is adapted from GPL-3.0 source; attribution and modifications are
in `licenses/photoreceptor/NOTICE.md`. It is currently outside the application's
import graph and is not a new release feature.
