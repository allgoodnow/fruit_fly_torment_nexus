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

## What this does not establish

The authors' complete light-response model also includes photon absorption,
stochastic biochemical events within microvilli, summed light-induced currents
and voltage feedback on those currents. Those components are not implemented
here. Potassium-channel and ion dynamics alone do not reproduce the full
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
