# Fruit Fly Torment Nexus

Native Python/Qt application with an articulated 3D NeuroMechFly body and a
persistent 127,400-neuron connectome model. **Version 0.5.0 adds a visible food
patch whose actual foot contacts drive the reference taste neurons.** DNa02
neural activity still controls left/right walking drive on a shared clock. The
white lab interface, black text, red logs and live anatomical 3D brain remain.
The motor decoder, baseline gait and pooled contact-to-taste mapping are engineering
approximations. Food-seeking and feeding behavior remain unfinished. Audio is deferred.

## Run the Linux prototype

Extract `fruit-fly-nexus-0.5.0-linux-x86_64.tar.gz`, open `FruitFlyNexus`, and run
`./fruit-fly-nexus`. Keep the `_internal` folder alongside the executable.
Python is bundled. The build targets Linux x86-64 and was built on Fedora 44;
compatibility with older Linux distributions and other operating systems is
not yet established. A functioning desktop OpenGL driver is required.

The default session loads both models and opens the Brain tab. Select **DNa02
left** or **DNa02 right**, then choose **Run sequence** to see neural steering.
**Run / Pause**, **Step 10 ms**, and **Reset body + brain** operate on both models
from either tab. Drag either scene to orbit and scroll to zoom. The Body tab's
**Baseline walking drive** sets the authored walking input. Neural output
attenuates the left or right side of that input. Uncheck **Enable neural steering**
to block the decoder while the neural model continues: this is a causal control.
The Body graph shows the engineered leg-controller oscillators. Export diagnostics
saves recent commands and telemetry, not a restorable simulation checkpoint.

To see environmental feedback, open **Body → Reset + run food demo**. The fly
walks across the visible food patch; actual foot contacts activate the taste
input and produce downstream MN9 spikes. The Body tab can remove food, disable
feedback, move the patch ahead/under the fly, and change its radius or model input
rate. The status beneath the fly separates physical contact from enabled input.
See [docs/FOOD_FEEDBACK.md](docs/FOOD_FEEDBACK.md) for the proxy's limits and tests.

The neural controls also offer sugar sensory cells, MN9, and custom decimal
FlyWire IDs. Set the input rate, and click **Apply stimulation**, then **Run both**.
Applying stimulation while paused changes
the pending input without advancing time. **Silence target outputs** blocks outgoing
synaptic effects and the silenced cells' contribution to the motor decoder.
**Release manual interventions** removes manual inputs and silencing while retaining
voltages, synaptic state, pending delayed activity, decoder filter state and time.
The decoder's 100 ms smoothing means motor effects can decay after release.
Food input continues while contact and feedback are enabled. Removing food clears
only environmental input; manual stimulation remains. Overlapping channel rates
use the higher rate. Disable Taste feedback for isolated manual assays.

For orders prepared in advance, set **Baseline**, **Stimulation**, and **Recovery**,
then choose **Run sequence**. The sequence uses the selected targets and rate.
It starts at the current brain time, releases existing interventions, applies the
stimulus after the baseline, releases it after the pulse, and pauses at the end.
Pause/resume preserves the schedule; manual stimulation, silencing, or release
cancels it. Save/load JSON supports more elaborate sequences. Times are relative
to sequence start and can resolve to 0.1 ms. **Reset body + brain** deliberately starts
over with the same random seed and resets the body and decoder too.

For the earlier independent laboratory mode, run `./fruit-fly-nexus --independent`.
That mode has separate controls and clocks and requires **Brain → Load brain**.
`--run-demo` starts the currently selected default input and both simulations.
`--food-demo` starts a food crossing without manual stimulation.

The 3D brain shows **127,322 actual v630 cell anchors**. Each point is one cell's
annotated location, not its complete branching morphology. Red points mark spikes
within the previous 150 ms of simulated time; black markers identify stimulated
targets. Drag to orbit, scroll to zoom, and use Home to restore the view. The
78 cells without coordinates still participate in the network. Two remote source
coordinates are retained; Fit all includes them. See [docs/BRAIN_VIEW.md](docs/BRAIN_VIEW.md)
for provenance and coordinate handling. The view updates during simulation and
freezes with the brain clock when paused. Loading alone produces no activity.

The spike plot below shows recent neural events by model index. Exact cumulative
spike counts are retained; display history and exported diagnostics are bounded.
There are no validated
fear, nociception, seizure, or subjective-experience presets in this build.

## Develop

The prepared workspace launches with `./run.sh`. For a new checkout, install
Python 3.12 and Git, then:

```sh
git clone https://github.com/NeLy-EPFL/flygym vendor/flygym
git -C vendor/flygym checkout 38c8ec61034cd59bc5ba0de20688d4a3c0000d60
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
git clone https://github.com/philshiu/Drosophila_brain_model vendor/brain-reference
git -C vendor/brain-reference checkout 91bdd1e7dcf193f3e7ca5a8933497fcef63b7960
.venv/bin/python scripts/prepare_brain_pack.py
git clone --branch v1.1.0 --depth 1 https://github.com/flyconnectome/flywire_annotations vendor/flywire-annotations
.venv/bin/python scripts/prepare_brain_anatomy.py
.venv/bin/python scripts/prepare_motor_registry.py
.venv/bin/python launch.py
```

The lock includes research/build tools for later milestones. The app needs no
JavaScript, web server, CUDA, system C++ compiler, or network connection at runtime.
Default fly assets and the prepared brain are bundled locally. Numba/LLVM compiles
the neural kernel at first load. Cache and diagnostic files go in Qt's application-data
location, normally `~/.local/share/Nexus/Fruit Fly Torment Nexus/`.

## Verify and package

```sh
.venv/bin/python -m pytest -q
.venv/bin/python launch.py --coupled-smoke-test runs/coupled_native_acceptance
.venv/bin/python launch.py --brain-smoke-test runs/independent_v05_acceptance
.venv/bin/python scripts/probe_steering.py
.venv/bin/python scripts/steering_endurance.py
.venv/bin/python scripts/probe_food.py
.venv/bin/python launch.py --food-smoke-test runs/food_native_acceptance
.venv/bin/python scripts/endurance.py --seconds 10
.venv/bin/python scripts/brain_endurance.py
.venv/bin/python scripts/prepare_release.py
PYINSTALLER_CONFIG_DIR="$PWD/.runtime/pyinstaller" .venv/bin/python -m PyInstaller --noconfirm packaging/nexus.spec
dist/FruitFlyNexus-0.5.0/fruit-fly-nexus --food-smoke-test runs/food_packaged_acceptance
.venv/bin/python scripts/archive_release.py
```

The native smoke test opens the actual window, exercises physics controls, saves
an application screenshot and JSON report, and exits. It requires display access.
The body/worker tests run without rendering. Packaging includes the required
meshes, trajectories, Python interpreter, shared libraries, and dependency notices.

The numerical full-network comparison is in `scripts/benchmark_brain.py`; it
requires the development-only Brian2 reference and a C++ compiler. See
[docs/PERSISTENT_BRAIN.md](docs/PERSISTENT_BRAIN.md) for measurements and limits.

Read [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the remaining scientific
and engineering work, [PROGRESS.md](PROGRESS.md) for validation, and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution.
The decoder's evidence, equations, measurements and limits are in
[docs/NEURAL_STEERING.md](docs/NEURAL_STEERING.md).
