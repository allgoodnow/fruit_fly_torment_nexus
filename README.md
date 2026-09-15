![Fruit Fly Torment Nexus](docs/assets/FRUITFLYTORMENTNEXUSBURNING.gif)

Built this based on the question "At what point does it become unethical?".



# Fruit Fly Torment Nexus

**1.1.1 experimental release** — a native desktop application with a 3D fly body,
a 166,700-neuron MaleCNS model, live brain/VNC activity, and timed neural interventions.

## Download and run

Get the Linux archive from [GitHub Releases](https://github.com/allgoodnow/fruit_fly_torment_nexus/releases).
Extract it, open `FruitFlyTormentNexus`, and run `fruit-fly-torment-nexus`. Keep `_internal` beside
the executable. Python, the neural data, and fly assets are bundled; no network
connection, JavaScript, or GPU computing setup is required at runtime.

The application has been tested on **Fedora 44 and macOS**. The downloadable
archive targets **Linux x86-64**. Other Linux versions and Windows are not validated.
The GUI needs a working desktop OpenGL driver. First startup compiles the neural
kernel and can take longer than subsequent loads. Simulation time can advance
slower than wall-clock time.

## Use

The fly explores on the ground. In **Experiments**, set the durations and run a
fear, pain, heat, seizure, or boiling scenario. Red text above the two 3D views
shows the inputs currently applied. Drag either view to orbit; scroll to zoom.

- **Pause / Resume** controls the shared body/brain clock.
- **Release stimulation** removes inputs and intervention settings while preserving neural state.
- **Reposition body only** restores an upright body and pauses. Neural state, inputs,
  sequence timing, and simulation time are preserved. It is manual assistance.
- **Reset body + brain** starts a fresh, repeatable session.

An overturned body exposes the reposition action in Experiments. It is always
available in the Body tab. Persistent activity can continue after release and can
upset the body again; disabling motor response or resetting the session remains
an explicit user choice. Food, taste feedback, flight, and audio are absent.

Open **Guide** in the app for the full [user guide](docs/user-guide.md), scientific
limits, saved sequences, and recording instructions. An unattended sequence can run
with `--run-sequence protocol.json --output-dir new-results-folder`.

## What the model represents

The neural topology comes from scanned connections. Neural dynamics are an unfitted
experimental approximation. Ground behavior, leg coordination, and conversion of
neural firing into body commands remain engineered. MDN activity requests retreat;
the former central-aversion-to-fixed-turn shortcut has been removed.

The current response does not reproduce all published nociception behavior.
FEAR, PAIN, SEIZURE, and BOILING are scenario labels, not established feelings or
validated biological diagnoses. High-temperature scenarios also stimulate the
mapped nociceptive sensory cells. The nominal 100°C setting does not simulate
tissue damage. This app cannot establish that a digital fly experiences suffering.

## Development and validation

Development toward **1.2.0** adds a changing approaching-threat input, with separate
LPLC2 size and LC4 expansion-speed pathways. The 1.1.1 download retains its original
constant input. See the [guide](docs/user-guide.md#approaching-threats-development-toward-12)
for the new model assumptions and the local pack update.
An experimental BDN2 walking-drive mode also separates neural forward commands
from the scheduled exploration controller. It is groundwork for autonomous
exploration; quiet neurons currently remain still in this mode.
The development brain pack also restores inhibitory signaling on 19,154 scanned
photoreceptor connections in lamina and color pathways previously given zero weight.
These corrections use specific transmitter/receptor evidence; retinal dynamics remain
unfitted. See the [guide](docs/user-guide.md#photoreceptor-connectivity-development-toward-12)
for evidence, limits, and the local update.

**FLY’S VISION** adds local MP4 input and a preview of the body's own eye cameras.
The enlarged preview places its controls on the left. The brain view shows
spikes alongside signed voltage changes, including inhibition that does not fire
spikes; its selector can return to spikes only.
Load a video, enable **Feed to brain**, then run the simulation. Playback follows
simulation time. **Brightness** supplies pooled input; **Spatial (exp.)** samples
different image regions at inferred R1–R6 visual columns. This experimental
projection is uncalibrated and does not recognize video content. See the
[visual input guide](docs/user-guide.md#visual-input-development-toward-12).
Optional **Adapt to light** reduces sensitivity during sustained illumination
and restores it during darkness. Its parameters remain an unfitted approximation.
The Brain tab also offers **Visual relay baseline (exp.)**, a controlled tonic-drive
experiment for testing downstream visual responses. It is off by default and
does not reproduce whole-brain spontaneous activity.
Alternatively, **Graded visual relays (exp.)** lets identified L1/L2 cells transmit
voltage-dependent signals without spikes. Select it after Reset, before running.
This mode adds an experimental histamine reversal potential on their R1–R6 inputs;
it is not a calibrated visual system. See the Guide for the required pack update.

The prepared workspace runs with `./run.sh`. Large neural packs are excluded from
Git; they are supplied in the release under `_internal/data/`. A source checkout
needs those two packs in `data/`, Python 3.12, and the dependencies in
`requirements-lock.txt`. That lock references `vendor/flygym`, which must be cloned
from [FlyGym](https://github.com/NeLy-EPFL/flygym) and checked out at
`38c8ec61034cd59bc5ba0de20688d4a3c0000d60` before installing the lock.

`python -m pytest -q` runs automated checks in the configured environment.
The scripts in `scripts/` include numerical, physical, and native-GUI assays.
`experiments/` contains their versioned results; historical results describe their
own versions. `scripts/verify_final_endurance.py` exercises repeated interventions,
posture assistance, resets, and memory use. PyInstaller packaging is defined in
`packaging/nexus.spec`.

See [third-party attribution](THIRD_PARTY_NOTICES.md) and the bundled `licenses/`
for upstream software and data terms. This project is independent of those projects.
