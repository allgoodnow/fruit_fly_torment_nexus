![Fruit Fly Torment Nexus](docs/assets/FRUITFLYTORMENTNEXUSBURNING.gif)

Built this based on the question "At what point does it become unethical?".



# Fruit Fly Torment Nexus

**1.0 experimental release** — a native Linux application with a 3D fly body,
a 166,700-neuron MaleCNS model, live brain/VNC activity, and timed neural interventions.
The interface is white with black text and red stimulation labels and logs.

## Download and run

Get the Linux archive from [GitHub Releases](https://github.com/allgoodnow/fruit_fly_torment_nexus/releases).
Extract it, open `FruitFlyNexus`, and run `fruit-fly-nexus`. Keep `_internal` beside
the executable. Python, the neural data, and fly assets are bundled; no network
connection, JavaScript, or GPU computing setup is required at runtime.

The build targets **Linux x86-64**. Fedora 44 is the tested platform; other Linux
versions, Windows, and macOS are not validated. The GUI needs a working desktop
OpenGL driver. First startup compiles the neural kernel and can take longer than
subsequent loads. Simulation time can advance slower than wall-clock time.

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
validated biological diagnoses. The nominal 100°C setting does not simulate tissue
damage. This app cannot establish that a digital fly experiences suffering.

## Development and validation

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
