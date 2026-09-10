> Historical development record. For the 1.0 release, see [README](README.md), [the user guide](docs/user-guide.md), and the versioned results in `experiments/`.

# Implementation progress — 9 September 2026

## Current version: 0.5.0 — contact-driven taste feedback

The native arena now has a visible food patch. Actual MuJoCo distal-foot/ground
contacts within its footprint drive the existing 21-cell sugar cohort through an
explicitly pooled contact proxy. Food placement/removal, feedback ablation, patch
radius and model input rate are controllable in the Body tab. Contact and input
status remain visible beside the simulation. The food demo crosses the patch
without manual stimulation, and releases input on leaving it.

Manual and sensory channels coexist; overlapping targets use the higher rate.
Manual release preserves ongoing food input. Removing food preserves manual input
and neural state. The unchanged neural kernel still passes the reference checks.
19 automated tests and nine native food acceptance checks passed. Visual QA caught
and corrected the cylinder-size convention so the patch renders flat on the floor.

Four full-network/physics assays each ran for 1.2 simulated seconds. Food contact
produced 0.51 s of input, 9,210 spikes and 53 MN9 spikes. Feedback ablation retained
contact but produced zero spikes; removing food produced neither contact nor
spikes. Silencing sugar-cell outputs preserved 2,131 spikes in the driven cells
but eliminated the MN9 response. The fly stayed upright, clocks agreed, and all
input was released by the end of the crossing. Reports: `runs/food_probe/` and
`runs/food_native_acceptance/`. See `docs/FOOD_FEEDBACK.md` for semantics and limits.

This establishes contact-dependent sensory input. Food-seeking, stopping to feed,
proboscis movement, consumption, and satiation remain unimplemented. The measured
taste response did not change the baseline body trajectory through DNa02.

The v0.5.0 frozen app passed all nine food acceptance checks from `/tmp`, without
Python environment variables or a system compiler. Native steering retained all
10 checks with taste feedback disabled, and independent mode retained all 13
checks. Final source tests passed 19/19. Reports: `runs/food_packaged_acceptance/`,
`runs/coupled_v05_acceptance/`, and `runs/independent_v05_acceptance/`.

## Version 0.4.0 — experimental neural steering

The default app now loads brain and body into one worker with a shared clock.
Actual spike counts from the source-matched left/right DNa02 pair modulate the
corresponding walking-controller inputs. New presets select left, right or both.
Run, pause, step, reset and timed sequences coordinate both models. A visible
decoder toggle supplies a causal ablation control. `--independent` preserves the
earlier laboratory mode. Neural activity overlays remain visible even at the
same 3D position as a black target marker.

Six full-network/physics trials show opposite steering from left/right activation.
Blocking the decoder preserves spikes and restores the baseline physical trajectory;
silencing the target's outgoing effects does likewise. A 5 ms versus 10 ms coupling
comparison preserves neural spike totals and changes heading by approximately
0.0041 radians. All clocks agree and all six 0.6-second trials remained upright.
Measured rate is about 0.12× wall time. Raw results: `runs/steering_probe/report.json`.

16 automated tests passed, including coupled clock/sequence boundaries, reset,
outgoing silencing, ablation, command rejection, deduplication and shutdown.
Native coupled acceptance passed 10 checks. Evidence, decoder equations, target
provenance, assay limits and remaining work: `docs/NEURAL_STEERING.md`.

The frozen v0.4.0 app also passed 10 coupled checks from `/tmp`, with no Python
environment or system compiler. The independent mode retained its 13 native
checks after the shared telemetry refactor. A 3-second left/release/right/release
sequence completed in 25.18 wall seconds, stayed upright (minimum sampled cosine
0.9918), kept both clocks aligned, and recovered to a motor-drive difference below
0.006. Reports: `runs/coupled_packaged_acceptance/`,
`runs/independent_v04_acceptance/`, `runs/steering_endurance/report.json`.

Baseline walking and the low-level gait remain engineered. World-to-brain sensory
feedback, neural initiation/stop, intervention characterization and full checkpoints
remain unfinished. This is a first steering bridge, not the completed autonomous loop.

## Version 0.3.0 — white lab and live anatomical brain

The native interface now uses white surfaces, black text, and a shared red event
log. The 3D fly and anatomical brain remain visible together. FlyWire v1.1.0
annotations, explicitly matching v630, provide 127,322 cell anchors in the exact
model index order. All 127,400 neurons still participate in the simulation.
Recent spikes are highlighted using actual last-spike ticks from every neuron;
the view freezes with simulated time when paused and clears on reset. OpenGL
handles rendering while the existing CPU worker handles neural computation.

The 9 existing tests and 3 new anatomy tests passed. Native UI acceptance passed
13 checks, including concurrent body/brain execution, actual OpenGL activity,
reset, and orbiting the paused brain. Reports and screenshots are in
`runs/lab_native_acceptance/`. See `docs/BRAIN_VIEW.md` for coordinate provenance,
the 78 unmapped cells, two retained source outliers, and display limitations.
Body and brain remain independent; audio and circuit curation remain deferred.

The frozen v0.3.0 executable also passed all 13 acceptance checks from `/tmp`,
with Python environment variables removed and no system compiler in PATH.
Packaging explicitly includes both EGL and GLX: the first frozen run exposed a
missing EGL plugin under Wayland, which was fixed before the successful run.
Report and screenshots: `runs/lab_packaged_acceptance/`.

## Version 0.2.0 — persistent brain and timed sequences

The native app now includes the real v630 whole-brain graph in a **Brain** tab:
127,400 neurons and 14,687,178 signed connection rows. Its continuous runtime
supports direct ID-based stimulation, outgoing-effect silencing, release,
pause/step/reset, and sequences prepared before execution. The existing 3D body
and the brain run in separate processes and can advance concurrently.

The brain does **not yet drive the body**. Taste inputs and MN9 are the available
named presets; fear, nociception, and seizure-like interventions still need circuit
curation and characterization. There is no anatomical brain rendering or complete
session checkpoint yet. Audio and final styling remain deferred.

Validated this update:

- **9 tests passed**, covering the body/worker, neural voltage/spike agreement,
  deterministic reset, chunk invariance, preserved state on release, bounded
  recording, input rejection, and exact timed-sequence boundaries. Additional
  checks confirm that output silencing blocks downstream activity and release
  restores it, and that dense activity preserves exact counts while capping
  returned spike records.
- Full-network replay against Brian2: **1,428 identical spikes over 100 ms**,
  maximum final voltage difference **1.22e-12 mV**. This tests the numerical
  implementation, not subjective experience or every possible intervention.
- Persistent neural runtime: **17.63 ms median** per 10 ms advance, **0.567× real
  time** in the short measured taste sequence. It requires no system C++ compiler
  or CUDA; Numba/LLVM handles first-load compilation.
- Ten simulated seconds completed in **25.89 wall seconds** during a concurrent
  packaging build. Peak standalone neural RSS stabilized at **381.57 MiB**.
  History stayed capped at 20,000 records while exact total counts reached
  136,735 spikes. Report: `runs/brain_endurance/report.json`.
- Source native acceptance passed **11 checks**, including full brain loading,
  precisely timed inputs/releases, recorded neural spikes, paused manual controls,
  reset, and concurrent brain/body run and pause.
- The final frozen brain build passed all **11 native acceptance checks** from
  `/tmp`, with Python/development variables removed, PATH set to a directory with
  no tools, and CC/CXX pointing to nonexistent compilers. Full neural loading,
  first-use JIT, timed stimulation, and concurrent 3D rendering all passed.
  Report and screenshots: `runs/neural_packaged_acceptance/`.
- The **320,203,523-byte v0.2.0 archive** passed SHA-256 verification and all
  11 checks after extraction into `/tmp/nexus-v02-release-check`. Report:
  `runs/neural_archive_acceptance/`. This remains a test on the current Fedora
  host, not proof of clean-machine or cross-distribution compatibility.

Implementation and scientific limits: `docs/PERSISTENT_BRAIN.md`.
The earlier 0.1.0 work below is retained as history; statements about the brain
being absent describe that earlier version.

## Body prototype (0.1.0)

Implemented a native PySide6 desktop app, separate simulation process, actual
MuJoCo 3D renderer, and official NeuroMechFly hybrid walking controller.
The scene supports run/pause, single-step, deterministic reset, orbit/zoom,
automatic steering, stride drive, left/right bias, and diagnostics export.
Plots and labels explicitly describe the engineered controller. The connectome
and neural intervention buttons are not attached yet.

The body has 67 joints and 48 actuators (42 position actuators and 6 adhesion
actuators). Controller and physics use 0.1 ms simulation steps. Camera tracking
uses the body's actual thorax position. Rendering runs in the physics worker;
the native UI polls bounded display queues so it can stay responsive.

## Verified

- Source app acceptance passed: advances and renders, pause freezes simulation
  time, one step advances exactly 10 ms, camera redraws while paused, and reset
  reinitializes state. Screenshot and report: `runs/native_acceptance/`.
- Body and worker tests: **2 passed**. Reset repeats a trajectory; small chunks
  match one large advance; camera does not mutate physics; repeated command IDs
  cannot duplicate a step; mixed commands retain order; shutdown exits cleanly.
- Ten simulated seconds of autonomous walking completed in **67.85 wall seconds**
  on the current CPU (about **0.147× real time**, headless). Thorax height remained
  0.800–1.197 mm, minimum upright-axis cosine was 0.973, state remained finite,
  and maximum displacement was 60.90 mm. Report: `runs/endurance/metrics.json`.
- Default simplified mesh assets work locally without a runtime download.
- Standalone PyInstaller acceptance **passed all five checks** from `/tmp` with
  `PYTHONPATH` and `VIRTUAL_ENV` removed and `PATH=/usr/bin:/bin`. It loads the
  bundled interpreter, Qt, MuJoCo/GLFW, fly meshes, and walking trajectories.
  Screenshot/report: `runs/packaged_acceptance/`. The extracted bundle is about
  701 MiB. Initial packaging failures led to explicit inclusion of GLFW,
  IPython (required by mediapy), dependency metadata, and the legacy NumPy import
  needed to read the upstream trajectory asset.
- The actual 267,817,838-byte download archive was SHA-256 verified, extracted
  into a separate `/tmp` folder, and passed the same native acceptance checks.
  Report: `runs/archive_acceptance/`. This still uses the host's display drivers
  and system libraries; it is not a separate clean-machine test.

## Machine and runtime

Fedora 44 Linux x86-64; Ryzen 7 7735HS (8 cores / 16 threads), about 14 GiB reported
RAM. Bundled Python 3.12.14 supports the current scientific dependencies.
FlyGym 2.1.0 is pinned to commit `38c8ec61034cd59bc5ba0de20688d4a3c0000d60`.
MuJoCo 3.9.0, PySide6 6.11.2. Full environment: `environment.json` and
`requirements-lock.txt`.

The NVIDIA RTX 4060 is visible but NVIDIA-SMI cannot communicate with its driver.
The integrated AMD Radeon 680M provides accelerated OpenGL 4.6 / Mesa 26.1.8 and
successfully renders the app. No graphics driver changes were needed.

## Limits of version 0.1.0

Current physics is slower than real time; simulated time is displayed explicitly.
There is no full brain runtime, brain-to-body mapping, neural intervention,
anatomical brain view, restorable session checkpoint, or flight controller yet.
Diagnostic export records recent observations and commands only.

Next: finish the reference assay comparison, resolve candidate circuit IDs, and measure persistent-runtime
performance before connecting neural output to the body. See the full
`IMPLEMENTATION_PLAN.md` for the six project milestones.

## Brain reference work started

Cloned the unmodified reference repository at commit
`91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`. Both included snapshots pass uniqueness,
index-bounds, index-to-ID, contact-count, and signed-weight consistency checks.
Source file hashes and counts are in `data/manifests/brain-reference.json`.

| Snapshot | Neurons | Connection rows | Sum of contact counts |
|---|---:|---:|---:|
| v630 | 127,400 | 14,687,178 | 52,793,639 |
| v783 | 138,639 | 15,091,983 | 54,492,922 |

These are the reference tables' counts, not a claim of unique edges or of complete
coverage of all cells/synapses in the animal. Source neuron IDs stay 64-bit integers
internally and decimal strings in JSON.

The harness `scripts/brain_assay.py` calls upstream `run_trial` without modifying
its equations. It records actual spike times, seeds, MN9 rates, memory, and timing.
It uses Cython/GCC for this research run; this is not yet a compiler-free packaged
brain runtime. No-input baseline produced zero spikes. Three trials each at
100/150/200 Hz input are an initial integration sample, not the reference's 30-trial
reproduction or a validation of nociception/defense/seizure behavior.

MN9 means were 62.33, 81.00, and 91.67 Hz respectively. The upstream saved
`sugarR` file has driven-cell activity closer to 200 Hz despite the current code's
150 Hz default. This explains a misleading apparent mismatch when comparing
against that archive as though it used the current defaults. Full measurements,
limits, and reproduction commands are in `docs/BRAIN_REFERENCE.md`.

The original sugar target `720575940620900446` is absent from the v783 table.
The v783 harness intentionally rejects missing IDs; resolve that target using
documented snapshot lineage before repeating the comparison. MN9 is present.

The Linux build is an early local prototype. A run on the development machine
from outside the source directory is useful packaging verification but does not
establish clean-machine or cross-distribution compatibility.

## Version 0.3.0 archive verification

The final 321,798,922-byte archive was SHA-256 verified, extracted into a fresh
`/tmp/nexus-v03-release-check/` directory, and passed all 13 native checks with
no Python environment or system compiler. Report: `runs/lab_archive_acceptance/`.
Checksum and release metadata: `dist/SHA256SUMS`, `dist/release-0.3.0.json`.

## Version 0.4.0 archive verification

The final 321,802,510-byte archive was SHA-256 verified, extracted to a fresh
`/tmp/nexus-v04-release-check/`, and passed all 10 coupled checks outside the
source tree with no Python environment or system compiler. Report:
`runs/coupled_archive_acceptance/`. Metadata: `dist/release-0.4.0.json`.

## Version 0.5.0 archive verification

The 321,816,414-byte final archive was SHA-256 verified, extracted into a fresh
`/tmp/nexus-v05-release-check/`, and passed all nine food checks without a Python
environment or system compiler. Report: `runs/food_archive_acceptance/`. Release
metadata: `dist/release-0.5.0.json`.
