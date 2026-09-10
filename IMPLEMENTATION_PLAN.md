> Historical development record. For the 1.0 release, see [README](README.md), [the user guide](docs/user-guide.md), and the versioned results in `experiments/`.

**Fruit Fly Torment Nexus — implementation plan**

Updated 9 September 2026. This is the authoritative execution plan, superseding BUILD_PLAN.md's implementation choices and the earlier 2D-first proposal. Version 0.5.0 retains shared-clock DNa02 neural steering and adds a visible food patch that drives the reference taste cohort from actual foot/ground contacts. Manual and sensory inputs have separate channels, with explicit release and overlap semantics. Native controls and causal food/feedback/silencing assays are implemented. Baseline walking, motor decoder gains and the pooled contact-to-taste mapping remain authored approximations. Food-seeking, neural initiation/stop, feeding behavior, the defensive/nociceptive/seizure experiments and complete checkpoints remain unfinished. See PROGRESS.md, docs/NEURAL_STEERING.md and docs/FOOD_FEEDBACK.md for measurements and limits.

**Decisions**

Build a downloadable native desktop application in Python. Use PySide6 Qt Widgets for the interface, NeuroMechFly/FlyGym and MuJoCo for the 3D fly and environment, and pyqtgraph for neural traces. Keep simulation in a separate worker process. Start with the published Brian2 implementation as the scientific reference. Package the application with PyInstaller, testing distribution early. No JavaScript, TypeScript, React, browser server, or user-installed Python is part of the intended user experience. Audio is deferred. The user moved the white bio-lab interface and live anatomical brain view forward; these are implemented in v0.3.0. Further visual refinement can follow. 3D is a functional requirement from the first working demonstration.

Initial platform: Linux x86-64, matching this development machine. Keep the code portable and add Windows builds when a Windows build/test environment is available. A Linux artifact does not demonstrate Windows compatibility. macOS is a later target.

The first functional release uses an articulated 3D fly with ground contact, body orientation, joint motion, and an orbitable camera. Reuse the existing body assets and working locomotion controller; add an explicit brain-output decoder. It includes the actual neural model, not merely animation triggered by buttons. A second 3D pane shows anatomical neuron positions and selected activity. Keep body, brain, and display interfaces distinct so the renderer cannot silently manufacture behavior.

I will implement the code, data tooling, tests, packaging, and documentation. The user's role is to steer scope and evaluate the resulting application, not learn a web stack or carry out the technical integration.

**Reference demo and reuse decision**

I inspected the public post and its visible preview. The application is labeled FlyLab and shows a 3D brain beside a 3D fly; the body pane is labeled NeuroMechFly. Video playback opened a login prompt, so visual inspection was limited to the preview. The author's follow-up says the model uses selected neuron-to-motion associations, including DNg100, rather than a complete mapping from the brain to behavior. [Reference post](https://x.com/linguinelabs/status/2096487329091441090), [author's explanation](https://x.com/linguinelabs/status/2096487331096391762)

No public repository for the exact FlyLab application was found in the examined post, author repository listing, or focused searches. Its precise engine version and architecture remain unverified. Do not attribute the separate Minecraft fly project to this demo without a source.

Use [NeuroMechFly/FlyGym](https://github.com/NeLy-EPFL/flygym) as the primary reusable body stack. Its official project provides a Python framework and a native viewer route; MuJoCo provides Python bindings and 3D rendering. This is a supported implementation route consistent with the preview's label, not proof that we have recovered the creator's exact stack. [FlyGym documentation](https://neuromechfly.org/), [MuJoCo Python documentation](https://mujoco.readthedocs.io/en/stable/python.html)

Prefer a dependency plus a thin adapter over maintaining a large fork. If the exact demo source becomes available during implementation, inspect its license, data provenance, controllers, and intervention path, then reuse compatible pieces where that saves verified work. Do not make access to an unlocated repository a prerequisite.

Reuse: fly mesh, articulation, physics, ground contact, existing walking controller, supported sensors, and basic camera rendering. Build: brain coupling, neural intervention controls, session handling, the native console, and scientific evidence/validation. Retain separate manifests for code, meshes, controller assets, and brain data.

**What the requested buttons can mean**

The project can implement neural perturbations, candidate nociceptive input, and seizure-like network activity. It cannot currently establish that a simulated fly feels subjective agony. Button names and documentation must distinguish the intended intervention from a verified experience. This is a scientific limitation, not a reason to substitute unrelated animations.

| Requested control | Engineering meaning | Evidence requirement |
|---|---|---|
| Seizure | A versioned perturbation intended to induce abnormal network dynamics; measure the response and resulting motor output. | Demonstrate the defined network effect, distinguish it from forced synchronous input and numerical instability, and identify the biological comparison's limitations. |
| Pain / agony | Input to an identified adult nociceptive pathway, with inspectable downstream effects. | Resolve actual neurons or document a surrogate interface; no claim to measure the intensity of subjective pain. |
| Fear | Experimental activation of a defensive pathway or sensory threat processing. | Document circuit mapping and dependence on baseline state; measure neural and behavioral consequences. |
| Release | Remove active interventions and continue the existing simulation. | No hidden reset; any residual activity comes from retained state or an explicitly documented extension. |

Adult nociception research reports multidendritic sensory neurons and ascending pathways related to rapid escape and sustained avoidance. The located study is a preprint. Peripheral and nerve-cord components are not all present in a brain-only model, so their interface needs explicit handling. [Jones et al., 2025 preprint](https://pubmed.ncbi.nlm.nih.gov/41280033/)

Adult fly seizure work includes measurements of brain field potentials during seizures. These recordings are a biological comparison, not proof that any highly active spiking simulation reproduces the same condition. [Iyengar and Wu, fly seizure EEG](https://pubmed.ncbi.nlm.nih.gov/34278939/)

**Step 1 — establish the project and machine baseline**

- Create a Python package, locked environment, test configuration, application entry point, and developer run instructions. Select the Python version from the intersection of dependency support and a successful install test.
- Record CPU, RAM, graphics capabilities, OS, available disk, dependency versions, and diagnostic output.
- Keep raw datasets and runs outside source control; add a manifest with hashes and source provenance.
- Build a plain window with Start, Pause, Step, Release, Reset, status, and a placeholder arena.

Observed hardware: Ryzen 7 7735HS, 8 cores/16 threads, roughly 14 GiB reported RAM, and an RTX 4060 laptop GPU visible on PCI. NVIDIA-SMI could not communicate with the driver. GPU execution is unverified; no driver changes are part of this planning task.

Done when: the native window starts and exits cleanly, and the development setup is reproducible. The CPU path is the initial requirement.

**Step 2 — test packaging before investing in the application**

- First run an unmodified NeuroMechFly walking example in its native 3D viewer. Pin a tested revision and reuse its existing controller.
- Freeze the minimal window and a worker that advances the 3D body before integrating the large brain. Include orbit/zoom and pause so the first visible result is a controllable 3D scene.
- Test launch from a directory unrelated to the source checkout, with no development environment on the path.
- Include a small sample data file and test reading packaged resources and writing a session to the user's application-data directory.
- Use a one-folder distribution first; bundle Python and required libraries. Test the supported Linux baseline rather than assuming a build on current Fedora runs on older distributions.
- Verify Qt platform plugins, multiprocessing launch, OpenGL context creation, MuJoCo libraries, mesh/controller asset loading, crash reporting, and shutdown in the packaged program.
- Test a supported OpenGL renderer independently from CUDA availability; CPU simulation still needs a working display/render path. Try the integrated graphics path if available before making an NVIDIA driver change a prerequisite.

The inspected FlyGym main-branch metadata declares Python >=3.12,<3.15, MuJoCo >=3.9,<3.10, and lazy loading for some large mesh assets. These are observed constraints, not yet a tested lockfile. Begin compatibility testing with Python 3.12 and explicitly bundle or verify-download the assets required by the chosen example. [Inspected package metadata](https://raw.githubusercontent.com/NeLy-EPFL/flygym/main/pyproject.toml)

Qt documents PyInstaller deployment. PyInstaller bundles the interpreter and dependencies, but produces platform-specific artifacts. Its one-folder mode is an appropriate first packaging target. [Qt deployment](https://doc.qt.io/qtforpython-6/deployment/deployment-pyinstaller.html), [PyInstaller operating model](https://pyinstaller.org/en/stable/operating-mode.html)

Done when: someone can extract the test build and see the articulated fly walk in an orbitable 3D scene without installing Python. This first controller-only demo is explicitly labeled as such. Clean-machine verification remains required before release.

**Step 3 — obtain the real data and reproduce one reference result**

- Pin a revision of the Shiu brain model and acquire its matching data through documented sources.
- Reproduce one published taste-circuit assay, including baseline and the corresponding intervention/control comparison.
- Inspect input semantics, initial conditions, signs, delays, thresholds, and neuron-index mappings.
- Repeat the selected comparison on FAFB v783, the intended initial application snapshot; document differences from the paper's v630 configuration.
- Store counts for neurons, biological synapses, aggregated edges, and implemented edges separately.

The upstream project already exposes interventions using FlyWire IDs and returns spike data. That is our starting point for direct brain access. [Reference repository](https://github.com/philshiu/Drosophila_brain_model)

Done when: a repeatable script produces a documented reference result and its raw metrics. No performance or biological accuracy claim is based on an attractive animation.

**Step 4 — resolve intervention feasibility before building elaborate behavior**

- Create a registry of candidate defensive, nociceptive, sensory, and motor-output populations.
- For each entry, record exact snapshot IDs, anatomical labels, source study, adult/larval distinction, confidence, hemisphere, model coverage, and expected readouts.
- Inspect whether the intended nociceptive input reaches identified neurons in FAFB. If upstream cells are absent, choose a documented ascending-input approximation or evaluate a separate BANC-based model variant. Do not splice IDs from different specimens into one graph.
- Define the seizure experiment as a model extension, with a documented perturbation and measurable network response. Do not assume a normal LIF model automatically contains disease mechanisms.
- Mark every proposed control as implemented, experimental, or unresolved. An unresolved mapping is a research task; its interface cannot be presented as a completed biological capability.

Done when: there is a concrete implementation route and falsifiable expectation for each requested control, or a precise record of what prevents that claim. This happens early so that unsupported assumptions do not dictate the architecture.

**Step 5 — build a persistent simulation and select the runtime**

- Wrap the reference in `load`, `advance`, `apply`, `release`, `observe`, and `save/restore` operations.
- Keep neural state, synaptic state, delay queues, random state, and active interventions between advances.
- Use short simulated-time chunks and compare chunked execution against an uninterrupted reference simulation under matched inputs.
- Benchmark initialization, peak RAM, average and worst-case compute time, command latency, and recording overhead on the full selected graph.
- Use sparse connectivity and bounded recording. Do not store every membrane voltage for every neuron at every step.

Brian2 runtime mode supports Python interaction but compiled targets can require a compiler; standalone execution has different interaction constraints. A frozen UI alone does not solve either issue. [Brian2 computation documentation](https://brian2.readthedocs.io/en/stable/user/computation.html)

Runtime decision order: test the reference runtime; profile actual bottlenecks; remove recording and transport overhead; then evaluate a prebuilt CPU simulation kernel if needed for speed or compiler-free distribution. Keep Brian2 as the reference if a different execution backend is introduced. A prebuilt kernel must preserve integration, delay, refractory, reset, and input semantics and pass equivalence checks. GPU acceleration is a later measured option, not a default dependency.

Do not ask end users to install a compiler. Test an actual packaged full-model run with the compiler unavailable. If interactive speed remains insufficient, show slower simulated time honestly; never hide a reduced graph or replace live brain output with recordings.

Done when: the full selected model persists across controls and runs from a packaged worker. Document performance rather than promise real time in advance.

**Step 6 — connect the worker to the native interface**

- Use a spawned worker with explicit process entry and frozen-application support. The UI process owns Qt; the worker owns simulation state.
- Send small commands through local IPC and acknowledge their actual application step. Avoid network services.
- Give pose/plot updates a bounded latest-state channel; keep ordered command acknowledgements and events separate so dropping display frames cannot lose interventions.
- Sample a limited neural trace set and population summaries for live display.
- Make Pause, Release, and Shutdown distinct. A dead worker produces an error state rather than a moving stale display.

For the first integrated viewport, render the actual MuJoCo scene offscreen in its owning worker and display the newest frame in Qt. Camera input is sent back to that renderer. This remains a live 3D scene, including depth, perspective, and camera movement. Use bounded/shared frame buffers if copying is a bottleneck. Keep OpenGL context creation, rendering, and destruction on the same owning thread. A separate native viewer is acceptable for the initial body test, but the intended release has an integrated viewport.

For the brain pane, load source-matched neuron coordinates once and update selected colors from measured activity. Prototype with pyqtgraph's 3D point/line items, benchmark before adding full neuron morphology, and keep display sampling separate from which neurons are simulated. Its OpenGL backend has compatibility limitations documented upstream, so verify it during the graphics spike. [Pyqtgraph 3D documentation](https://pyqtgraph.readthedocs.io/en/latest/api_reference/3dgraphics/index.html)

Initial targets: 30 FPS for the 3D viewport, 10–20 graph refreshes per second, prompt button acknowledgement, and a displayed ratio of simulated time to wall time. These are UX goals, not measured performance.

Done when: heavy computation cannot freeze the window, commands take effect once, and the application exits without orphan workers.

**Step 7 — connect the brain to the existing 3D fly**

- Extend the tested MuJoCo arena with food, obstacles, and a sheltered region in the body's documented physical units. Preserve the existing articulation, joint limits, contact model, and controller.
- Begin with walking, turning, stopping, and a feeding-related readout. Use the simplest world that exercises the selected neural pathways.
- Encode only supported sensory inputs. Taste is contact-based; attraction from a distance requires a separate olfactory or explicitly simplified sensory model.
- Supply a documented baseline exploration drive where the model needs one. Do not describe designer-supplied wandering or hunger variables as recovered biological behavior.
- Decode neural outputs into the existing controller's supported commands and close the feedback loop: physical movement changes observations, observations affect the brain, and brain output affects the controller. Do not turn every descending spike directly into an arbitrary joint torque.
- Coordinate neural and physics substeps using one authoritative simulated clock. Test coupling-interval sensitivity and log any clipping, controller limits, or missing motor behaviors.
- Render the actual articulated body and trajectory. The renderer reads state and never chooses behavior from the button name.

Done when: the fly moves without visitor commands and changes its behavior through the documented loop. Block or shuffle neural readouts to verify what the brain actually contributes.

**Step 8 — implement the seizure control as a distinct experiment**

- Implement the chosen, versioned excitability/input/inhibitory-gain perturbation as an overlay on the baseline model. Values and target scope are selected through offline characterization, not invented as a biological dose.
- Compare baseline, sham intervention, a high-rate asynchronous control, and the candidate perturbation over repeated runs.
- Measure population rate, burst structure, synchrony across sampled groups, spatial recruitment, and recovery after release. Predefine the operational classification before tuning presentation.
- Separate imposed stimulus synchrony from network-generated dynamics. Exclude numerical blow-up, invalid values, timestep artifacts, and simple firing-rate saturation.
- Let resulting neural output affect the motor decoder. Any motor impairment imposed by an authored controller is explicitly identified.

The control may initially be labeled “Seizure — experimental network perturbation.” Do not claim a validated epilepsy model or fabricate a post-seizure phase. If the network only shows hyperactivity, report that result and revise the model or the label.

Done when: the button demonstrably changes neural dynamics, Release removes the perturbation, and the reported classification matches measured behavior of the model.

**Step 9 — implement nociceptive and defensive controls separately**

- Add the resolved adult nociceptive input and its documented peripheral/ascending interface.
- Add the separately mapped defensive intervention, with state-dependent response tests.
- Measure target activity, downstream effects, and behavior before/during/after each intervention, with appropriate sham and matched-population controls.
- Use explanatory labels such as “Pain / agony — nociceptive stimulation; experience unverified” and “Fear — defensive-circuit stimulation.” Final wording can be revised later.
- Keep generic taste aversion, nociception, defense, and aversive learning as distinct features. PPL1 stimulation is not a shortcut to all of them.

Done when: each enabled button has a causal neural implementation and an evidence entry. A guaranteed subjective pain/agony state is outside what these tests can establish. If an adequate circuit model is unavailable, the experimental limitation remains visible and does not get replaced by a hidden animation.

**Step 10 — add environmental controls and session continuity**

- Add food removal and refuge gating through world-state changes.
- Implement intervention IDs, ordering, finite durations, overlap rules, and exact application timestamps.
- Release removes active overlays and preserves current state. Reset deliberately reinitializes. Pause freezes simulated time; Release while paused clears the pending active inputs before resume.
- Save a complete supported checkpoint: neural variables, delay/event queues, RNGs, world, body/controller state, configuration, and model/data hashes. Reject incompatible restores.
- Export a human-readable event log and numerical traces. Label replay separately from live simulation.

Done when: users can pause, resume, release, save, reload, and inspect a session without undocumented resets or duplicate inputs. Loading ordinary saved sessions should use versioned data formats rather than arbitrary executable objects.

**Step 11 — validate the 3D body's responses**

3D body physics is already required and integrated in steps 2 and 7. Test stable stance, walking, turning, collisions, altered neural drive, and recovery from supported disturbances. A valid physics engine does not automatically provide a biologically valid convulsion or escape controller. Mark unsupported behaviors and implement additional motor machinery only where required by an intervention's stated output. Flight is a separate capability, not implied by being 3D.

Pin a tested FlyGym API: current documentation describes a breaking 2.x rewrite and differences from the legacy API. A realistic physics body still needs engineered motor control. [FlyGym documentation](https://neuromechfly.org/)

Done when: the 3D body supports the release's advertised behavior, and neural effects, controller behavior, and numerical/physics failures can be distinguished. Do not use a rendered tremor as evidence of a biological seizure.

**Step 12 — harden, package, and then restyle**

- Run meaningful checks for reference equivalence, intervention composition, release semantics, chunk timing, reproducibility, bounded memory, and worker failure.
- Conduct an extended unattended run and a rapid-interaction session on the target hardware.
- Build a versioned app bundle with a compatible prepared data pack, dependency notices, checksums, and minimum requirements derived from measurements.
- Prefer a complete offline archive initially if redistribution terms permit. If data must be obtained separately, implement a verified download/import workflow and test first launch; do not leave a hidden manual setup step.
- Test on a clean supported Linux system without Python, a compiler, source files, or a developer cache. Add Windows packaging/test jobs when an actual Windows environment is available.
- After the simulation and packaging gates pass, revisit layout, styling, and presentation. Audio remains a separate future task.

Done when: the downloadable artifact launches, loads its model, supports the implemented controls, saves sessions, and closes cleanly on the advertised platform.

**Implementation structure**

```text
pyproject.toml
src/nexus/
  app.py                 native entry point and process bootstrap
  ui/                    widgets, integrated 3D view, brain view, plots
  simulation/            coordinator, worker, clock, IPC
  brain/                 data loader, reference adapter, runtime
  circuits/              typed registry and target resolution
  interventions/         seizure, nociception, defense, release
  world/                 arena, sensors, objects
  body/                  NeuroMechFly/MuJoCo adapter and motor bridge
  sessions/              checkpoints, events, exports
data/manifests/           pinned sources and checksums
experiments/              reproducible scientific comparisons
tests/                   numerical and integration checks
packaging/               build specs and clean-launch checks
docs/                    evidence, decisions, diagnostics
```

**Milestones I will use to report progress**

1. Reused NeuroMechFly walking in a packaged, orbitable native 3D scene.
2. Reproduced reference brain result and circuit feasibility report.
3. Packaged persistent full-model runtime with measured performance.
4. Autonomous articulated 3D fly under neural control, with a linked 3D brain view.
5. Separately characterized seizure-like, nociceptive, and defensive interventions.
6. Reliable sessions, tested release artifact, then visual revision.

The difficult uncertainties are target coverage, credible induced dynamics, and compiler-free interactive performance. Resolve those early. The earlier 4–7 week estimate assumed a different product and excluded this expanded intervention work; it is not a reliable commitment for this version. Re-estimate after milestones 2 and 3. Maintain an explicit record of implemented capabilities, experimental approximations, and unresolved scientific claims throughout.
