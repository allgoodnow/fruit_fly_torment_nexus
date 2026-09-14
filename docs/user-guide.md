# Fruit Fly Torment Nexus

## Stimulation display

The red **STIMULATION** display shows the inputs currently applied to the model.
It follows acknowledged simulation state, not the last button clicked or a future
step in a prepared sequence. Labels can appear together, for example **FEAR + PAIN**.
The clock beside it uses simulation time.

| Label | Applied model input |
| --- | --- |
| FEAR | A virtual approaching threat or constant input to a mapped looming pathway is active. |
| PAIN | The nociception sensory candidates or the older central aversion candidates are being stimulated. |
| SEIZURE | Inhibitory connection strength is reduced. The standard preset also supplies neural input. |
| BOILING | The nominal 100°C heat scenario is applied, including mapped thermal nociception when available. The boiling preset also reduces inhibition; BOILING covers that combined preset. |
| HEAT | Heat input below the nominal boiling scenario, without reduced inhibition; high-temperature settings also recruit mapped nociceptive sensory cells. |
| CUSTOM | Manually selected neurons are stimulated, outside a reduced-inhibition scenario. |
| SILENCING | Selected neurons' outgoing effects are blocked. |
| NONE | No monitored inputs or network overlays are currently applied. |

These are **scenario names**, not measurements of feelings. Subjective fear and
pain, biological seizures, and thermal injury have not been established in this
simulation. Reduced inhibition and the nominal temperature are available in
exported diagnostics for inspecting combined inputs in detail.

When paused, the labels show the inputs held in the model and the clock says
**PAUSED**. No simulation time passes. During a sequence's baseline and after
release, the display returns to NONE if no other input remains. Continuing spikes
or movement do not keep an old stimulation label on screen.

## Run an experiment

The **Experiments** tab contains the fear, pain, seizure, heat, and boiling controls.
Set the baseline, stimulus, and after-release durations, then select a run button.
Durations are simulation milliseconds; a slow computer can take longer in real time.
The baseline releases existing manual interventions before the next input begins.

- **Pause / Resume** stops or continues advancement while preserving model state.
- **Release stimulation** clears manual inputs, silencing, and reduced inhibition.
  It preserves voltages, accumulated spikes, and delayed neural events. Activity
  and movement can therefore continue afterward.
- **Reposition body only** restores an upright posture at the same ground location
  and pauses. It preserves neural state, inputs, decoder state, simulation time,
  and the pending sequence. The body walking controller is reinitialized; this is
  manual posture assistance, not spontaneous biological recovery. The action is
  always in the Body tab and appears in Experiments when the fly is overturned.
- **Reset body + brain** reinitializes both models and their clocks.
- **Enable motor response** controls the retreat/escape/disruption decoder. Neural activity
  continues when that decoder is disabled.
- **Resume free behavior after experiment** allows normal ground behavior to
  continue at the sequence endpoint, when free ground behavior is also enabled.

The heat value is nominal. Warmth-cell input saturates at 40°C; selecting 100°C does
not simulate tissue boiling, thermal damage, or short-circuits. The boiling preset
adds an authored reduction of inhibition to the heat inputs.

### Approaching threats (development toward 1.2)

The source build's **Run fear** button now supplies one virtual approach over the
selected stimulus duration. The released 1.1.1 binary uses constant LPLC2 input.
The new input follows a constant-speed approach whose full apparent angle grows
from 5° to 120°. A shorter duration gives faster angular expansion. This is a
neural input scenario; there is no rendered predator or retinal image processing.

The updated MaleCNS registry contains 185 LPLC2 and 126 LC4 cells. The source audit
finds 311 direct connections from these cohorts to the two giant fibers, totaling
11,224 synaptic contacts. Research supports LPLC2 size encoding and LC4 angular
velocity encoding in this pathway. [Ache et al., 2019](https://doi.org/10.1016/j.cub.2019.01.079)
The specimen-specific audit is in `experiments/looming-evidence-v1.json`.

The simulator's LPLC2 input uses a Gaussian size envelope centered at 45° with
20° width and 200 Hz peak. LC4 input is 0.2 times expansion speed in degrees/second,
capped at 200 Hz. These values and angular endpoints are **unfitted engineering
choices**, not the paper's fitted giant-fiber voltage model or measured sensory
firing rates. Both cohorts receive uniform input; receptive-field position,
retinal adaptation, and learned threat responses are not reconstructed.

The profile is evaluated every 0.1 ms of neural time. Pause holds it, its endpoint
removes the approach input, and Release clears it immediately. Neither action
resets the neural state. Other active inputs survive the approach's automatic
endpoint. Explicit constant looming commands replace the approach. Overlapping
input sources use the highest rate per cell, without duplicate independent pulses.

Diagnostics include `looming_input` with geometry, instantaneous rates, and LC4
availability. Recorded traces include the same endpoint snapshot for each interval.
Older packs without LC4 continue with the LPLC2 size channel and report LC4 as
unavailable. No identifiers from another specimen are substituted. Flight remains
disabled; the existing motor decoder reads resulting neural activity.

To update a local MaleCNS pack from 1.1.1, place its exact official annotation file
in `data/raw/male-cns-v1.0/` and run `python scripts/install_looming_circuit.py`.
The script checks the source hash, verifies both visual pathways reach the mapped
giant fibers, and switches the pack to a content-addressed registry. It preserves
the previous registry, neural weights, and nociception cohort. Newly prepared
packs include the LC4 mapping automatically.

Saved protocols can use `{"at_ms": 100, "action": "loom", "duration_ms": 500}`.
The approach must fit inside the protocol; a later Release can interrupt it.
Legacy `circuit` commands keep their constant-rate behavior. New `loom` protocols
require this source build or a later release, and cannot run in 1.1.1.

`experiments/looming-v1-results.json` records 15 full brain/body trials: three seeds
with normal approach, LPLC2 output blocked, LC4 output blocked, both blocked, and
the motor bridge disabled. Blocking either cohort changes downstream activity
and the body trajectory. Blocking both removes the escape decoder signal during
input in all three seeds. Disabling the motor bridge preserves exact neural
counts while changing movement. These checks establish causal effects inside
this model, not biological fidelity.

The blockade is removed with the input at 600 ms. Delayed spikes and retained
neural state can then produce a short response. The report separates the input
period from the 200 ms after release; whole-trial escape counts include both.

## Body, brain, and camera

Drag either 3D view to orbit and scroll to zoom. The body camera follows the fly.
Use **Reset camera**, **Home**, or **Fit all** to restore the corresponding view.

The body uses NeuroMechFly/FlyGym with MuJoCo physics. Its walking rhythms and
explore/turn/rest behavior are authored controllers. Measured neural responses can
interrupt that behavior. DNa02 activity changes left/right walking drive; giant-fiber
activity and distributed network activity drive authored escape/disruption effects.
Mapped MDN descending-neuron activity can request backward stepping.
This is not a reconstructed mapping from every VNC neuron to every leg muscle.
Flight is not enabled.

### Neural walking drive (development toward 1.2)

The Body tab's **Neural walking drive (experimental)** option replaces the
scheduled explore/turn/rest drive with a readout of two mapped BDN2 neurons
(MaleCNS DNg100, IDs 10045 and 10056). DNa02 still influences steering and MDN
activity can request retreat. The default remains the existing ground controller.

Experiments associate BDN2 with forward walking initiation and speed. The official
MaleCNS annotations explicitly identify both DNg100 cells as BDN2.
[Study](https://doi.org/10.1038/s41586-024-07854-7)
The mapping audit is in `experiments/walking-evidence-v1.json`.

This option is a **motor-readout experiment, not spontaneous exploration**.
The current LIF model has no background excitation, so without input its quiet
neurons remain quiet. No random drive is injected to hide that limitation.
Body-to-brain sensory feedback and suitable ongoing neural dynamics are still
needed for an exploration loop driven by the network.

The decoder filters delivered BDN2 spikes over 100 ms and averages both cells.
100 Hz mean activity supplies unit forward drive; the walking-drive slider scales
that gain. These are engineering choices. Outgoing silencing also suppresses the
readout. Quiet forward/retreat/escape/disruption levels below 0.01 select a resting
pose. FlyGym still coordinates the legs, and the existing escape/disruption
waveforms remain authored. This does not reconstruct individual motor-neuron
activation of muscles.

The mode switch preserves the brain, input settings, and clock. The ordinary
exploration schedule is held while the mode is active. Pause holds neural drive;
Release preserves its filtered activity, which can decay as simulation continues.
Reset clears activity while retaining the selected mode. Existing packs without
the bilateral readout disable this option; there is no fallback to foreign IDs.

Run `python scripts/install_walking_readout.py` to add the readout to a local pack
using its original official annotations in `data/raw/male-cns-v1.0/`. The installer
checks the annotation hash and bilateral identity, preserves previous circuits,
and switches to a content-addressed registry. Freshly prepared packs include it.

Use `scripts/probe_neural_walking.py --output-dir new-results-folder` to compare
quiet, BDN2 stimulation, BDN2 output blockade, and motor-bridge blockade with three
seeds. This assay supplies neural input explicitly; successful walking in it does
not establish autonomous exploration.

The default neural dataset is **MaleCNS v1.0**, containing **166,700 classified
neurons** and **25,582,938 directed connections** in the prepared model. The live
view includes brain and ventral nerve cord (VNC) cell anchors. It displays 140,638
positioned cells; 26,062 lack the required position information. Points are cell
anchors, not complete neuron shapes. Red points indicate spikes during the last
150 simulation milliseconds; black points identify stimulated targets. The graph
below shows mean firing rate across all neurons in 10 ms simulation-time bins,
retaining the last five simulated seconds. Every spike contributes, including
those omitted from the bounded raw raster in diagnostics. Silent intervals remain
visible as zero activity. Pause holds the data; reset clears it. This is a population
spike rate, not an EEG or a readout of subjective feelings. The Body tab's graph
shows controller leg activity instead of neural spikes.

The neural dynamics are an experimental, unfitted leaky integrate-and-fire model.
Synaptic signs, connection scaling, delays, and decoder gains are model assumptions;
they are not a biological validation of a living fly. The 1.1.1 release omits
monoamine and histamine effects. The development policy includes the narrow
histamine exception described below; other histamine and monoamine effects remain
omitted, not biologically absent.

The current PAIN preset stimulates 24 abdominal multidendritic sensory candidates
mapped from a published MANC cohort through the official MaleCNS cross-specimen
annotations. Ambiguous, missing, and unclassified matches were excluded. This is
an experimental nociception pathway, not evidence of felt pain. The older central
aversion cohort remains available in the underlying registry and legacy dataset.

### Photoreceptor connectivity (development toward 1.2)

The `male-cns-lamina-histamine-lif-v2` policy restores negative weights on existing
histaminergic R1-R6 photoreceptor connections to L1/L2 lamina cells. These cells
use the Ort/HCLA histamine-gated chloride receptor, supporting an inhibitory sign.
[Gengs et al., 2002](https://doi.org/10.1074/jbc.M207133200)
This is a cell-type inference applied to the scan; receptors were not measured
at each synapse in this MaleCNS specimen.

The audited pack changes **6,560 weights**, representing **211,051 scanned synaptic
contacts**. Its 166,700 neurons and 25,582,938 connections are unchanged. R7/R8, T1,
L3 targets, and other histamine connections remain outside this rule. There is no
global histamine sign assignment. Manifest coverage records the default zero sign
and the number of inhibitory exceptions separately.

The magnitude still uses the inherited 0.275 mV per contact approximation, with
the same generic LIF dynamics and delay. Photoreceptors normally use graded
signaling; spike-triggered inhibitory currents here do not reconstruct that
transmission, chloride reversal potentials, or calibrated receptor kinetics.
This addition does not supply light input, working vision, or spontaneous
exploration. The approaching-threat preset still directly drives LPLC2/LC4.
The experimental reduced-inhibition control scales these negative weights too.

Run `python scripts/install_histamine_signs.py` with the original structural pack
and official annotations/transmitter files in their `data/` locations. The installer
checks provenance, graph ordering, and the original weight policy; validates a
complete candidate; then atomically selects an immutable weight file. It preserves
the original weights and a `manifest-before-histamine-<checksum>.json` alongside
them. Close the app before restoring that saved file as `manifest.json` to roll
back. Already running sessions retain their loaded weights; restart to use an
updated pack. Repeated installation is rejected. Freshly prepared packs apply
the same rule automatically when matching edges exist.

Run `python scripts/probe_histamine.py --output-dir new-results-folder` to compare
the original and updated weights with identical input draws. It includes direct
photoreceptor stimulation, output blockade, and comparisons across FEAR, PAIN,
SEIZURE, and BOILING. Voltage changes test the implemented inhibitory connection;
they do not establish biological accuracy or any subjective experience.

The recorded assay in `experiments/histamine-v1-results.json` contains 33 full-network
trials across three seeds. The selected photoreceptor now hyperpolarizes its targets;
presynaptic output blockade reproduces the original zero-weight response. All four
preset comparisons retain identical per-neuron spike counts in these trials. The
native app also passes 31 checks, including the activity graph and neural walking
controls. These results describe this scoped sign change, not a validation of vision.

### Heat input and nociception

Version 1.1 adds a separate thermal input to the 24 mapped abdominal md sensory
cells in the current MaleCNS pack. The adult-fly study reports md responses to
noxious heat at 40°C. This supports including those cells in the heat pathway.
It does not provide a calibrated spike-rate curve for this simulator.
[Study](https://doi.org/10.1101/2025.10.28.684868)

The scenario uses 40°C as an explicit gate and the existing nociception input
rate of 100 Hz at or above it. Both are approximation choices: the gate is not a
measured universal biological threshold, and input Hz is not pain intensity.
The warmth-cell input remains unchanged. Neither input grows further above
40°C; 100°C still does not model tissue damage or actual thermal conduction.
The separate reduced-inhibition overlay remains part of the BOILING preset.

Cooling or clearing temperature removes the heat-driven md input while preserving
any separately applied PAIN input. Overlapping inputs use the greater rate, never
two independent pulses per cell and tick. Direct warmth-cell stimulation replaces
the temperature scenario and clears its thermal md input. Release clears both
intervention channels without resetting neural state. Legacy packs without a mapped
md cohort keep their original warmth-only behavior.

The HEAT and BOILING labels include this thermal pathway; they do not add a second
PAIN label. Exported telemetry identifies the thermal targets and rate separately.
The motor adapter receives only the resulting neural activity.

### Pain input and movement

Version 0.16 removes the central-aversion-to-slowdown-and-fixed-turn rule from
version 0.15. A central aversion response alone no longer forces a movement.
Instead, the motor adapter reads delivered spikes from four MaleCNS neurons
annotated as MDN (two per side). These descending neurons have an experimentally
established role in backward walking. Their connections to the LBL40 leg premotor
neurons are present in the loaded connectome.

MDN activity requests backward stepping through FlyGym's signed CPG interface.
The 80 ms rate filter, 5–60 Hz transfer range, and maximum reverse command of 0.8
remain engineering choices. The leg trajectories and coordination are still
provided by the locomotion controller, not reconstructed from every leg neuron.
There is no fixed pain-triggered turn. The old FlyWire dataset has no newly mapped
MDN readout and does not use this response.

**Known model mismatch:** the published abdominal-md experiments report a rapid
increase in forward walking without consistent directional turning. Our current
neural model recruits MDNs instead, producing a retreat prediction. That is not a
validated reproduction of nociceptive behavior. The neural weights and input rate
were not adjusted to force the published outcome. A connectome alone does not
supply the missing physiological calibration or evidence of subjective pain.

The response can occur under any input that recruits MDNs and can persist after
release while neural activity remains. **Enable motor response** disables the
retreat, escape, and disruption mappings. Settling requires retreat to fall below
0.05 along with the other monitor criteria.

References: [MDN motor-circuit experiments](https://www.nature.com/articles/s41467-020-19936-x)
and [abdominal nociception preprint](https://pmc.ncbi.nlm.nih.gov/articles/PMC12636578/).

## Advanced controls

The Body tab controls free ground behavior, baseline walking drive, and the camera.
The arena has no food marker or automatic taste feedback. These features were
removed from the application in version 0.14. Historical research scripts and
recorded results may still refer to them.

The Brain tab provides custom neuron IDs, input rate, target-output silencing,
inhibition strength, motor-bridge controls, and timed sequences. The inhibition
control changes inhibitory connection strength; the standard seizure-like preset
supplies input as well, because reduced inhibition alone need not initiate activity.
These are advanced model interventions, not validated biological seizure controls.

Run, Pause, Step, and Reset normally affect the shared body/brain clock. The optional
independent mode uses separate workers and does not connect brain output to movement.

## Prepared sequences and results

Use **Brain → Load JSON…** to run a saved protocol. **Save JSON…** saves the sequence
configured by the Brain tab's controls. The `experiments` folder contains presets
for MaleCNS and a separate `flywire-v630` folder for the older model. A protocol
tagged for one dataset is rejected by the other. Continuous sequences preserve
neural state between stages; later responses can depend on earlier stimulation.

The native executable can also run a protocol without opening the GUI:

```text
fruit-fly-torment-nexus \
  --run-sequence experiments/male-cns-continuous-sequence.json \
  --output-dir results/my-run
```

This uses the same 3D physics. The output directory must be new. It contains the
protocol and model provenance, a 10 ms trace of neural/body measurements, and
per-neuron starting/final spike counts. These files are not a complete spike raster
or a resumable checkpoint. Add `--disable-motor-bridge` for a motor-link comparison;
`--seed` selects the neural random seed. Diagnostics can also be exported from the GUI.

## Release and settling

The red event log records applied commands and post-release observations. The
monitor checks neural activity, motor output, and body posture separately without
modifying the simulation. Its combined settling check requires 500 consecutive
simulation milliseconds with all of these conditions:

- Population activity at most 0.1 Hz per neuron in each observed interval.
- Retreat, escape, and disruption levels at most 0.05; steering readout at most 10 Hz.
  In neural walking mode, forward drive must also be at most 0.05.
- Upright body-axis value at least 0.8, with posture information available.

Meeting these operational thresholds is not biological recovery, subjective relief,
or zero activity. Losing any condition returns the monitor to observation. Paused
wall-clock time does not count. Some tested disruption sequences remain active and
overturned for at least ten seconds after release; the monitor does not force recovery.
If the fly is stuck, use Reposition body only, then Resume. It can overturn again
if neural disruption persists. Disable motor response to inspect activity without
its body effects, or explicitly reset both models for a fresh trial. Repositioning
is logged and restarts the posture-settling interval; it never clears neural activity.

## Release and local files

The version 1.1.1 download targets Linux x86-64. Fedora 44 is the automated test
platform; macOS application use has also been reported by the maintainer. Keep the executable
and `_internal` directory together after extracting the archive. No runtime network
connection is required. The first neural compilation can take longer than later
loads. This CPU build is not guaranteed to run at real time. Other operating systems
and Linux distribution versions are not covered by the packaged acceptance checks.

The GUI stores caches and diagnostics in the platform application-data location,
normally `~/.local/share/Nexus/Fruit Fly Torment Nexus/`. Unattended runs use the
output directory you specify and cache under `~/.cache/fruit-fly-torment-nexus/` unless
standard XDG cache settings override it. Closing the app ends the current in-memory
session; exported diagnostics and counts are not resumable checkpoints.

## Sources and attribution

Data, cohort mappings, model assumptions, and recorded checks accompany the source
repository's `data/manifests` and `experiments` folders. See `THIRD_PARTY_NOTICES.md`
and the bundled licenses for attribution and terms.

- [MaleCNS data and annotations](https://male-cns.janelia.org/download/)
- [NeuroMechFly / FlyGym](https://github.com/NeLy-EPFL/flygym)
- [Reference neural model](https://github.com/philshiu/Drosophila_brain_model)
- [Nociceptive pathway cohort and analysis](https://github.com/jesmjones/nociceptive_pathways_paper)
