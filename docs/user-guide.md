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
150 simulation milliseconds; black points identify stimulated targets. In the
development build, **Spikes + voltage** also displays orange cells above the
model's resting voltage and yellow cells below it. **Spikes only** restores the
previous view. These are measured states from the simulation, not region animations
or measured calcium signals. Changes smaller than 0.5 mV are omitted; color and
point size saturate at a fixed 5 mV deviation from −52 mV. Display saturation never
clips or changes the simulated voltage. The legend states what each color means.
Hover over the counts for unlocated cells. Pause freezes both channels; release
preserves the current state, and reset clears it.

The graph
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

The current `male-cns-visual-receptor-lif-v3` policy assigns inhibitory signs to
**19,154 scanned connections**, representing **312,624 synaptic contacts**. It
preserves the previous 6,560 L1/L2 corrections and adds 12,594 connections. The
166,700 neurons and 25,582,938 connections are unchanged. Each rule requires
both exact annotated cell types and a histamine consensus label on the source cell.

| Sending cells | Receiving cells | Restored connections | Evidence |
| --- | --- | ---: | --- |
| R1–R6 | L1, L2 | 6,560 | [Ort receptor](https://doi.org/10.1074/jbc.M207133200) |
| R1–R6 | L3 | 3,076 | [HCLA expression in L1–L3](https://doi.org/10.1523/JNEUROSCI.1654-08.2008) |
| R7p, R7y | Dm8a, Dm8b | 5,239 | [Ort-dependent Dm8 inhibition](https://doi.org/10.1016/j.cub.2021.01.105) |
| R7p, R7y | Tm5a, Tm5b | 1,139 | [Ort-expressing Tm5 types](https://doi.org/10.1016/j.neuron.2008.08.010) |
| R8p, R8y | L1, Tm5c, Tm9, Tm20 | 3,140 | [R8 receptor segregation](https://doi.org/10.1038/s41586-023-06681-6) |

These are cell-type inferences, not receptor measurements at individual synapses
in this specimen. Dm8a/b are connectomic subdivisions of Dm8; their correspondence
with molecular yellow/pale subtypes remains uncertain. The rules do not equate
those subtypes or enforce yellow/pale pairing. Tm5 correspondence and these mapping
limits follow the [visual parts-list study](https://doi.org/10.1038/s41586-024-07981-1).

R8 can release both histamine and acetylcholine. This policy models the dominant
inhibitory component at the selected Ort targets; it omits the reported minor
cholinergic component and does not enable the separate excitatory AMA pathway.
It does not assign a global sign to R8. Ambiguous or dorsal R7/R8 annotations,
T1, and unlisted histamine pathways remain excluded. Manifest coverage records
the default zero sign and the inhibitory exceptions separately.

The magnitude still uses the inherited 0.275 mV per contact approximation, with
the same generic LIF dynamics and delay. Photoreceptors normally use graded
signaling; spike-triggered inhibitory currents here do not reconstruct that
transmission, chloride reversal potentials, or calibrated receptor kinetics.
The sign correction alone does not supply light input or spontaneous exploration.
The optional brightness input below now drives mapped R1–R6 cells.
The approaching-threat preset still directly drives LPLC2/LC4.
The experimental reduced-inhibition control scales these negative weights too.

Run `python scripts/install_receptor_signs.py` with the original structural pack
and official annotations/transmitter files in their `data/` locations. The installer
checks provenance, graph ordering, and the previous weight policy; validates a
complete candidate; then atomically selects an immutable weight file. It preserves
the original weights and a `manifest-before-histamine-<checksum>.json` alongside
them. Close the app before restoring that saved file as `manifest.json` to roll
back. Already running sessions retain their loaded weights; restart to use an
updated pack. It accepts either the original policy or the L1/L2-only policy.
Repeated installation is rejected. Freshly prepared packs apply the current rules
automatically when matching edges exist. `install_histamine_signs.py` is retained
only for reproducing the earlier L1/L2-only update.

Run `python scripts/probe_receptor_signs.py --output-dir new-results-folder` to compare
the previous and updated weights with identical input draws. It includes a direct
photoreceptor assay for each added rule, output blockade, and comparisons across FEAR, PAIN,
SEIZURE, and BOILING. Voltage changes test the implemented inhibitory connection;
they do not establish biological accuracy or any subjective experience.

`experiments/receptor-sign-v1-results.json` records 60 full-network trials: three
seeds with input, original weights, and output blockade for each added rule, plus
paired runs of the four presets. Each tested visual pathway now hyperpolarizes
its targets, and blockade removes that effect. Per-neuron spike counts in the
four preset comparisons remain identical to the prior model. The update also
passes 160 automated tests and 31 native GUI checks. It improves the visual
connection model; it does not yet change how the default fly explores.

The earlier L1/L2-only assay in `experiments/histamine-v1-results.json` contains 33 full-network
trials across three seeds. The selected photoreceptor now hyperpolarizes its targets;
presynaptic output blockade reproduces the original zero-weight response. All four
preset comparisons retain identical per-neuron spike counts in these trials. The
native app also passes 31 checks, including the activity graph and neural walking
controls. These results describe this scoped sign change, not a validation of vision.

### Visual input (development toward 1.2)

The **FLY’S VISION** panel sits beneath the brain, beside the activity graph.
The enlarged preview has its controls in a column on the left.
It is available with the updated MaleCNS pack. These controls are source-development
features; the 1.1.1 download does not include them.

1. Choose **Load video…** and select a local MP4 (H.264 recommended). Loading pauses
   the simulation and previews the first decoded frame, with visual input disabled.
2. Enable **Feed to brain**, then **Run / Resume**. Both playback and visual input
   follow the shared simulation clock. Slow simulation means slow playback.
3. **Pause** holds the frame and current input. Disabling **Feed to brain** removes
   that input and freezes playback, even if the body keeps running.
4. At the end of the clip, visual input turns off automatically; the last frame
   remains visible. **Restart** returns to frame zero and pauses, with input off.
   It preserves the brain's state and elapsed time. Enable input and resume again.
5. **Use eyes** unloads the video. Enable input to sample the two cameras attached
   to the fly's head; the preview shows the left and right views together.

For videos, the mapping selector offers **Brightness** and **Spatial (exp.)**.
Brightness remains the default. Spatial mode requires the visual-column pack
update described below. Changing mode pauses the simulation and disables visual
input; enable **Feed to brain** and resume afterward. Restart retains the selected
video mapping. Switching to physical eye cameras or resetting restores Brightness.

**Adapt to light** is an optional experimental response setting, off by default.
It works with both video mappings and the physical eye cameras. Changing it pauses
and releases visual input; enable **Feed to brain** and resume to apply it.

**Release stimulation** disables visual input. Starting a prepared sequence also
disables it, and visual input cannot be enabled during that sequence. Release the
sequence before using vision again. **Reset body + brain** unloads the video and
clears visual input. Failed playback releases visual input and reports an error.
The red **VISION** label means a nonzero visual input is applied, including while
paused; it does not label what the clip depicts. A black frame supplies zero input.

Frames are sampled every 50 ms of simulated time. The local decoder produces
RGB frames at 20 fps, at most 320 × 180 pixels, without audio. The displayed video
frame is the frame used for input. In Brightness mode its average RGB intensity
drives both eyes equally. Camera input averages each eye separately, excluding pixels
outside FlyGym's retinal mask. Camera previews are reduced for display; the input
uses the original masked frames. Camera rendering adds processing cost only when
eye feedback is enabled. The camera implementation comes from
[FlyGym's eye renderer](https://neuromechfly.org/api_reference/flygym/simulation/).

The official MaleCNS annotations identify **1,112 left and 2,265 right R1–R6 cells**
within the current pack. Selection requires the R1–R6 type, sensory classification,
histamine consensus, and an unambiguous `rootSide`. This preserves the scan's
unequal coverage. We do not invent missing cells or balance the counts. The
brightness-to-input mapping is an **unfitted linear 0–100 Hz proxy**. Overlapping
manual and visual inputs use the maximum rate per cell, rather than adding rates.

Pooled brightness does not encode image position. Neither mode models fly spectral
sensitivity, motion detection, or object recognition. R7/R8 color pathways receive no new direct light input.
Photoreceptors still use the generic spiking approximation. The downstream model
lacks chloride reversal potentials: a full-field white test drove some targets
to approximately −113 to −121 mV. Those uncalibrated voltages demonstrate a model
limitation, not a physiological prediction. More realistic retinal and synaptic
dynamics are needed before interpreting scene-guided behavior biologically.
This update does not establish vision-driven exploration or subjective experience.

The earlier spikes-only display hid most of this response. Only **13 of the 3,377
R1–R6 inputs** have recorded cell anchors, and inhibition of downstream cells need
not produce spikes. The voltage overlay now exposes that subthreshold response:
the white-frame assay shows up to **732 positioned inhibitory targets** changing
voltage, versus zero when photoreceptor output is blocked. Across nine trials,
the previous spike counts and voltage minima remain identical. The display adds
no excitation and fabricates no locations. The full target set contains 2,405
cells, of which 738 have usable anchors. Missing anatomy and pooled brightness
still limit how much spatial structure this view can show.
The display checks are recorded in `experiments/vision-display-v2-results.json`.

#### Experimental spatial video input

This mode gives different input rates to photoreceptors according to local video
brightness, rather than averaging the entire frame. The projection is a testable
approximation, not a calibrated map of the fly's optical field of view.

R1–R6 cells have no direct `assignedOlHex1/2` annotations in our source table.
We infer a cell's column only when its same-eye L1 and L2 connections each have a
unique strongest annotated column and both choose the same one. At least ten
eligible synaptic contacts are required, with at least 90% supporting that column.
Counts come from the original structural graph, independent of the current runtime
weights. L1 and L2 carry official column annotations; the connection-based extension
to R1–R6 is our inference. See the
[MaleCNS column-assignment methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC12636603/).

This selects **1,071 left and 2,084 right R1–R6 cells**. They cover 290 left and 493
right columns, out of 876 and 892 columns present in the eligible L1/L2 annotations.
The remaining **222 photoreceptors receive no spatial-video drive**; they remain
in the network and can receive other inputs. No missing cells or connections are
invented. Source records and per-cell supporting contact counts are recorded in
`experiments/visual-column-evidence-v1.json`.

The projection draws each eye's hexagonal chart in the video plane using
`x = sqrt(3)/2 × (hex1 − hex2)` and `y = −(hex1 + hex2)/2`. It centers that chart
and scales its longest extent to a unit square, using the full annotated L1/L2
grid, including columns with no selected photoreceptors. Each cell samples the
video at its inferred column through bilinear interpolation. Both eyes receive
the same video plane through their own charts; there is no stereo projection.
Image intensity still maps linearly to 0–100 Hz. The axes, orientation, image aspect
mapping, and field of view have not been registered to the specimen's optics.
Physical eye cameras therefore continue using pooled brightness.

This changes the **input**, not the scanned connection graph, synaptic weights,
voltage display, or motor decoding. Equal-mean images can now activate different
photoreceptor groups and produce different downstream voltage patterns. It does
not yet supply an ON/OFF circuit, realistic graded photoreceptor transmission,
direction-selective motion processing, object recognition, or autonomous navigation.
The previously documented uncalibrated inhibitory voltages remain a limitation.

Close the app and run `python scripts/install_visual_columns.py` after the eye-input
installer. It checks annotation, structural graph, and runtime ordering before
validating and selecting a new registry. Weights and anatomy are unchanged.
Freshly prepared packs include this registry automatically.

`python scripts/probe_spatial_vision.py --output-dir new-results-folder` compares
equal-mean left-bright and right-bright clips in both input modes, with and without
photoreceptor output, across three seeds. It checks neural differences, dark and
excluded cells, shared body/brain timing, and input release at the end of playback.
This test checks implemented spatial routing, not biological accuracy. Its results
are stored in `experiments/spatial-vision-v1-results.json`.

#### Experimental light adaptation

The static input modes give the same brightness the same rate regardless of prior
illumination. **Adapt to light** adds memory of recent brightness: a sustained
bright patch progressively reduces its own input gain, while exposure to darkness
restores sensitivity. Each inferred spatial input has its own state; pooled mode
has one state per eye. A dark channel receives zero visual drive even while its
adaptation state recovers. The displayed video remains the original input frame.

Adaptation to background illumination is supported by intracellular recordings
of Drosophila photoreceptors in
[Juusola and Hardie (2001)](https://doi.org/10.1085/jgp.117.1.3).
Our implementation is a small authored approximation, not a reproduction or fit
of that study's data or phototransduction mechanisms.

For normalized brightness `L`, the background state `A` follows
`dA/dt = (L − A) / 250 ms`. Input is `100 × L / (1 + 3 × A)` Hz. Both the 250 ms
time constant and strength 3 are **unfitted choices**. The initial background is
zero. Constant white therefore starts at 100 Hz and approaches 25 Hz; black
immediately supplies zero input. Darkness must actually be supplied while feedback
is enabled for sensitivity to recover. Disabling input represents an experiment
pause, not a simulated dark scene.

The background integrates the previously sampled light analytically over elapsed
exposure time. Rates update on the existing 50 ms visual sampling clock and are
held between updates. Integer exposure ticks keep this independent of GUI refresh
and simulation chunk sizes. Pause holds both input and adaptation; disabling
feedback or releasing stimulation clears input and freezes adaptation even if
the body continues. Loading/restarting a video, changing the mapping or adaptation
setting, and switching sources clear the adaptation state. Restart retains its
checkbox selection and the brain state. **Reset body + brain** also turns the
checkbox off. A prepared sequence disables visual input as before.

This can reduce sustained drive and its downstream inhibition. It does not impose
a physiological voltage floor, change the current-based synapse model, or prevent
excessive inhibition during initial bright input. It also does not add a dark-onset
burst, calibrated contrast tuning, ON/OFF pathways, graded photoreceptor release,
motion recognition, or biological vision-guided locomotion. Those limits remain
separate from this input adaptation model.

`python scripts/probe_light_adaptation.py --output-dir new-results-folder` runs
a white/dark/white clip with static input, adaptation, and photoreceptor-output
blockade across three seeds in the full body/brain simulation. The results in
`experiments/light-adaptation-v1-results.json` test sustained response reduction,
dark recovery, downstream sensitivity, shared timing, and release at playback end.

Different demonstrations can visualize different signals and representations.
For example, [Flyvis](https://turagalab.github.io/flyvis/) implements a trained
connectome-constrained visual network; our generic whole-CNS LIF model is not that
trained network. A broad visual glow alone cannot establish that the underlying
signal processing matches either model.

With the app closed, run `python scripts/install_eye_inputs.py` to add the eye
registry to an existing pack. It checks the original annotation and transmitter
hashes against the structural manifest, validates a candidate pack, and selects
the new registry without altering connections or weights. Restart the app after
updating. Freshly prepared packs include these cohorts automatically.

`python scripts/probe_vision.py --output-dir new-results-folder` tests a black/white
MP4 in the full body/brain simulation, with matched photoreceptor-output blockade
and input-off controls across three seeds, then tests actual eye-camera input.
The versioned results are in `experiments/vision-v1-results.json`. Blockade preserves
input spike draws while removing the measured downstream inhibition. This checks
signal routing, not biological fidelity or scene understanding.

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
