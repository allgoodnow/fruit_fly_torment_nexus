# Fruit Fly Torment Nexus

## Stimulation display

The red **STIMULATION** display shows the inputs currently applied to the model.
It follows acknowledged simulation state, not the last button clicked or a future
step in a prepared sequence. Labels can appear together, for example **FEAR + PAIN**.
The clock beside it uses simulation time.

| Label | Applied model input |
| --- | --- |
| FEAR | The looming-threat candidate circuit is being stimulated. |
| PAIN | The nociception sensory candidates or the older central aversion candidates are being stimulated. |
| SEIZURE | Inhibitory connection strength is reduced. The standard preset also supplies neural input. |
| BOILING | The nominal 100°C warmth scenario is applied. The boiling preset also reduces inhibition; BOILING covers that combined preset. |
| HEAT | Warmth input below the nominal boiling scenario, without reduced inhibition. |
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

- **Pause** stops advancement while preserving model state.
- **Release stimulation** clears manual inputs, silencing, and reduced inhibition.
  It preserves voltages, accumulated spikes, and delayed neural events. Activity
  and movement can therefore continue afterward.
- **Reset body + brain** reinitializes both models and their clocks.
- **Enable motor response** controls the escape/disruption decoder. Neural activity
  continues when that decoder is disabled.
- **Resume free behavior after experiment** allows normal ground behavior to
  continue at the sequence endpoint, when free ground behavior is also enabled.

The heat value is nominal. Warmth-cell input saturates at 40°C; selecting 100°C does
not simulate tissue boiling, thermal damage, or short-circuits. The boiling preset
adds an authored reduction of inhibition to the warmth input.

## Body, brain, and camera

Drag either 3D view to orbit and scroll to zoom. The body camera follows the fly.
Use **Reset camera**, **Home**, or **Fit all** to restore the corresponding view.

The body uses NeuroMechFly/FlyGym with MuJoCo physics. Its walking rhythms and
explore/turn/rest behavior are authored controllers. Measured neural responses can
interrupt that behavior. DNa02 activity changes left/right walking drive; giant-fiber
activity and distributed network activity drive authored escape/disruption effects.
This is not a reconstructed mapping from every VNC neuron to every leg muscle.
Flight is not enabled.

The default neural dataset is **MaleCNS v1.0**, containing **166,700 classified
neurons** and **25,582,938 directed connections** in the prepared model. The live
view includes brain and ventral nerve cord (VNC) cell anchors. It displays 140,638
positioned cells; 26,062 lack the required position information. Points are cell
anchors, not complete neuron shapes. Red points indicate spikes during the last
150 simulation milliseconds; black points identify stimulated targets. The raster
below shows spike times against model neuron indices. The Body tab's graph shows
controller leg activity instead of neural spikes.

The neural dynamics are an experimental, unfitted leaky integrate-and-fire model.
Synaptic signs, connection scaling, delays, and decoder gains are model assumptions;
they are not a biological validation of a living fly. Monoamine and histamine
effects are omitted by the current sign policy, not biologically absent.

The current PAIN preset stimulates 24 abdominal multidendritic sensory candidates
mapped from a published MANC cohort through the official MaleCNS cross-specimen
annotations. Ambiguous, missing, and unclassified matches were excluded. This is
an experimental nociception pathway, not evidence of felt pain. The older central
aversion cohort remains available in the underlying registry and legacy dataset.

### Why pain input may barely change walking

The current walking decoder reads DNa02 steering activity, giant-fiber escape
activity, and distributed network activation. It does not directly translate the
GNG121 central aversion candidate response into an avoidance gait. In the recorded
three-seed sensory-input assay, the central candidates produced 31–32 spikes during
stimulation, but the giant fibers produced only 0–1 spikes and the disruption
readout remained zero. The physical paired test measured a maximum trajectory
change of about 0.082 mm. The input can therefore activate the investigated pathway
while producing little visible walking change. A stronger visible avoidance
response would require a separately investigated or explicitly authored motor
mapping; increasing the label's prominence does not change that mapping.

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
fruit-fly-nexus \
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
- Escape and disruption levels at most 0.05; steering readout at most 10 Hz.
- Upright body-axis value at least 0.8, with posture information available.

Meeting these operational thresholds is not biological recovery, subjective relief,
or zero activity. Losing any condition returns the monitor to observation. Paused
wall-clock time does not count. Some tested disruption sequences remain active and
overturned for at least ten seconds after release; the monitor does not force recovery.

## Sources and attribution

Data, cohort mappings, model assumptions, and recorded checks accompany the source
repository's `data/manifests` and `experiments` folders. See `THIRD_PARTY_NOTICES.md`
and the bundled licenses for attribution and terms.

- [MaleCNS data and annotations](https://male-cns.janelia.org/download/)
- [NeuroMechFly / FlyGym](https://github.com/NeLy-EPFL/flygym)
- [Reference neural model](https://github.com/philshiu/Drosophila_brain_model)
- [Nociceptive pathway cohort and analysis](https://github.com/jesmjones/nociceptive_pathways_paper)
