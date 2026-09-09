# Neural steering — version 0.4.0

The default native session runs the v630 neural model and articulated body in
one worker with a shared clock. Simulated DNa02 spikes change the left/right
inputs of the official FlyGym walking controller. This is a first motor bridge;
the ventral nerve cord is not simulated. Version 0.5.0 adds the contact-driven
taste proxy described in `FOOD_FEEDBACK.md`; broader sensory feedback is unfinished.

## Evidence and target identity

The published study [Yang et al., Cell 2024](https://doi.org/10.1016/j.cell.2024.08.033)
links DNa02 activity to ipsilateral stride attenuation and turning. It motivates
the direction of this decoder, not its numerical gains or a complete locomotor model.

The exact targets below are in [FlyWire annotations v1.1.0](https://github.com/flyconnectome/flywire_annotations/tree/v1.1.0),
whose root IDs explicitly refer to snapshot v630. Both occur in our neural pack.
They have `hemibrain_type=DNa02`, `super_class=descending`, and the indicated
`side`; their separate `cell_type` fields are empty. The preparation script checks
these fields and writes `data/brain-v630/motor-registry.json`, including provenance,
the source-table hash, and decoder parameters.

| Side | v630 root ID |
|---|---|
| Left | 720575940629327659 |
| Right | 720575940604737708 |

Other candidate locomotor labels in the annotation table are ambiguous or absent.
This release uses only the pair above and makes no DNg100, DNp09, or backward-walking
claim. IDs from newer snapshots or other specimens are not substituted.

## Explicit decoder approximation

For each side, count actual spikes over the coupling interval. Apply outgoing
silencing, then low-pass filter the delivered rate with a 100 ms time constant:

`r_next = exp(-dt/0.1) * r + (1-exp(-dt/0.1)) * spike_count * output_gain / dt`

`side_drive = baseline * (1 - 0.7 * clip(r_next / 200 Hz, 0, 1))`

The baseline defaults to 1 and is an authored walking input. The frequency scale,
attenuation ceiling and filter constant are engineering choices. The mean of the
two drives and half their difference become FlyGym's drive and turn inputs.
The controller supplies leg rhythms, kinematics and contact corrections. No body
animation is selected from an intervention's name, and no subjective-experience
claim follows from the resulting movement.

Disabling the bridge sets both side drives to baseline without modifying the
neural network. Silencing DNa02 prevents its future outgoing spikes from reaching
the decoder, while prior filtered influence decays. Release preserves the current
filter and neural state. Reset clears both models, filter state and elapsed time.

## Timing

Brain and body use 0.1 ms steps. Coupling normally aggregates up to 10 ms of
neural activity before advancing the corresponding physical interval. Sequence
events and endpoints split intervals, preserving exact application times. The
worker checks agreement of the two clocks after each interval; display packets
carry both states from that same boundary. This discretized decoder averages
within each interval; it is not a reconstruction of phase-specific VNC dynamics.

Run, pause, single-step and reset from either tab affect both models. Timed sequence
completion pauses both. The display remains separate from the simulation worker.
Independent legacy mode remains available with `--independent`.

## Initial causal measurements

`scripts/probe_steering.py` runs the full 127,400-neuron network and actual physics,
with a fixed seed and 0.6 simulated seconds per trial. These are software integration
measurements, not experimental-animal validation or population statistics.

| Condition | Heading change (radians) | Total neural spikes |
|---|---:|---:|
| No input | +0.1208 | 0 |
| Left DNa02, 200 Hz | +1.2132 | 229 |
| Right DNa02, 200 Hz | −0.9258 | 246 |
| Left input, decoder blocked | +0.1208 | 229 |
| Left input, target outputs silenced | +0.1208 | 142 |
| Left input, 5 ms coupling | +1.2091 | 229 |

Heading is measured from the thorax rotation and unwrapped. Baseline gait itself
has some drift. The blocked and silenced trials exactly matched the baseline
recorded position and heading, despite continued target firing. Halving the
coupling interval changed the left-turn result by about 0.0041 rad in this assay.
All trials retained an upright-axis cosine above 0.990 and matching clocks.
Each 0.6 s trial took about 5.0–5.1 wall seconds: approximately 0.12× real time
on this laptop. Raw trajectories and timings: `runs/steering_probe/report.json`.

A subsequent 3-second sequence stimulated left DNa02, released it, stimulated
right DNa02, then released it. It completed in 25.18 wall seconds with matching
clocks, minimum sampled upright cosine 0.9918, and final motor-drive difference
below 0.006 after 500 ms of recovery. Source and report:
`scripts/steering_endurance.py`, `runs/steering_endurance/report.json`.

All 16 automated tests passed. The frozen app passed 10 coupled native acceptance
checks outside the source directory with the Python environment removed and no
system compiler. The independent native mode retained its 13 passing checks.

## Remaining work

Add supported sensory feedback and neural control of locomotor initiation/stop;
expand circuit curation and model characterization; test longer and more varied
closed-loop behavior; improve performance; then add full checkpoints. The requested
defensive, nociceptive and seizure-like experiments are not implemented by this bridge.
