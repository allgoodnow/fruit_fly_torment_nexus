# Receptor voltage into the connectome

The offline replay now connects the selected
[video-driven molecular receptors](video-phototransduction.md) to the actual
MaleCNS graph: receptor voltage → declared release curve → delayed synaptic
conductance → downstream voltage and spikes. The graph's edge weights and
connections are unchanged. This is a connectivity experiment with an explicit
release assumption, **not a calibrated photoreceptor synapse model**.

## What the biology supports

Drosophila R1–R6 photoreceptors use graded voltage and tonic histamine release;
their light-dependent depolarization does not need a synthetic spike train.
Terminal calcium and other presynaptic mechanisms influence release
([Astorga et al., 2012](https://doi.org/10.1371/journal.pone.0044182)). This
supports the transmitter identity and a voltage-dependent release interface.
It does not establish the numerical curve used here.

The microvillar calcium state and the membrane model's bulk calcium state are
not treated as measurements of synaptic terminal calcium. Vesicle pools,
terminal calcium dynamics, release noise, adaptation at the terminal and
feedback into the receptor are still absent. Replaying a receptor voltage
clamps that cell and excludes incoming synaptic feedback; it is not a closed
loop physiological simulation.

## An explicit release experiment

The runner requires `--release-curve`; it has no hidden default conversion.
The example `experiments/receptor-release-assay-curve.json` declares unfitted
piecewise linear knots:

| Voltage (mV) | Equivalent release events/s |
| --- | --- |
| −80 | 0 |
| −70 | 5 |
| −60 | 20 |
| −40 | 40 |
| 20 | 40 |

These numbers are assay assumptions, not experimental measurements. “Equivalent
events” means multiplication in the connectome's existing weight units; it is
not a receptor firing rate, vesicle count or histamine concentration. The
tonic component at −70 mV is also an assumption. Changing this curve can
change downstream responses substantially. Values outside its declared voltage
range are rejected rather than silently extrapolated.

The receptor's supplied end-of-interval voltage is copied at every 0.1 ms step.
These sources bypass the LIF integrator, threshold, reset, input pulses and
incoming feedback. Their spike counts remain zero. The −70 mV reference is an
explicit display reference, not a claim that the published membrane initial
condition is an equilibrated dark state. Telemetry uses that reference for
these cells so that an initial −70 mV receptor is not falsely shown as
hyperpolarized relative to the rest of the LIF network's −52 mV baseline.

Release is sampled at the end of each complete millisecond. A packet is
`release_equivalent_hz × 0.001`, follows the existing outgoing edges and uses
the pack's 1.8 ms delay. Delivery occurs after integration; the target's
voltage first reflects that packet in the following integration step. Packets
are excluded from spike counts, rasters and population spike graphs.

Negative edges increase inhibitory conductance using the existing −70 mV
reversal, and positive edges increase excitatory conductance using the existing
0 mV reversal. Output blockade and negative-edge gain apply at delivery, just
as for the other synapses. Those conductance scales, reversals, delays and the
existing relay release law remain unfitted model extensions.

## Running and interpreting the comparison

First generate a video response and a matched darkness response with the
[video runner](video-phototransduction.md), using the same clip, duration,
receptor IDs and seed. For darkness set both absorbed-photon rates to zero.
Then run:

```sh
python scripts/run_receptor_brain_replay.py \
  --video-response video-response-folder \
  --dark-response darkness-response-folder \
  --release-curve experiments/receptor-release-assay-curve.json \
  --output-dir new-replay-folder
```

The experiment compares video and darkness twice: receptor output enabled,
then receptor output blocked. Both blocked conditions keep their supplied
receptor voltages. If blocking eliminates their downstream difference, the
effect depended on the receptors' outgoing synapses. This isolates propagation
through the model; it does not validate the chosen release curve or establish
perception or feelings.

`response.npz` stores selected 0.1 ms voltage traces and full-network voltage
and cumulative spike counts sampled every 50 ms, plus the final endpoint.
Trace cells are selected by connection strength before their responses are
computed: up to eight sources, eight lamina cells, eight medulla cells and
eight subsequent targets. The PNG shows up to four traces per stage. Other
brain differences are measured at the listed snapshots, not continuously.
`report.json` records the curve, IDs, controls, hashes, timings and limits.

The published compact result is
`experiments/receptor-brain-replay-v1-results.json`. Its input is the existing
500 ms quadrant exposure with eight explicit receptors. The remaining eye
cells receive no external light input and retain their existing LIF proxy;
only the selected sources become nonspiking voltage clamps. Other neurons start at model rest;
tonic output arises from the declared receptor and relay laws, not a validated
whole-brain spontaneous baseline. Initial transients are shared by the paired
conditions.

In that run, all 21 direct targets had a sampled video-versus-dark voltage
difference of at least 0.01 mV; the largest was 1.14 mV. Beyond those targets,
24,001 cells exceeded 0.5 mV at one of the sampled times, and 807 cells had
different final spike counts. Blocking receptor output removed all downstream
voltage and spike-count differences, despite preserving the receptor voltages.
Both conditions were identical before the receptor input began to differ.
Sources and graded relays emitted zero spikes in all four conditions.

The broad spread can include recurrent amplification and sensitivity to spike
timing, threshold crossings and voltage resets. It is not a measured visual
receptive-field map or evidence that this many biological cells would respond
to the clip. The unfitted tonic relay law also generates substantial shared
activity in darkness. Paired controls isolate dependence on receptor output;
they do not validate that baseline or the size of the response.

The normal-use check covers delayed propagation through a small synthetic
relay chain, output blockade, exact source voltages, zero source/relay spikes,
chunk consistency and reset. The full-connectome experiment retains the
actual pack weights. The native GUI now has an optional
[prepared receptor playback mode](user-guide.md#prepared-receptor-playback).
It supplies stored voltages to the running brain and body, with exact pause
and recording-end semantics. The ordinary visual mappings still use their
earlier input. Whole-eye scaling, an evidence-based terminal release model,
live molecular integration and feedback into receptors remain unfinished.

The focused native check uses that same 500 ms recording in the real app, with
rendered articulated physics and OpenGL anatomy. It pauses and releases input
mid-recording, resumes, stops exactly at 500 ms, compares full-brain final
voltages and spike counts against the offline replay, then resets the models.
`experiments/receptor-playback-gui-v1-results.json` records the checks and hashes.
The two new normal-use checks cover ordinary controls and a short timed
sequence on a small synthetic graph; the other three focused checks retain
replay propagation and legacy video/shared-clock behavior.
