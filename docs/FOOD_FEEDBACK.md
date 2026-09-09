# Contact-driven taste feedback — version 0.5.0

The native arena now contains a visible food patch. Actual foot/ground contacts
inside its footprint enable the reference sugar sensory input. Moving away,
removing the patch, or disabling feedback clears that input without resetting
the neural model. The existing DNa02 motor decoder and shared clock remain.

## Scientific basis and explicit approximation

[Shiu et al., Nature 2024](https://doi.org/10.1038/s41586-024-07763-9) models the
pathway from sugar-sensing neurons to feeding-related neural outputs, including
MN9. We reuse the exact 21-neuron sugar cohort from the pinned reference example
and retain the existing MN9 readout.

The new peripheral mapping is an **authored pooled contact proxy**: any supported
foot contact with the patch activates that whole reference cohort at the chosen
model rate. It is not a reconstructed foot-by-foot gustatory circuit. A 200 Hz
setting is computational input, not a sugar concentration or biological dose.
There is no distance-based smell or attraction signal. The patch is a flat visual
ground annotation, with no added obstacle, collision geometry or friction change.

The sample scene does not yet make the fly seek food, stop, extend its proboscis,
consume food, or become satiated. MN9 is a measured neural readout here. The tested
food response did not alter the recorded body trajectory through DNa02; those
behavioral links require additional circuitry or explicitly characterized decoders.

## Geometry and timing

Food defaults to centre `(5, 0)` mm, radius `2.5` mm. The visual disk and contact
test use the same centre and radius. The body adapter selects actual MuJoCo
contacts between `ground_plane` and the six distal `tarsus5` geometries, with
non-positive contact distance and a valid constraint address. Neither thorax
proximity nor an elapsed-time script can activate food input.

Contacts are sampled at shared brain/body boundaries, at most 10 ms apart (with
shorter intervals at protocol boundaries). A sample affects the next neural
interval. Detection therefore has up to one coupling interval of discretization;
it is not continuous collision-event timing. Cumulative exposure counts simulated
time for which environmental input was enabled, not wall time.

Moving or removing food while paused immediately updates pending sensory input
and the rendered patch. It does not advance either clock or erase spikes, voltages,
synaptic state, delay queues or motor-filter state. Reset preserves food settings,
clears model/counter state, and samples the reset body's contacts again. Food under
the reset fly can therefore supply pending input at time zero.

## Manual and environmental inputs

The runtime now has separate manual and environmental channels. Their union is
passed to the unchanged neural kernel. If both target the same neuron, the higher
rate wins; a cell does not receive duplicate pulses from overlapping channels.
The input list preserves manual ordering when there is no sensory channel, keeping
the prior independent-mode random sequence and reference comparison intact.

**Release manual interventions** clears the direct input and restores silenced
outputs. Food input continues if physical contact and feedback remain enabled.
Removing food clears only the sensory channel. Output silencing still prevents
the selected cells' downstream synaptic effects and motor-decoder contribution.
Both channel IDs and effective rates are exported. Changing the set of active
input columns changes subsequent random draws; comparisons across different
sensory conditions are not claimed to share identical per-neuron noise.

Timed sequences continue to schedule manual stimulation/silencing/release. Their
baseline and recovery periods may include ongoing food input. Disable **Taste
feedback** for isolated manual-circuit assays. World and neural journals are
bounded and retain explicit simulation timestamps.

## Native controls

The Body tab has **Food present**, **Taste feedback**, model input rate, patch
radius, **Place ahead**, **Place under fly**, and **Reset + run food demo**.
The last button resets both models, restores the standard scene and baseline
walking drive, and starts a crossing with no manual stimulation. `--food-demo`
does the same on launch. Contact/input status remains visible beneath the fly
even while the Brain tab is selected.

## Validation

`scripts/probe_food.py` runs the full neural model and physical body for 1.2
simulated seconds per condition, starting from the same reset state and seed.

| Condition | Contact observed | Input exposure (s) | Total spikes | MN9 spikes |
|---|---|---:|---:|---:|
| Food feedback enabled | Yes | 0.51 | 9,210 | 53 |
| Feedback disabled, food retained | Yes | 0 | 0 | 0 |
| Food removed | No | 0 | 0 | 0 |
| Sugar outputs silenced | Yes | 0.51 | 2,131 | 0 |

All conditions kept matching clocks and a minimum sampled upright cosine above
0.990. The fly crossed and left the patch without a scripted input schedule;
the sensory target list was empty by trial end. Each condition took about 10.1
wall seconds on this host. These are integration assays with one seed, not a
biological validation or a feeding-behavior model. Raw reports are in
`runs/food_probe/`.

Tests cover channel overlap, source-specific release, rejected settings, state
preservation and contact feedback. Native acceptance additionally checks the
rendered patch, food-driven activity, paused placement/removal, simultaneous manual
and sensory inputs, feedback ablation, and reset.

All 19 automated tests passed. The frozen native app passed nine food checks
outside the source tree with no Python environment or system compiler. Steering
and independent-mode regression runs retained their 10 and 13 passing checks.
Packaging includes the existing reference graph and positions; no network access
or additional data download is required for the food environment.
