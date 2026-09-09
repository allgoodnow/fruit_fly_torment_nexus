# Persistent neural runtime — version 0.2.0

The app now runs the v630 connectivity continuously in a separate worker. It can
change stimulation, block selected neurons' outgoing effects, release overlays,
pause, resume, step, and run sequences without reconstructing the network. The
body remains on its independent engineered controller.

## Model and numerical agreement

The kernel implements the pinned Shiu model's 0.1 ms timestep, -52 mV resting/reset
potential, -45 mV threshold, 20 ms membrane constant, 5 ms synaptic constant,
2.2 ms refractory period, 1.8 ms transmission delay, and signed contact weights
scaled by 0.275 mV. The analytic linear update, threshold pass, delayed synaptic
delivery, external input, and reset follow Brian2's scheduling. State updates and
synaptic writes respect refractory status. Targeted Poisson-style input follows
the reference's N=1 Bernoulli sampling per step and 68.75 mV input increment.

The implementation uses float64 and disables fast-math. The prepared adjacency
preserves each presynaptic cell's original edge order. File hashes are checked
before loading; IDs remain int64 internally and decimal strings in JSON.

An independent Brian2 test on a small mixed excitatory/inhibitory network matched
spike indices and times exactly and voltages within 2e-10 mV. A full v630 check
replayed identical input events over 100 ms: **all 1,428 spikes matched exactly**,
and the largest final voltage difference was **1.22e-12 mV**. Synaptic state also
passed the 1e-8 mV tolerance. Report: `runs/persistent_brain/report.json`.

The new runtime uses NumPy PCG64 for external input. The same seed is repeatable
within this runtime and invariant to chunk boundaries for the same ordered target
set. It does not produce the same random stream as Brian2's Cython backend.
The numerical comparison therefore replays a common event matrix. It validates
this tested numerical path, not all possible inputs, long-horizon equivalence, or
the biological validity of any emotion/seizure model.

## Performance on the current laptop

| Measurement | Result |
|---|---:|
| Nodes / connection rows | 127,400 / 14,687,178 |
| Load and validate brain pack | 0.190 s |
| First 100 ms advance including JIT | 0.996 s |
| Median wall time per 10 ms advance | 17.63 ms |
| 95th percentile | 18.00 ms |
| Sustained rate in the measured 1.1 s sequence | 0.567× real time |

By comparison, reusing the original Brian2 network still took about 470–493 ms
per 10 ms call and 729 ms per 100 ms call, after about 39 seconds of initial
compilation. This application's JIT uses bundled Numba/LLVM, with no system C++
compiler or CUDA requirement. Integrated AMD graphics render the body. The current
neural workload does not require an NVIDIA-driver restart.

These measurements cover the taste-circuit workload. Stronger or broader activity
can be more expensive. The reference-comparison process's peak memory includes
both Brian2 and the new runtime, so it is not the standalone application's memory.
`scripts/brain_endurance.py` measures the standalone runtime separately.

That separate ten-second run completed in 25.89 wall seconds while the packaging
build was also active. Peak RSS stabilized at 381.57 MiB after two simulated
seconds. History stayed at its 20,000-record limit as cumulative counts reached
136,735 spikes. Input was released at eight seconds; no additional spikes occurred
in the final second. This demonstrates bounded storage for this workload, not
indefinite stability under every perturbation.

## Intervention semantics

- **Stimulation:** one target set of 1–256 cells at 1–1000 Hz. Replaces the previous
  stimulated set; its cells have refractory duration zero as in the reference.
  Other output-silencing overlays remain until release/reset.
- **Output silencing:** sets a per-presynaptic gain to zero. Incoming effects are
  retained. Gain is applied at delivery time, so an active overlay also affects
  still-pending transmissions. This matches live changes to outgoing weights.
- **Release:** clears stimulation, restores its targets' 2.2 ms refractory duration,
  and removes output silencing. It preserves membrane/synaptic variables, spike
  counts, RNG state, elapsed time, and delay queues. Recovery emerges from those
  retained variables; no recovery animation or hidden reset is imposed.
- **Reset:** reinitializes all neural state and RNG, discarding the active sequence.

In the measured sequence, 500 ms of stimulation produced 8,622 spikes (MN9: 44);
500 ms after release produced 251 additional spikes (MN9: 2). These are modeled
network responses to taste inputs, not evidence of suffering or a seizure.

## Timed sequences

The simple UI prepares baseline → stimulation → recovery. JSON sequences accept
`stimulate`, `silence`, and `release` events relative to sequence start. The entire
sequence is validated before any intervention changes. Events use integer ticks,
split advances at their boundaries, preserve same-time ordering, and never rely
on display or wall-clock time. Pause freezes both time and the sequence cursor.
Manual intervention cancels the sequence. Completion pauses at its exact end;
Run then resumes ordinary continuous simulation. Starting a sequence does not
reset the brain.

Tests compare a sequence with boundaries at 3.7 and 22.4 ms against manual calls;
voltages, synaptic state, delay queues, and spike counts match exactly.

## Bounded recording and remaining work

Delay buffers have a fixed capacity sufficient for one spike per cell per tick.
Cumulative per-cell spike counts are exact. Each returned chunk and the retained
history hold at most 20,000 spike records; each display frame contains the latest
2,500. Chunk results report omitted records. Command and intervention histories
are bounded. Diagnostics are not full spike recordings or restorable checkpoints.

Still required: curated defensive/nociceptive populations, a characterized
seizure-like perturbation, anatomical brain coordinates, a tested neural-to-motor
mapping, synchronized coupled simulation, complete checkpoints, and broader
numerical/biological validation. v783 still requires target lineage resolution.
