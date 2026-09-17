# Isolating phototransduction scheduler differences

The earlier roughly 80% response-area gap does not justify reducing the Python
model's gain. In this follow-up, refining the upstream time rounding brings its
channel response close to the unrounded result. A second issue is a reaction
rate computed before photon arrival, which remains stale for the first reaction.
The existing Python model already uses unrounded waits and recalculates rates
after photons arrive. No runtime parameters were changed on this evidence.

## Method

`scripts/ablate_phototransduction_timing.py` generates diagnostic copies of the
[pinned source](https://github.com/JuusolaLab/Microsaccadic_Sampling_Paper/tree/a4453f7e47abf2ea2a924c376d4c1c157c6011e2/BiophysicalPhotoreceptorModel).
Every copy guards duplicate inputs and checks delivery logs. Each receives
exactly one photon at 20 ms, runs for 300 ms, and uses seeds 77100–77131 in
GNU Octave 10.3.0. The 320 trials cover ten variants, run in three batches.

Additional observation records each post-reaction open-channel count and its
execution time. Integrating these states avoids dependence on the source's
1 ms output filling. States superseded at the same timestamp contribute zero
area. Recorded full state/time outputs match the prior unobserved guarded runs
exactly for seeds 77100 and 77121. Unit tests separately check repeated
timestamps, fractional intervals and the final interval.

Channel·ms is integrated channel count, not ionic charge. The source still
executes each reaction before its waiting time; this investigation does not
replace that rule. Equal starting seeds do not imply identical reaction paths
after changing a scheduler rule.

## Time rounding dominates the initial area difference

| Source time rounding | Responses | Mean event-integrated channel·ms | Transitions without advancing time |
| --- | ---: | ---: | ---: |
| Original 0.1 ms | 32/32 | 26.222 | 7,494 |
| 0.01 ms | 32/32 | 48.951 | 1,697 |
| 0.001 ms | 32/32 | 49.489 | 196 |
| Unrounded | 32/32 | 49.492 | 0 |

All other rules stay fixed in this comparison. At the original resolution,
many successive reactions share a timestamp. The channel response changes
substantially when that resolution is refined; 0.001 ms and unrounded means
differ by less than 0.01% in this sample. This is numerical convergence evidence
for the tested condition, not physiological validation or proof of convergence
for arbitrary illumination.

The unrounded source's mean on the common 1 ms sampling grid is 49.813 channel·ms.
The earlier Python result on that grid is 47.281, about 5.1% lower. No equivalence
margin or statistical equivalence test was defined, so these are descriptive
comparisons, not a claim that the simulators match.

## Input ordering interacts with rhodopsin protection

The source calculates `z(3,1) = kap_G*y(8)*y(4)` before adding the photon to
active rhodopsin, `y(4)`. It calculates the rhodopsin removal rate afterward.
Consequently, the first reaction can see a removal rate for the new rhodopsin
but no corresponding G-protein activation rate. The source's special protection
against the first rhodopsin removal masks part of this asymmetry.

The `fresh_input_rate` variant recomputes only the G-protein activation rate
after photon delivery. It changes no reaction equation or parameter.

| Changes relative to guarded source | Responses | Mean event-integrated channel·ms |
| --- | ---: | ---: |
| None | 32/32 | 26.222 |
| Remove waiting-rate offset | 29/32 | 23.641 |
| Remove random-draw floor | 32/32 | 26.600 |
| Remove rhodopsin protection | 19/32 | 14.913 |
| Remove protection; refresh activation rate after input | 30/32 | 24.553 |
| Remove rounding, offset, floor and protection | 15/32 | 23.589 |
| All four removals; refresh activation rate after input | 32/32 | 50.082 |

Refreshing the rate restores many responses lost when protection is removed.
The interaction means individual effects cannot simply be added together.
The combined fresh-rate variant's common-grid mean is 49.719, versus Python's
47.281, with 32/32 versus 31/32 responses. Reaction/wait ordering, calcium refresh
timing, output sampling and random streams still differ. Paired photons,
adaptation and complete receptor populations have not been tested by this assay.

The [paper](https://elifesciences.org/articles/26117) describes quantum-bump
waveform, latency and refractoriness as relevant to photoreceptor output.
Matching these recorded biological responses remains a separate requirement;
agreement between two numerical implementations cannot establish it.

## Results and reproduction

All per-trial summaries, source hashes and generated-variant hashes are in
[`phototransduction-timing-ablation-v1-results.json`](../experiments/phototransduction-timing-ablation-v1-results.json).
Raw state arrays and generated source copies remain in ignored results folders.
The published report combines the three disjoint batches; the default command
runs all ten variants in one invocation:

```sh
python scripts/ablate_phototransduction_timing.py \
  --source-dir /path/to/Microsaccadic_Sampling_Paper/BiophysicalPhotoreceptorModel \
  --octave /path/to/octave-cli \
  --output-dir runs/new-timing-ablation --repeats 32
```

A relocated Octave installation may need `OCTAVE_HOME` set to its prefix. The
output directory must be new. `--variants` accepts a subset of names listed by
`--help`; two independent Octave processes run at a time. The live application
does not import this script or the offline molecular model.
