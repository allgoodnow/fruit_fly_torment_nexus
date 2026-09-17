# Recovery after light input

The offline molecular model retains suppression after a photon response and
recovers during darkness without a reset, an added refractory timer, or gain
retuning. This assay characterizes that behavior; it does not establish a
quantitative match to recorded photoreceptors.

[Song et al. (2012)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3420010/) describe
how calcium-dependent negative feedback suppresses subsequent photon responses
in an activated microvillus. Different units recover at different times, while
unactivated units remain available. This motivates checking both a repeatedly
stimulated single unit and a complete receptor population. A response-area
recovery curve is not the same quantity as a refractory-period distribution.

## Controlled second-pulse assay

The first pulse arrives at 20 ms. A second pulse follows after a chosen gap.
At the second pulse's arrival, the complete conditioned model is copied into
two branches: one receives the pulse, the other continues in darkness. Both
preserve molecular states, scheduled events, membrane state and random-generator
state. A third, previously unstimulated model ages in darkness until the same
absolute time and then receives the same pulse.

All three branches run for another 300 ms. The incremental response is the
stimulated branch's integrated open-channel count minus that of its no-pulse
control. This removes the first response's residual tail in the ensemble mean.
The recovery ratio divides the mean incremental response by the mean response
of the rested controls.

Negative differences and nonresponding trials are retained. Individual
differences are not classified as quantum bumps: the extra input can alter
subsequent random paths, even when both branches started with the same RNG
state. Ratios are computed from group means, not averaged trial ratios.
Channel·ms measures integrated channel count, not ionic charge.

## Single microvillus

Each pulse supplies one absorbed photon to the same microvillus. There are 128
seeds per gap (79100–79227), or 896 trial sets and 2,688 post-probe branch runs.

| Gap between photons | Incremental response / rested response | Bootstrap 95% interval |
| --- | ---: | ---: |
| 20 ms | 2.8% | 0.8–5.2% |
| 50 ms | 0.6% | −0.4–2.2% |
| 100 ms | 1.6% | 0.0–3.8% |
| 200 ms | 16.4% | 12.4–20.6% |
| 300 ms | 60.9% | 55.3–66.9% |
| 500 ms | 91.2% | 87.1–95.8% |
| 1,000 ms | 99.5% | 93.9–105.9% |

At 20 ms the no-probe branch still produces a mean 28.735 channel·ms; calling
all post-probe activity a second response would strongly overstate recovery.
At 50 ms, calcium-bound calmodulin averages 131.40 and the available G-protein
pool averages 42.73, out of 50. By 1,000 ms these become 1.05 and 49.73. These
recorded states demonstrate persistent molecular history, rather than an
imposed on/off recovery schedule.

These are pointwise, seed-bootstrap intervals with 2,000 resamples, not
biological uncertainty bounds or simultaneous confidence bands. Values above
100% or below zero are possible and are not clipped. The assay has no fitted
physiological target or acceptance threshold.

## Full receptor population

The population assay uses 30,000 explicit microvilli and 30,000 absorbed photons
per pulse. Photons are allocated uniformly by the model; no smaller population
is multiplied to produce a full-sized signal. Six seeds are run for each of
three gaps (50, 300 and 1,000 ms), giving 18 trial sets and 54 post-probe branches.

| Gap between flashes | Incremental response / rested response | Bootstrap 95% interval |
| --- | ---: | ---: |
| 50 ms | 37.5% | 37.3–37.7% |
| 300 ms | 75.1% | 74.7–75.5% |
| 1,000 ms | 99.8% | 99.5–100.2% |

The single-unit recovery ratio cannot be applied as a global gate to the whole
receptor. Some units were never hit by the first pulse and can contribute to
the next response. With this allocation rule the expected fraction unhit is
`(1 - 1/30000)^30000`, approximately 36.8%. This is an allocation expectation,
not a predicted response amplitude or a measured biological fraction.

The six-seed population check is limited. It exercises real population state,
photon allocation and membrane integration, but does not validate the complete
eye, arbitrary light sequences, physiological voltage amplitudes or real-time
performance. The molecular cascade still uses its fixed-voltage calcium
calculation, as described in the [model documentation](phototransduction.md).

## Reproduction

```sh
python scripts/verify_phototransduction_recovery.py \
  --output-dir runs/new-recovery-single --repeats 128

python scripts/verify_phototransduction_recovery.py \
  --output-dir runs/new-recovery-population --microvilli 30000 \
  --photons-per-pulse 30000 --repeats 6 --gaps-ms 50 300 1000
```

Use new output directories. Raw channel and voltage traces remain in those
ignored directories. Reports contain all per-trial areas, pre-probe molecular
state means, seeds, configuration, model hashes and summary statistics:

- [Single-unit report](../experiments/phototransduction-recovery-single-v1-results.json)
- [Population report](../experiments/phototransduction-recovery-population-v1-results.json)

The live application is unchanged. Optical calibration, histamine release and
integration with the connectome remain separate work.
