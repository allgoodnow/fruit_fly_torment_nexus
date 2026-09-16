# Phototransduction timing audit

The offline Python reaction model does **not** reproduce the original cascade's
response distributions in this comparison. Matching its scalar equations was
insufficient: the event scheduler also matters. These results do not establish
which scheduler is more biologically accurate. The live application is unchanged.

## Direct source execution

The audit executes the unmodified MATLAB cascade in GNU Octave 10.3.0, using
[the pinned source revision](https://github.com/JuusolaLab/Microsaccadic_Sampling_Paper/tree/a4453f7e47abf2ea2a924c376d4c1c157c6011e2/BiophysicalPhotoreceptorModel).
It compares one microvillus per trial, with 32 seeds (77100–77131) per condition
and 300 ms per trial. Each model receives the same physical photon-arrival
times. Equal seed labels do not match random draws between Octave and NumPy.
Neither the membrane nor a complete receptor population is compared here.

Both traces are sampled at 1–299 ms in 1 ms increments. Original-grid summaries
are also retained. The source fills output rows using ceiling-based time bins;
matching sample grids does not remove that convention. Channel area is a
rectangular sum in channel·ms, not ionic charge or a physiological measurement.
Nonresponding trials remain in the area and peak averages. Latency statistics
include only responding trials.

| Absorbed photon arrival times | Source responses | Python responses | Source mean channel area | Python mean channel area |
| --- | ---: | ---: | ---: | ---: |
| None | 0/32 | 0/32 | 0 | 0 |
| 20 ms | 32/32 | 31/32 | 26.281 | 47.281 |
| 20, 40 ms | 32/32 | 32/32 | 24.844 | 50.000 |
| 20, 120 ms | 32/32 | 32/32 | 26.281 | 48.938 |

Single-photon mean area is about 80% larger in Python. Median sampled latency
is 18 ms in the source and 16 ms in Python. Excluding seed 77121 from both
single-photon groups leaves mean areas of 26.387 and 46.968 respectively, still
about 78% apart. This is a descriptive sample comparison, without a predefined
equivalence tolerance or a physiological fit.

The complete per-trial summaries, parameters and source hashes are in
[`phototransduction-timing-v1-results.json`](../experiments/phototransduction-timing-v1-results.json).
The `execution_success` flag means the audit ran successfully, not that the
models agree.

## Confirmed upstream duplicate input

For a single photon at 20 ms with seed 77121, the source activates rhodopsin
twice at that timestamp. A logging-only copy records:

```text
time_ms  photons  Rh_before  Rh_after
20       1        0          1
20       1        1          2
```

Its complete molecular-state and time arrays exactly equal the unmodified
source output. The duplicate is therefore not introduced by logging. The
source's rolling input bookkeeping can revisit an integer timestamp.

A separate diagnostic copy replaces that bookkeeping guard with a per-input
delivered flag. Four cases compare original, logging-only and guarded copies:
the failing single photon, a single-photon control (seed 77100), and both paired
schedules with seed 77121. All 12 runs complete. Logging preserves full outputs
in all cases, and the guard delivers every specified photon exactly once. The
control channel trace is unchanged; the other three traces change.

The coarse check `maximum Rh > supplied photons` only flags the single-photon
trial. Logging also confirms duplicates in the two paired seed-77121 trials;
prior rhodopsin removal can hide duplicate delivery from that maximum check.
The timing report's exclusion group is therefore **not** a set of certified
duplicate-free source trials. The main timing comparison uses original,
unguarded source outputs throughout.

This guard is a reproduction tool, not an upstream patch or a calibrated
replacement model. It changes only generated copies in the results folder.
Results are in
[`phototransduction-source-input-v1-results.json`](../experiments/phototransduction-source-input-v1-results.json).
The Python model also passes a regression check on the failing input schedules.

## Reproduction

Use the project's Python environment, GNU Octave, and a checkout at the source
revision above. The scripts verify source file hashes before execution. With
a relocated Octave environment, set `OCTAVE_HOME` to its installation prefix;
a normal system installation may not need it.

```sh
python scripts/compare_phototransduction_timing.py \
  --source-dir /path/to/Microsaccadic_Sampling_Paper/BiophysicalPhotoreceptorModel \
  --octave /path/to/octave-cli \
  --output-dir runs/new-timing-audit --repeats 32

python scripts/check_phototransduction_photon_delivery.py \
  --source-dir /path/to/Microsaccadic_Sampling_Paper/BiophysicalPhotoreceptorModel \
  --octave /path/to/octave-cli \
  --output-dir runs/new-input-audit
```

Choose new output directories. `--reuse-recordings` on the timing script can
recompute summaries from a completed matching audit. Raw state arrays, generated
source copies and logs stay in the results directories, outside version control.

The remaining numerical differences include reaction-before-wait versus
reaction-after-wait, rounded waiting times, the empirical latency offset,
first-rhodopsin protection, and calcium refresh timing. This audit does not
isolate their individual effects. No gain or rate was retuned to conceal the
mismatch, and no live sensory pathway was replaced on this evidence.
