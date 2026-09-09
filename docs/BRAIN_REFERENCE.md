# Initial reference-model integration

The research harness calls the unmodified `run_trial` in
https://github.com/philshiu/Drosophila_brain_model at commit
`91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`.
It uses the included v630 tables, all 21 sugar inputs from `example.ipynb`,
and its MN9 readout (`720575940660219265`). The neural integration timestep is
0.1 ms; each trial lasts one simulated second. Seeds are 73100, 73101, and 73102.
No source equations or connectivity weights were changed.

## Reproduce these integration checks

```sh
git clone https://github.com/philshiu/Drosophila_brain_model vendor/brain-reference
git -C vendor/brain-reference checkout 91bdd1e7dcf193f3e7ca5a8933497fcef63b7960
.venv/bin/python scripts/inspect_brain_data.py
.venv/bin/python scripts/brain_assay.py --trials 3
.venv/bin/python scripts/brain_assay.py --trials 3 --rates 200 --resume
.venv/bin/python scripts/compare_brain_assay.py
```

The compiled reference backend needs a C++ compiler in development (GCC on this
machine). This dependency is not present in the released body prototype and must
be removed from the eventual user's brain-runtime setup. `runs/brain_assay_630`
contains spike indices/times as NPZ, lossless neuron IDs as NPY, and a JSON report
with hashes, timing, memory and trial rates. The default sample has three trials
per stimulated condition, compared with 30 in the saved upstream examples.

## Discovered reference-condition mismatch

The current source defaults to 150 Hz Poisson input. However, the saved upstream
`sugarR.parquet` contains an average **196.75 Hz** in the 21 driven neurons across
its 30 one-second trials. The saved `sugarR_100Hz.parquet` contains **98.98 Hz**.
Thus the first file appears to represent a roughly 200 Hz input condition,
not the current 150 Hz default. This is an inference from the recorded spikes;
the parquet result does not itself establish the generating parameter settings.

Do not compare the new 150 Hz output to that saved file as a same-condition
reproduction. The added 200 Hz trials investigate the discrepancy. A source-code
history/parameter audit and an adequately repeated comparison are still required.

## Observed results

| Condition | Trials | Mean driven-cell rate (Hz) | MN9 mean ± SD (Hz) |
|---|---:|---:|---:|
| New baseline | 1 | 0 | 0 |
| New 100 Hz | 3 | 98.14 | 62.33 ± 2.87 |
| Saved sugarR_100Hz | 30 | 98.98 | 67.03 ± 6.60 |
| New 150 Hz | 3 | 148.40 | 81.00 ± 2.83 |
| New 200 Hz | 3 | 195.76 | 91.67 ± 3.40 |
| Saved sugarR (rate unspecified) | 30 | 196.75 | 93.27 ± 3.15 |

SD is the population standard deviation of the per-trial spike rates, matching
the upstream analysis convention. These initial samples show a similar response
range at 100/200 Hz, but are not an equivalence test. The 200 Hz input-output
comparison supports the inferred archived input-rate mismatch. Preserve the
uncertainty until the generating parameters are established.

Each stimulated one-second reference trial took about 44–46 wall seconds, including
network construction/code-generation work. Peak process RSS reached about 4.12 GiB.
This is not a benchmark of a reused persistent network; it does show why a direct
copy of the trial runner into a desktop button handler would be unresponsive.

## Snapshot coverage

Both tables have unique neuron IDs, valid index ranges, exact ID/index agreement,
nonnegative contact counts, and signed weights matching their source columns.
Counts and SHA-256 hashes are in `data/manifests/brain-reference.json`.

The sugar-input ID `720575940620900446` from the v630 example is absent in v783.
The assay refuses to silently omit it or substitute another cell. MN9 is present
in both snapshots. Resolve the absent cell through documented lineage before
making the v783 comparison. A shared or changed ID alone does not establish
identical morphology or circuitry between snapshots.

## Scientific scope

This assay concerns modeled taste processing. It provides no validation of fear,
nociception, subjective suffering, or seizure dynamics. The network is not yet
connected to the walking body. Upstream silencing zeros outgoing synaptic weights;
it does not remove all incoming connections as the README broadly suggests.
The reference creates a new trial network each time, so continuous interventions,
release semantics, bounded traces, and brain-to-body mapping remain separate work.
