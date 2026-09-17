# Exact-preserving molecular scheduler optimization

The offline phototransduction model now skips groups of microvilli when none
has a reaction due, maintains the total open-channel count incrementally, and
validates each event without allocating temporary boolean arrays. These are
computational changes. Reaction equations, timing rules, parameters, photon
allocation, calcium updates and population size are unchanged.

## Scheduling invariants

Each block of 128 cells stores a lower bound on its next event time. A block
can be skipped only when this bound is later than the current sample boundary.
An active block is still processed in ascending cell order, with each cell's
due events processed in their original order. A global time-sorted queue would
change which seeded random draws reach which cell; this optimization avoids
that change.

Incoming photons can move an event earlier or later. Updating the block bound
with the minimum of its old value and the new event time cannot hide an earlier
event. An unnecessarily early bound merely triggers a scan, which refreshes
the exact block minimum. The index is rebuilt from pending events on every
`advance()`, so reset, rollback and chunking require no additional persistent
state. The last block may contain fewer than 128 cells.

The summed channel count starts from the actual molecular states and changes
only by the channel difference across each reaction. Integer counts are exact
at these population sizes. No channels or reactions are omitted, and the block
size is not a biological grouping or population multiplier.

## Direct comparison

The benchmark loads the pre-optimization module from commit `c9b8372`, verifying
its SHA-256 before importing it. Both versions use the unchanged membrane
component. It compares 30,000-microvillus receptors across three seeds, five
inputs and two chunk sizes: one 300 ms advance, or thirty 10 ms advances.

After **every chunk**, all of the following must be exactly equal:

- Open-channel and membrane-voltage traces.
- Every molecular state and microvillus reversal potential.
- Pending event times and reaction identities.
- Complete random-generator state.
- Membrane state, clocks, photon totals and event counters.

All 30 comparisons passed. Tests also compare block scheduling with an
independent exhaustive traversal at exact boundaries, stale bounds, empty
schedules and population sizes 1, 127, 128, 129 and 257. Invalid-state rejection
and atomic failure remain checked.

## Measured time

These are median wall times over three seeds for **300 simulated ms in one
receptor**, using a single 300 ms call. JIT compilation and construction are
excluded; copying state and membrane integration within `advance()` are included.

| Input | Previous | Optimized | Speedup |
| --- | ---: | ---: | ---: |
| Darkness | 133 ms | 44 ms | 3.02× |
| 3,000-photon pulse at 50 ms | 458 ms | 338 ms | 1.35× |
| 30,000-photon pulse at 50 ms | 2,401 ms | 1,776 ms | 1.35× |
| 30 photons/ms during 50–150 ms | 449 ms | 343 ms | 1.31× |
| 3,000-photon pulses at 50 and 150 ms | 730 ms | 556 ms | 1.31× |

With 10 ms chunks, the measured gains are 2.48× in darkness and 1.31–1.36× for
stimulated cases. Bright input still requires millions of molecular events;
the optimization reduces overhead without removing those events. These local
measurements do **not** establish real-time performance for all 3,377 receptors,
GPU performance or a speedup in the live GUI, which does not import this model.

[The benchmark report](../experiments/phototransduction-performance-v1-results.json)
records every timing, seed, event count, implementation hash and comparison.
Timings depend on hardware and system load. Exactness is established for the
tested inputs, not proved for every possible state or input.

## Reproduction

From the repository root, save the pinned baseline into an ignored directory:

```sh
mkdir -p runs/performance-reference
git show c9b8372:src/nexus/brain/phototransduction.py > runs/performance-reference/phototransduction.py
python scripts/benchmark_phototransduction.py \
  --baseline-file runs/performance-reference/phototransduction.py \
  --output-dir runs/new-performance-comparison
```

The output directory must be new. The baseline retains the same GPL attribution
as the current component. Raw output traces and copied baseline code stay in
ignored results folders.
