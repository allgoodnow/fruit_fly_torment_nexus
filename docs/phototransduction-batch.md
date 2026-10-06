# Independent receptor batches

`nexus.brain.phototransduction_batch.BatchedPhototransduction` accepts different
absorbed-photon inputs for several receptors. Each retains its own explicit
microvilli, molecular state, event streams, photon-allocation generator and
membrane state. This extends the [single-receptor GPU cascade](phototransduction-cuda.md).
The equations and scientific limitations of that component still apply.

## Operation

Input has shape `(milliseconds, receptors)`. Output channel counts and voltages
have shape `(milliseconds * 10, receptors)`. Supply one seed per receptor;
each seed reproduces the corresponding `ParallelPhototransduction` stream.
Repeated seeds deliberately reproduce the same randomness. Receptors do not
share channels or receive scaled copies of another receptor's response.

The GPU advances up to `batch_cells` molecular populations together, then
reduces channels separately for each receptor. Twenty-millisecond internal
chunks bound scratch arrays without changing continuous event times. Shards
return their candidate states to the host before the next shard runs. Photon
assignment remains on CPU. Public calls retain all molecular states on the
host; an advance additionally copies those states for its candidate update.
State and time commit after membrane integration completes.

The membrane backend is explicit: `"cpu"` by default, or `"cuda"` for the
[existing GPU membrane](photoreceptor-cuda.md). Small membrane populations can
be faster on CPU even when molecular reactions benefit from CUDA. Both choices
integrate the same BG1 channel-to-voltage equations. The molecular cascade
requires physical CUDA in either case.

```python
import numpy as np
from nexus.brain.phototransduction_batch import BatchedPhototransduction

# Each receptor contains 30,000 explicit microvilli by default.
receptors = BatchedPhototransduction([87200, 87201, 87202], batch_cells=2)
photons = np.zeros((300, 3), dtype=np.int64)
photons[50, 1] = 3000
photons[50:150, 2] = 30
result = receptors.advance(photons)
# Receptor 0 stays dark. Receptors 1 and 2 receive distinct light protocols.
# result['open_channels'] and result['voltage_mv']: shape (3000, 3)
```

## Memory and practical scope

Eight full receptors retain 24.96 MB of molecular host state. Their largest
20 ms molecular working batch has approximately 95.05 MB of device array
payload, including state, photon allocations, per-unit channel samples, event
counts, status and reduced channels. This is a calculated payload, not a
measurement of peak GPU memory; driver, compiler and allocator caches add
overhead. Returned traces and membrane integration have additional storage.

Device batching bounds active molecular work. It does **not** bound the complete
host population or total computation when a single `BatchedPhototransduction`
owns every receptor. The 3,155 receptors with video coordinates would require
9.84 GB of persistent molecular host arrays, plus a similar candidate copy and
other storage. There are 3,377 R1–R6 cells in the eye cohorts; 222 lack those
coordinates.

The [video preparation driver](video-phototransduction.md) now bounds host
molecular memory as well: it constructs at most eight independent receptors,
completes their entire clip into disk arrays, and discards their states before
constructing the next group. It preserves each cell's seed and full population.
This is valid for the current offline model without receptor feedback. It
reduces resident state, not total molecular computation or trace storage.

## Focused verification

The normal-use check gives three receptors darkness, a pulse and steady input.
It compares separate CPU reference receptors with two shard sizes and split
versus continuous advancement, using both membrane choices. Channels, random
streams and counts agree exactly; CUDA membrane voltages use the established
1e-7 mV tolerance.

`scripts/benchmark_phototransduction_batch.py` runs eight complete receptors for
300 ms: two each in darkness, with a 3,000-photon pulse, with a 30,000-photon
pulse, and with 30 photons/ms during 50–150 ms. It compares all channels and final
states against separate per-unit-stream CPU reference receptors. Timings
include complete advance calls and exclude construction, stream initialization
and compilation. Three repetitions compare molecular shard sizes 1, 4 and 8,
and CPU versus GPU membrane integration at size 8.

On the RTX 4060 Laptop GPU, eight-receptor molecular batches with CPU membrane
integration take a median **2.60 s**, compared with **3.01 s** for eight separate
single-receptor GPU calls: approximately **1.16× faster** for this workload.
The GPU membrane option takes 2.74 s at this small population. CPU is therefore
the membrane default here; the GPU option supports larger future populations.
All 12 batch comparisons preserve channel traces and random-stream states
exactly. The largest voltage difference with the CUDA membrane is 1.43e-14 mV.

The measured timings and source hashes are in
`experiments/phototransduction-batch-v1-results.json`. The independent CUDA
comparison is the previous single-receptor GPU implementation, not the older
optimized CPU scheduler. The matching CPU reference is a correctness reference
whose slower scheduler should not be used to exaggerate speed improvements.

```sh
python scripts/benchmark_phototransduction_batch.py --output-dir new-results-folder
```

This is an offline component. A [video exposure adapter](video-phototransduction.md)
now supplies explicit intensity-to-photon assumptions for selected receptors.
Calibrated optics, transmitter release and live brain integration remain
separate work. Attribution and GPL terms are in
`licenses/photoreceptor/NOTICE.md`.
