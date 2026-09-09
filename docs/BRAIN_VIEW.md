# Anatomical brain view — version 0.3.0

The white native interface keeps the articulated fly and brain visible together.
Ordinary labels are black, both workers' session events are red, and the floor is
light grey. These visual changes do not change the physics or neural equations.

## Data provenance

Coordinates come from the official [FlyWire annotations v1.1.0 release](https://github.com/flyconnectome/flywire_annotations/tree/v1.1.0),
commit `df6bb136f5b3d91c3992df4e8de2642329e2a384`,
`supplemental_files/Supplemental_file1_annotations.tsv`. Its column documentation
explicitly identifies `root_id` as the FlyWire v630 ID. Modern releases use other
snapshots and must not be joined by assumption.

The preparation script matches integer root IDs to the exact neural pack order.
There are no duplicate matched IDs or non-finite matched anchors. It maps 127,322
of 127,400 neurons, including all 21 sugar targets and MN9. The remaining 78 have
no displayed location and remain in the simulation. Activity among these cells
is counted separately in the brain caption.

`pos_x/y/z` are cell anchors, typically on the backbone, in 4 × 4 × 40 nm voxels.
The script multiplies by `[0.004, 0.004, 0.04]` to obtain micrometres. The viewer
rigidly transforms `(x,y,z)` to `(x,z,-y)` and centres it; relative distances and
aspect ratios are preserved. These points are neither complete neurite skeletons
nor guaranteed soma locations. No decorative synapses or invented positions are
added. The source and derived file checksums accompany the bundled data.

Two source anchors have extreme coordinates: roots `720575940632908113` and
`720575940627146117`. They are retained exactly as supplied. Home frames the
0.1–99.9 percentile extent; Fit all includes every mapped coordinate. This is a
camera choice, not a correction or exclusion from the simulation.

## Activity

The worker publishes each recently active neuron's index and actual last-spike
tick, at up to 20 updates per wall second. The highlight fades over 1,500 neural
ticks (150 ms). Pausing freezes highlights; rotating the camera changes only the
view. Reset clears them. Black markers identify the currently stimulated targets.
No-input spontaneous activity is not invented to animate the brain.

This stream reads all neurons' last-spike state, independently of the limited
2,500-event raster display, so a busy raster does not erase anatomy activity.
The viewer checks a hash of the worker's neuron order before mapping any events.
OpenGL renders locally using the available graphics driver; neural computation
still uses the CPU. Display updates are live, but the model's simulated time
advances slower than wall time on this laptop. Version 0.4.0's default session
connects the DNa02 steering decoder on a shared clock; `--independent` retains
the earlier separate workers. See `NEURAL_STEERING.md` for the bridge's limits.

## Validation

Coordinate tests check model ordering, voxel scaling, missing-coordinate handling,
spike expiry and reset, and rejection of mismatched data. The native acceptance
test exercises the full brain alongside body physics, validates an OpenGL context
and visible spike data, saves screenshots, and checks that camera movement while
paused does not advance neural activity. The frozen executable is tested from
outside the source tree with Python environment variables removed and no system
C/C++ compiler available. This is host-machine validation, not a claim of general
Linux compatibility.
