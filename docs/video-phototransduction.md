# Video-to-molecular eye input

The offline video runner connects an MP4 to selected explicit photoreceptors:
decoded RGB → relative intensity → absorbed-photon counts → molecular
reactions → open TRP channels → membrane voltage. It uses the existing
[independent receptor batches](phototransduction-batch.md) on CUDA, or separate
per-unit-stream receptors on CPU. Every selected receptor retains 30,000
microvilli; responses are not multiplied to represent unmodeled cells.

This is a controlled exposure model. **An MP4 does not supply absorbed-photon
flux.** Its white and black rates are explicit experimental assumptions in
photons/s per receptor, after absorption. No physical calibration is inferred
from the file. Its saved responses can now feed the native app through
[prepared receptor playback](user-guide.md#prepared-receptor-playback).

## Exposure assumptions

`--transfer` is required. `linear` treats normalized RGB codes as an idealized
linear stimulus. `srgb` applies the piecewise inverse sRGB transfer separately
to each component before averaging, following the
[W3C color specification](https://www.w3.org/TR/css-color-4/#color-conversion-code).
For example, code 128 becomes approximately 0.216 under sRGB, versus 0.502
under the linear-code interpretation. This choice is not automatic detection
of a video's encoding or a complete display color-management pipeline.

The equal RGB average is a relative intensity proxy. It is not a human
luminance measurement or a fit to fly opsin sensitivity. Spectra, ultraviolet,
absolute illumination, eye optics and pixel acceptance fields remain unmodeled.
The existing inferred visual-column coordinates supply an uncalibrated image
projection. Bilinear sampling is applied after the selected transfer conversion.
Both eye cohorts sample the same video plane, as in the existing video mode.

For sampled intensity `u` between zero and one, the expected absorbed rate is:

```text
rate [photons/s] = black_rate + (white_rate - black_rate) × u
photons in each 1 ms bin ~ Poisson(rate / 1000)
```

Each receptor owns a separate seeded input generator. These generators are
separate from molecular event randomness. Inputs and seeds persist across
reads, so splitting a video exposure into different call sizes does not change
its photon sequence. An optional positive black rate represents an explicit
background exposure; zero means an idealized zero-illumination black stimulus,
not a measurement of display leakage or intrinsic neuronal noise.

The external temporal Poisson counts are an additional input assumption. The
[RandPAM analysis](https://doi.org/10.3389/fncom.2016.00061) deliberately fixes
the total photon input when studying internal absorption statistics. Here the
incoming total varies, then the existing cascade allocates that realized count
uniformly across microvilli and preserves the total. No extra, independently
sampled per-microvillus arrivals are added.

Video decoding retains the existing bounded 20 fps input. Each decoded frame
is held for 50 ms; photon counts are generated in 1 ms bins and delivered at
their boundaries. Sub-bin arrival times and video detail above this sampling
rate are not recovered. Channels and voltage use the existing 0.1 ms schedule.

## Exposure history before a clip

Receptor responses depend on prior light exposure: molecular recovery and
feedback contribute to adaptation in the source model
([Juusola et al., 2017, Figure 1](https://elifesciences.org/articles/26117/figures)).
Preparation can now run a chosen interval of constant absorbed-photon flux
before the video, using the same molecular and membrane models continuously.
No artificial gain adjustment or preset adapted voltage is inserted.

`--pre-exposure-ms` defaults to zero, preserving the previous initial-state
recordings. Set a positive duration with `--pre-exposure-rate-hz` to condition
the selected receptors. A zero rate gives a dark interval. These settings are
assumptions for an exposure experiment, not measured ambient illumination or
a guarantee that the model reaches equilibrium. For example, adding
`--pre-exposure-ms 100 --pre-exposure-rate-hz 30000` runs 100 ms at an assumed
30,000 absorbed photons/s per cell before recording the clip.

The pre-interval retains all molecular pools, pending reactions, random streams
and membrane state into the video. Its photon generator uses a separate seed
branch, so changing pre-exposure leaves the clip's intensity and absorbed
photon counts identical for the same video settings and root seed. Responses
can differ because the receptor starts with a different history; this includes
the membrane's evolution as well as the cascade and its random-state evolution.

Video time still starts at zero. Only video-era traces are saved in
`response.npz`. The report records pre-exposure duration, rate, photon seeds,
realized photon totals, molecular event totals and the actual initial receptor
voltages. Per-receptor `molecular_events` continues to count video-era events;
pre-exposure events are recorded separately. Playback starts the selected
sources at their saved initial voltages. Older reports retain their declared
reference-voltage start.

Pre-exposure conditions **receptors only**. It neither advances the brain/body
clock nor supplies prior release to downstream neurons. Those neurons still
start at the existing model state, so this does not remove network startup
transients or establish a physiological whole-brain baseline. The offline
video/dark comparison requires matching conditioning and initial voltages.
Compute time grows with pre-exposure duration, including at a zero photon rate.

The [focused pre-exposure result](../experiments/receptor-pre-exposure-v1-results.json)
uses the existing eight-cell 500 ms clip with a 100 ms, 30,000-photons/s
pre-interval. Video intensities and photon counts stay identical, while the
receptors enter the clip at −62.00 to −59.69 mV with ongoing channel activity.
The largest voltage difference from the unconditioned recording is 10.55 mV
in this model. Disabling pre-exposure reproduces all five previous response
arrays exactly. A CPU check matches uninterrupted receptor continuation;
native CUDA preparation matches the offline conditioned recording, and native
playback matches the independent full-brain endpoint. These checks establish
state continuity and reproducibility, not physiological accuracy.

## Running a clip

The source app now offers **Prepare receptor response…** in the Brain tab.
It uses this same operation in a separate worker, with explicit exposure,
CPU/CUDA and release-curve choices. The [user guide](user-guide.md#prepared-receptor-playback)
describes progress, cancellation, saved recordings and loading. Preparation
leaves brain and body time unchanged; it is not simultaneous live molecular
vision. The GUI offers the default eight-cell subset, a 64-cell spatial sample,
or all 3,155 receptors with mapped image coordinates.

```sh
python scripts/run_video_phototransduction.py \
  --video /path/to/clip.mp4 \
  --white-rate-hz 30000 --black-rate-hz 0 --transfer srgb \
  --duration-ms 500 --backend cuda --output-dir new-results-folder
```

The default sample contains four mapped receptors near quadrant centers in
each eye. These eight cells are a named subset, not the complete retinal
population. `--receptor-ids` can specify a different mapped subset explicitly.
`--coverage sample64` extends the sample to 32 cells per eye, choosing successive
cells farthest from the current sample in the inferred image plane.
`--coverage mapped` includes all 3,155 mapped R1–R6 cells. Both presets retain the
first eight IDs and their seeds. The registry contains 3,377 R1–R6 cells in total;
the 222 without image coordinates are excluded from this video mapping.
`--backend cpu` runs the same per-unit-stream model without requiring a GPU,
but its exhaustive reference scheduler is slower. `--batch-cells`, from 1 to 8,
limits how many independent molecular receptors remain resident at once.

Preparation decodes the requested exposure once into temporary disk arrays.
It then completes each small group's entire exposure before constructing the
next group. Every cell uses its own image sample, photon stream, molecular
random streams, 30,000 microvilli and membrane state. These responses are not
copies of a sample cell. This ordering is valid because this offline model has
no feedback or interactions between receptors. It cannot substitute for a live
eye with synaptic feedback.

The default eight-cell group retains 24.96 MB of molecular state, plus a similar
candidate copy during advancement and other working storage. Trace storage and
total computation still grow with coverage and duration. Temporary disk arrays
are deleted when the operation finishes or is cancelled. GUI duration limits
retain the existing 2,127 ms cap for all mapped cells in this pack, now bounding
the temporary playback voltage file to 512 MiB. Playback validates that file
in bounded blocks and reads 50 ms windows, calculating release on demand.
The complete voltage and release arrays no longer stay resident. Longer
recordings can be prepared from the terminal but require enough storage for
the saved archive and an uncompressed temporary voltage trace; system file
caching and memory-backed temporary directories can still consume memory.

The output contains `response.npz` with input intensities, absorbed counts,
channel counts and voltage; `response.png` showing up to the first eight model
traces, labeled with the displayed/total count; and `report.json` with IDs,
coordinates, coverage, resident-group size, rates, seeds, source hashes and limits.
Video exposure stops at EOF rather than extending the last frame. Closing
light input does not reset molecular or membrane state; ongoing reactions
decay through the model. Without pre-exposure, the original published membrane
initial condition is used, not an assumed equilibrated dark state.

The shared implementation is `src/nexus/receptor_preparation.py`. GUI and
terminal paths keep the same seeds, 50 ms input batches, 0.1 ms membrane samples
and per-cell molecular populations. Reports identify the source implementation;
the packaging configuration supplies a source-hash manifest for frozen builds.
The updated packaged application and macOS preparation have not been checked.
Native workers now publish their latest progress and terminal outcome to an
atomic local JSON status file, removing the dependency on a progress message
queue. The dialog requires both a complete outcome and successful worker exit
before loading. Cancellation requests termination if cooperative stopping has
not completed within two seconds. This changes worker supervision, not the
input, molecular dynamics or playback. See
[`receptor-worker-status-v1-results.json`](../experiments/receptor-worker-status-v1-results.json)
for the focused workflow checks. The earlier stalled native attempt was not
reproduced, so these changes do not establish its original cause.
The focused result is
[`receptor-preparation-gui-v1-results.json`](../experiments/receptor-preparation-gui-v1-results.json):
the native CUDA preparation reproduces the previous eight-receptor recording
exactly, then its playback reproduces the previous final full-brain voltage and
spike counts. A CPU CLI run reproduces the matching one-receptor prefix. Four
focused checks cover CPU preparation, ordinary cancellation and existing
prepared-playback controls. These check integration and reproducibility rather
than physiological accuracy.

The coverage update is recorded in
[`receptor-coverage-v1-results.json`](../experiments/receptor-coverage-v1-results.json).
One 50 ms spatial-gradient exposure prepared all 3,155 mapped receptors on the
RTX 4060 Laptop GPU in 111.51 s, with 569.3 MiB peak host RSS in that preparation
process. This includes cold initialization/JIT and is not a live-vision rate or
a forecast for longer clips. The first eight response columns match the prior
driver exactly; two additional columns match independent CPU calculations.
Native playback matches the offline full-brain endpoint, including pause,
release and resume. A separate 500 ms native preparation preserves the previous
eight-cell recording and full-brain result. Four focused CPU checks cover the
ordinary preparation, cancellation, selection and bounded plot loading paths.

At the 50 ms endpoint, blocking the mapped receptors' outgoing output changes
44,250 non-source voltages by at least 0.01 mV. Both runs retain the same source
voltages. This checks dependence on actual outgoing connections, including
tonic release; it does not isolate the light response from baseline activity.
The existing unfitted relay laws and initial-state transients remain in place.

## Focused demonstration

The recorded demonstration uses a 320×180 lossless-RGB MP4 at 20 fps:
100 ms black, 100 ms left half white, 100 ms right half white, 100 ms full
code-128 gray, then 100 ms black. Assumed white is 30,000 absorbed photons/s;
black is zero and transfer is sRGB. A matched control runs the same video and
root seed with both photon rates zero. Results and implementation hashes are
in `experiments/video-phototransduction-v1-results.json`.

The normal-use test checks spatial input, black/white/gray interpretation,
photon sequence consistency across reads, and the exposure clock through EOF.
Existing ordinary video and spatial sampling checks cover the shared decoder
and bilinear helper.

This establishes video-driven molecular and voltage responses in selected
model receptors. The [offline connectome replay](receptor-brain-replay.md)
can now use those voltages with an explicitly supplied, unfitted release
curve. Neither experiment establishes physiological calibration, downstream
perception or natural behavior. The live application's generic spiking
photoreceptor input remains in place for ordinary visual mappings; prepared
playback uses stored voltages and an explicit release curve. The adapted
molecular and membrane components retain their GPL terms in
`licenses/photoreceptor/NOTICE.md`.
