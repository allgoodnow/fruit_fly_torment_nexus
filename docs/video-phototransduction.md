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
from the file. It does not feed the live GUI's brain yet.

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

## Running a clip

```sh
python scripts/run_video_phototransduction.py \
  --video /path/to/clip.mp4 \
  --white-rate-hz 30000 --black-rate-hz 0 --transfer srgb \
  --duration-ms 500 --backend cuda --output-dir new-results-folder
```

The default sample contains four mapped receptors near quadrant centers in
each eye. These eight cells are a named subset, not the complete retinal
population. `--receptor-ids` can specify a different mapped subset explicitly.
`--backend cpu` runs the same per-unit-stream model without requiring a GPU,
but its exhaustive reference scheduler is slower. CUDA molecular shard size
is configurable with `--batch-cells`.

The output contains `response.npz` with input intensities, absorbed counts,
channel counts and voltage; `response.png` with the measured traces; and
`report.json` with IDs, coordinates, rates, seeds, source hashes and limits.
Video exposure stops at EOF rather than extending the last frame. Closing
light input does not reset molecular or membrane state; ongoing reactions
decay through the model. The original published membrane initial condition
is used, not an assumed equilibrated dark state.

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
photoreceptor input remains in place; the new interface is offline. The adapted molecular and membrane
components retain their GPL terms in `licenses/photoreceptor/NOTICE.md`.
