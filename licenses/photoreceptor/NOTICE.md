# Photoreceptor membrane research component

`src/nexus/brain/photoreceptor.py` and `src/nexus/brain/phototransduction.py` adapt
the MATLAB membrane and microvillus reaction model by **Zhuoyi Song**, June 2017,
distributed by **JuusolaLab** with
*Microsaccadic sampling of moving image information provides Drosophila hyperacute
vision*, Juusola et al., eLife 6:e26117 (2017).

- Paper: https://doi.org/10.7554/eLife.26117
- Source: https://github.com/JuusolaLab/Microsaccadic_Sampling_Paper
- Revision: `a4453f7e47abf2ea2a924c376d4c1c157c6011e2`
- Source files: `BiophysicalPhotoreceptorModel/wt_cc_model_pump.m`,
  `NaCaPump_Body.m`, `NaKPump.m`, and the BG1 parameters and TRP current-feedback
  rule in `Vol_FeedbackCluster.m`.
- Additional reaction sources: `Initial_PR_Goodaa1new.m`,
  `RandomBumpModel_GillespieD_Continue_TRPLa.m`, `TRP_Rev_GHK_online.m`, and
  the feedback-strength selection in `Gillespie_sim_b2p3_Skew_speed205.m`.
- License: GNU GPL version 3; the upstream `LICENCE.txt` is copied as `LICENSE.txt`.

Changes: Python/Numba implementation, fixed 0.1 ms RK4 integration, persistent
batched cells, analytic limits at removable rate singularities, finite-state
validation, and physical-unit input validation. The published BG1 equations and
parameters are retained. The research verifier extracts scalar expressions from
the pinned source for comparison and records its checksums.

The September 2026 channel-input extension retains the source's 8 pS/+20 mV
inward-current rule and replaces its offline voltage iterations with continuous
ODE coupling. Current is recalculated at every RK4 substage. This numerical
change is documented and checked against a separate adaptive solver. It does
not include the upstream biochemical cascade or GHK microvillus model.

The subsequent offline phototransduction component adapts the reaction rates,
parameters and GHK/fast-calcium calculation. It uses unrounded exponential
event waits, seeded NumPy random draws, uniform photon allocation, persistent
state, causal channel sampling and atomic updates. This new event schedule
omits the source's empirical latency offset, uniform floor and special
rhodopsin protection; it is not a full reproduction of the source simulator.
See `docs/phototransduction.md` for numerical assumptions and limitations.

The timing-audit scripts generate source copies in a user-selected results
folder to observe events, prevent duplicate photon delivery, vary waiting-time
rules, and refresh the G-protein activation rate after input. These diagnostic
changes are described in `docs/phototransduction-timing-ablation.md`; they do
not modify the source checkout or the application's live sensory pathway.

The scheduler optimization uses conservative per-block event bounds, an
incremental channel total, and allocation-free scalar validation. It retains
the existing reaction and RNG order. Exact output/state comparisons against
the prior Python implementation are described in
`docs/phototransduction-performance.md`.

The optional `photoreceptor_cuda.py` component runs the same BG1 derivative and
TRP current functions with float64 RK4 integration on CUDA. The shared CPU
derivative was refactored to return a fixed tuple, and inward-current
rectification uses an equivalent explicit branch for device compilation.
Per-cell error reporting and atomic host commits preserve rollback semantics.
See `docs/photoreceptor-cuda.md` for numerical comparisons and scope.

The optional `phototransduction_parallel.py` and `phototransduction_cuda.py`
components share the reaction, GHK and fast-calcium equations through
allocation-free helpers. They introduce per-microvillus xoroshiro128+ streams,
separate photon-allocation randomness, inverse-CDF event waits, parallel
molecular execution and integer channel reduction. Seeded trajectories differ
from the previous shared-stream implementation. Float64 calculations, bounded
scratch chunks, per-unit work guards and candidate-state commits are described
in `docs/phototransduction-cuda.md`, along with comparisons and limitations.

This component is not imported by the application and is not enabled in the GUI.
Its GPL license applies to this adapted component; this notice does not grant a
different license to unrelated project files. Any future distribution of a
combined application incorporating it must comply with the GPL's requirements.
