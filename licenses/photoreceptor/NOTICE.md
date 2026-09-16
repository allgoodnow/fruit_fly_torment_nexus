# Photoreceptor membrane research component

`src/nexus/brain/photoreceptor.py` is a Python adaptation of the MATLAB membrane
model by **Zhuoyi Song**, June 2017, distributed by **JuusolaLab** with
*Microsaccadic sampling of moving image information provides Drosophila hyperacute
vision*, Juusola et al., eLife 6:e26117 (2017).

- Paper: https://doi.org/10.7554/eLife.26117
- Source: https://github.com/JuusolaLab/Microsaccadic_Sampling_Paper
- Revision: `a4453f7e47abf2ea2a924c376d4c1c157c6011e2`
- Source files: `BiophysicalPhotoreceptorModel/wt_cc_model_pump.m`,
  `NaCaPump_Body.m`, `NaKPump.m`, and the BG1 parameters and TRP current-feedback
  rule in `Vol_FeedbackCluster.m`.
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

This component is not imported by the application and is not enabled in the GUI.
Its GPL license applies to this adapted component; this notice does not grant a
different license to unrelated project files. Any future distribution of a
combined application incorporating it must comply with the GPL's requirements.
