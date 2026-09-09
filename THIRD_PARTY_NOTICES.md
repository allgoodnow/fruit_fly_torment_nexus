# Third-party software

This prototype uses **NeuroMechFly / FlyGym 2.1.0** from
https://github.com/NeLy-EPFL/flygym, pinned to commit
`38c8ec61034cd59bc5ba0de20688d4a3c0000d60` (Apache-2.0).
It reuses the official complex-terrain hybrid turning controller and bundled
fly meshes and walking trajectories. Upstream source is unmodified.
The license is in `licenses/flygym-LICENSE`.

Physics and rendering use **MuJoCo 3.9.0** (Apache-2.0),
https://github.com/google-deepmind/mujoco.
The native interface uses **PySide6 / Qt 6.11.2** (LGPLv3/GPLv3/commercial)
and **pyqtgraph 0.14.0** (MIT).
The package includes dynamically loaded Qt libraries in `_internal/PySide6`.
Corresponding upstream Qt sources are available at https://download.qt.io/archive/qt/6.11/.
Qt for Python sources are available at https://code.qt.io/cgit/pyside/pyside-setup.git/.
Users may replace compatible shared libraries in the extracted folder, and
reverse engineering to debug modifications to LGPL components is permitted.

Additional runtime dependencies and their installed license notices are collected
in `licenses/` by the packaging script. `requirements-lock.txt` records the
development environment, including research and build tools not shipped in the app.

The neural model follows Philip Shiu and Nico Spiller's **Drosophila brain model**
at https://github.com/philshiu/Drosophila_brain_model, commit
`91bdd1e7dcf193f3e7ca5a8933497fcef63b7960` (MIT; copyright 2023 Philip Shiu and
Nico Spiller). The reference license accompanies the prepared data as
`data/brain-v630/REFERENCE_LICENSE`. The numerical runtime is a new implementation
checked against Brian2. **Numba** and **llvmlite** provide runtime compilation;
their licenses are included in `licenses/`.

The prepared v630 connectivity and neuron-ID tables come from that pinned
reference repository. File hashes and the transformation are recorded in
`data/brain-v630/manifest.json`. These are FlyWire-derived data; credit the
Princeton FlyWire team and members of the Murthy and Seung labs for developing
and maintaining FlyWire, supported by BRAIN Initiative grant MH117815.
See https://edit.flywire.ai/credits.html and https://flywire.ai for underlying
project attribution and data terms. The upstream MIT software license is not a
blanket license for other FlyWire data products or EM imagery. This application
does not bundle the underlying EM images.

This application is independent of and is not endorsed by the upstream projects
or the creator of the FlyLab reference video. Its walking controller is engineered
locomotion. Version 0.4.0 adds an experimental DNa02-to-controller steering
decoder with authored gains; it does not reconstruct the ventral nerve cord.
Functional motivation: Yang et al., Cell (2024), doi:10.1016/j.cell.2024.08.033.
No subjective-experience claim is implied.

Version 0.5.0 adds a pooled foot-contact-to-sugar-input proxy using the existing
reference taste cohort. The pathway/readout motivation is Shiu et al., Nature
(2024), doi:10.1038/s41586-024-07763-9. The peripheral mapping and food geometry are
application-authored approximations, not data or a model validated by that study.

The anatomical cell-anchor positions use the official FlyWire annotations
repository at https://github.com/flyconnectome/flywire_annotations/tree/v1.1.0,
commit `df6bb136f5b3d91c3992df4e8de2642329e2a384`. Credit the FlyWire Consortium
and the annotation contributors listed by that project. The derived bundle
contains positions joined by v630 root ID, not EM images or neuron skeletons.
Source and derived checksums and coordinate conversion are recorded in
`data/brain-v630/anatomy-manifest.json`. These data retain the upstream FlyWire
attribution and data terms described above. PyOpenGL 3.1.10 supplies the Python
OpenGL bindings; its installed license accompanies the other dependency notices.

Version 0.10.0 uses the published abdominal multidendritic sensory cohort from
https://github.com/jesmjones/nociceptive_pathways_paper, commit
`39a75c0f3c5e7ce2facf56666a0a5908f3bf11e7` (Apache-2.0), accompanying
"Ascending nociceptive pathways drive rapid escape and sustained avoidance in
adult Drosophila", doi:10.1101/2025.10.28.684868. The upstream license is included
in `licenses/nociceptive-pathways/LICENSE`. We read the cohort from the figure 4
and supplementary figure 5 notebooks without executing their code. Our mapping
to MaleCNS uses the official cross-specimen annotations; it is an experimental
candidate mapping, not a validated model of subjective pain. Source checksums,
excluded correspondences, and derived targets are recorded in the accompanying
experiment files. MaleCNS attribution is in `licenses/MaleCNS/NOTICE`.
