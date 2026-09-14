from pathlib import Path
import json
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

root = Path(SPECPATH).parent
datas = collect_data_files("flygym") + collect_data_files("flygym_demo.complex_terrain")
datas += collect_data_files('imageio_ffmpeg')
datas += copy_metadata("flygym", recursive=True)
datas += [(str(root / "THIRD_PARTY_NOTICES.md"), ".")]
datas += [(str(root / "docs/user-guide.md"), "docs")]
datas += [(str(root / "licenses"), "licenses")]
datas += [(str(root / "data/brain-v630"), "data/brain-v630")]
pack = root / "data/brain-male-cns-v1.0-lif"
manifest = json.loads((pack / "manifest.json").read_text())
# Local rollback arrays are retained on disk, not duplicated in the download.
active = {'manifest.json', 'ids.npy', 'offsets.npy', 'posts.npy',
          manifest.get('weights_file', 'weights.npy'), manifest['circuit_registry'],
          'anatomy.npz', 'anatomy-manifest.json'}
datas += [(str(pack / name), "data/brain-male-cns-v1.0-lif") for name in sorted(active)]
datas += [(str(root / "src/nexus/brain/intervention-circuits.json"), "nexus/brain")]
a = Analysis(
    [str(root / "launch.py")], pathex=[str(root / "src")],
    binaries=collect_dynamic_libs("mujoco") + collect_dynamic_libs("glfw"), datas=datas,
    # Upstream walking trajectories were pickled using the legacy NumPy module path.
    hiddenimports=["nexus.worker", "nexus.body", "nexus.brain.worker", "numpy.core.multiarray",
                   "OpenGL.platform.egl", "OpenGL.platform.glx"],
    hookspath=[], hooksconfig={"matplotlib": {"backends": ["Agg"]}},
    excludes=["notebook", "pytest", "brian2", "pandas", "pyarrow", "tkinter",
              "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="fruit-fly-torment-nexus",
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=True)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="FruitFlyTormentNexus-1.1.1")
