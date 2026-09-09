import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".runtime" / "numba"))
os.environ.setdefault("FLYGYM_ASSET_CACHE_DIR", str(ROOT / ".runtime" / "assets"))
