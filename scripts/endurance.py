"""Longer headless walking run; record boundedness and actual stance stability."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))
os.environ.setdefault("FLYGYM_ASSET_CACHE_DIR", str(ROOT / ".runtime" / "assets"))
from nexus.body import FlyBody
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=10)
args = parser.parse_args()
body = FlyBody(render=False)
start = time.perf_counter()
z, upright, distances = [], [], []
try:
    for i in range(round(args.seconds / .01)):
        body.advance(.01)
        z.append(float(body.position()[2]))
        upright.append(float(body.sim.mj_data.xmat[body.thorax_id].reshape(3, 3)[2, 2]))
        distances.append(body.telemetry()["displacement_mm"])
        if (i+1) % 100 == 0:
            print(f"{body.time:.0f}s simulated; {time.perf_counter()-start:.1f}s wall time", flush=True)
    result = {
        "sim_seconds": body.time, "wall_seconds": time.perf_counter()-start,
        "z_mm_min_max": [min(z), max(z)], "upright_cosine_min": min(upright),
        "max_displacement_mm": max(distances), "path_samples": len(body.path),
        "finite_state": bool(np.isfinite(body.sim.mj_data.qpos).all()),
        "final": body.telemetry(),
    }
    out = ROOT / "runs" / "endurance"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "final"}, indent=2))
finally:
    body.close()
