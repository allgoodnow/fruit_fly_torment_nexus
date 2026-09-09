"""Reproduce the upstream walking loop and optionally render a QA frame."""
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

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=0.5)
parser.add_argument("--render", action="store_true")
args = parser.parse_args()
out = ROOT / "runs" / "body_probe"
out.mkdir(parents=True, exist_ok=True)
start = time.perf_counter()
body = FlyBody(render=args.render)
init_seconds = time.perf_counter() - start
start = time.perf_counter()
try:
    for _ in range(round(args.seconds / 0.01)):
        body.advance(0.01, wander=False)
    result = body.telemetry()
    result.update(initialization_seconds=init_seconds, wall_seconds=time.perf_counter() - start)
    if args.render:
        from PIL import Image
        Image.fromarray(body.render()).save(out / "fly.png")
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "path"}, indent=2))
finally:
    body.close()
