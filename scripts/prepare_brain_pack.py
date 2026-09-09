"""Prepare the validated v630 tables for fast, compiler-free runtime loading."""
import hashlib
import json
import os
from pathlib import Path
import sys
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".runtime/numba"))
import numpy as np
import pandas as pd
from nexus.brain.runtime import Connectome

source = ROOT / "vendor/brain-reference"
manifest = json.loads((ROOT / "data/manifests/brain-reference.json").read_text())
record = next(r for r in manifest["snapshots"] if r["snapshot"] == "630")
for name, expected in record["files"].items():
    with (source / name).open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected["sha256"]:
            raise RuntimeError(f"Reference hash mismatch: {name}")
ids = pd.read_csv(source / "2023_03_23_completeness_630_final.csv", index_col=0).index.to_numpy(dtype=np.int64)
edges = pd.read_parquet(source / "2023_03_23_connectivity_630_final.parquet",
                        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
graph = Connectome.from_edges(ids, edges.iloc[:, 0], edges.iloc[:, 1], edges.iloc[:, 2].to_numpy()*0.275, "630")
out = ROOT / "data/brain-v630"
out.mkdir(parents=True, exist_ok=True)
files = {}
for name in ("ids", "offsets", "posts", "weights"):
    path = out / f"{name}.npy"
    np.save(path, getattr(graph, name), allow_pickle=False)
    with path.open("rb") as stream:
        files[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
(out / "manifest.json").write_text(json.dumps({"format": "nexus-connectome-1", "snapshot": "630",
    "source_commit": manifest["commit"], "source_files": record["files"],
    "neurons": len(ids), "edges": len(graph.posts), "files": files,
    "weights": "mV; signed contact count * 0.275", "ordering": "stable by presynaptic index",
}, indent=2))
shutil.copyfile(source / "LICENSE", out / "REFERENCE_LICENSE")
print(f"Prepared {len(ids):,} neurons and {len(graph.posts):,} edges at {out}")
