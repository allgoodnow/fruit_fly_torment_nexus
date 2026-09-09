"""Run the unmodified Shiu reference taste-input assay, with explicit seeds.

Defaults to a small integration sample; use --trials 30 for the reference count.
No model equations, weights, or timestep are changed by this harness.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vendor" / "brain-reference"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))
# The bundled interpreter was built with Clang; this host provides GCC instead.
if not shutil.which("clang++") and shutil.which("g++"):
    os.environ.setdefault("CC", "gcc")
    os.environ.setdefault("CXX", "g++")
sys.path.insert(0, str(SOURCE))
import brian2 as b
import numpy as np
import pandas as pd
import model

parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=3)
parser.add_argument("--snapshot", choices=["630", "783"], default="630")
parser.add_argument("--rates", nargs="+", type=int, default=[100, 150])
parser.add_argument("--resume", action="store_true", help="Retain completed trials after validating their provenance")
args = parser.parse_args()
if args.trials < 1:
    parser.error("--trials must be positive")
if any(rate <= 0 or rate > 1000 for rate in args.rates):
    parser.error("assay input rates must be in (0, 1000] Hz")
b.prefs.codegen.target = "cython"
b.prefs.codegen.runtime.cython.cache_dir = str(ROOT / ".runtime" / "brian-cython")
b.defaultclock.dt = .1 * b.ms
comp = SOURCE / ("2023_03_23_completeness_630_final.csv" if args.snapshot == "630" else "Completeness_783.csv")
con = SOURCE / ("2023_03_23_connectivity_630_final.parquet" if args.snapshot == "630" else "Connectivity_783.parquet")

# Literal IDs from the pinned upstream example.ipynb; no cross-snapshot remapping.
SUGAR = [720575940624963786,720575940630233916,720575940637568838,
         720575940638202345,720575940617000768,720575940630797113,
         720575940632889389,720575940621754367,720575940621502051,
         720575940640649691,720575940639332736,720575940616885538,
         720575940639198653,720575940620900446,720575940617937543,
         720575940632425919,720575940633143833,720575940612670570,
         720575940628853239,720575940629176663,720575940611875570]
MN9 = 720575940660219265
ids = pd.read_csv(comp, index_col=0).index.to_numpy()
lookup = {int(fid): i for i, fid in enumerate(ids)}
missing = [str(fid) for fid in SUGAR + [MN9] if fid not in lookup]
if missing:
    raise RuntimeError(f"Snapshot {args.snapshot} lacks required IDs: {missing}")
out = ROOT / "runs" / f"brain_assay_{args.snapshot}"
out.mkdir(parents=True, exist_ok=True)
manifest = {
    "source": "https://github.com/philshiu/Drosophila_brain_model",
    "commit": "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960",
    "snapshot": args.snapshot, "neurons": len(ids),
    "files": {p.name: hashlib.file_digest(p.open("rb"), "sha256").hexdigest() for p in (comp, con, SOURCE / "model.py")},
    "brian2": b.__version__, "dt_ms": .1, "duration_ms": 1000,
    "trials_per_stimulus": args.trials, "input_ids": [str(i) for i in SUGAR],
    "readout_id": str(MN9), "results": [],
}
prior_wall = 0
if args.resume and (out / "report.json").exists():
    previous = json.loads((out / "report.json").read_text())
    for key in ("commit", "snapshot", "files", "brian2", "dt_ms", "duration_ms", "input_ids", "readout_id"):
        if previous[key] != manifest[key]:
            raise RuntimeError(f"Cannot resume: {key} differs")
    manifest["results"] = previous["results"]
    prior_wall = previous["wall_seconds"]
start = time.perf_counter()
conditions = [("baseline", 0, 1)] + [(f"sugar_{rate}Hz", rate, args.trials) for rate in args.rates]
for condition, rate, repeats in conditions:
    for trial in range(repeats):
        seed = 73100 + trial
        if any(r["condition"] == condition and r["trial"] == trial and r["seed"] == seed for r in manifest["results"]):
            if not (out / f"{condition}-{trial:02}.npz").exists():
                raise RuntimeError("Report references missing spike data")
            continue
        b.start_scope()
        b.seed(seed)
        params = dict(model.default_params)
        params["r_poi"] = rate * b.Hz
        print(f"Starting {condition}, trial {trial+1}/{repeats}", flush=True)
        wall = time.perf_counter()
        trains = model.run_trial([lookup[f] for f in SUGAR] if rate else [], [], [], comp, con, params)
        # Preserve actual spikes by neuron index; the adjacent ID map is lossless int64.
        active = sorted(trains)
        spike_i = np.concatenate([np.full(len(trains[i]), i, dtype=np.int32) for i in active]) if active else np.array([], dtype=np.int32)
        spike_t = np.concatenate([np.asarray(trains[i] / b.second) for i in active]) if active else np.array([], dtype=float)
        np.savez_compressed(out / f"{condition}-{trial:02}.npz", neuron_index=spike_i, time_seconds=spike_t)
        result = {"condition": condition, "trial": trial, "seed": seed,
                  "mn9_hz": len(trains.get(lookup[MN9], [])), "active_neurons": len(active),
                  "total_spikes": len(spike_i), "wall_seconds": time.perf_counter()-wall,
                  "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
        manifest["results"].append(result)
        manifest["wall_seconds"] = prior_wall + time.perf_counter()-start
        (out / "report.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps(result), flush=True)
np.save(out / "neuron_ids.npy", ids, allow_pickle=False)
for label in sorted({r["condition"] for r in manifest["results"]}):
    values = [r["mn9_hz"] for r in manifest["results"] if r["condition"] == label]
    print(f"{label}: MN9 {np.mean(values):.2f} ± {np.std(values):.2f} Hz across {len(values)} trials", flush=True)
