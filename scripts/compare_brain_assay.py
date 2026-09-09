"""Compare actual spike counts; preserve the saved reference's uncertain rate."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "runs" / "brain_assay_630"
report = json.loads((out / "report.json").read_text())
ids = np.load(out / "neuron_ids.npy", allow_pickle=False)
input_ids = np.array([int(i) for i in report["input_ids"]], dtype=np.int64)
indices = np.flatnonzero(np.isin(ids, input_ids))
records = []
for condition in sorted({r["condition"] for r in report["results"]}):
    trials = [r for r in report["results"] if r["condition"] == condition]
    input_rates = []
    for r in trials:
        with np.load(out / f"{condition}-{r['trial']:02}.npz", allow_pickle=False) as spikes:
            input_rates.append(float(np.isin(spikes["neuron_index"], indices).sum() / len(indices)))
    rates = [r["mn9_hz"] for r in trials]
    records.append({"condition": condition, "source": "new integration run", "n": len(trials),
                    "mn9_mean_hz": float(np.mean(rates)), "mn9_std_hz": float(np.std(rates)),
                    "mean_driven_neuron_hz": float(np.mean(input_rates))})
for name in ("sugarR_100Hz.parquet", "sugarR.parquet"):
    df = pd.read_parquet(ROOT / "vendor" / "brain-reference" / "results" / "example" / name)
    rates = df[df.flywire_id == int(report["readout_id"])].groupby("trial").size().reindex(range(30), fill_value=0)
    records.append({"condition": name, "source": "saved upstream example", "n": 30,
                    "mn9_mean_hz": float(rates.mean()), "mn9_std_hz": float(rates.std(ddof=0)),
                    "mean_driven_neuron_hz": len(df[df.flywire_id.isin(input_ids)]) / 30 / len(input_ids)})
result = {"duration_seconds": 1, "std_definition": "population standard deviation of trial rates",
          "caveat": "Saved sugarR stimulus rate is not explicitly recorded; its driven-cell spikes suggest roughly 200 Hz, whereas current source defaults to 150 Hz.",
          "comparisons": records}
(out / "comparison.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
