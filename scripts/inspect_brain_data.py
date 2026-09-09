"""Validate bundled reference tables and write a lossless source manifest."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vendor" / "brain-reference"
records = []
for snapshot, comp_name, con_name in (
    ("630", "2023_03_23_completeness_630_final.csv", "2023_03_23_connectivity_630_final.parquet"),
    ("783", "Completeness_783.csv", "Connectivity_783.parquet"),
):
    comp, con = SOURCE / comp_name, SOURCE / con_name
    ids = pd.read_csv(comp, index_col=0).index.to_numpy(dtype=np.int64)
    assert len(np.unique(ids)) == len(ids), "Duplicate neuron IDs"
    table = pq.ParquetFile(con)
    total_contacts = positive = negative = zero = 0
    for batch in table.iter_batches(batch_size=262144):
        arrays = {name: batch.column(name).to_numpy() for name in batch.schema.names}
        pre, post = arrays["Presynaptic_Index"], arrays["Postsynaptic_Index"]
        assert pre.min() >= 0 and post.min() >= 0
        assert pre.max() < len(ids) and post.max() < len(ids)
        np.testing.assert_array_equal(ids[pre], arrays["Presynaptic_ID"])
        np.testing.assert_array_equal(ids[post], arrays["Postsynaptic_ID"])
        counts = arrays["Connectivity"]
        signed = arrays["Excitatory x Connectivity"]
        assert counts.min() >= 0
        np.testing.assert_array_equal(counts * arrays["Excitatory"], signed)
        total_contacts += int(counts.sum())
        positive += int((signed > 0).sum())
        negative += int((signed < 0).sum())
        zero += int((signed == 0).sum())
    record = {
        "snapshot": snapshot, "neurons": len(ids), "connection_rows": table.metadata.num_rows,
        "summed_connectivity_contacts": total_contacts,
        "positive_weight_rows": positive, "negative_weight_rows": negative, "zero_weight_rows": zero,
        "checks": ["unique neuron IDs", "indices in bounds", "indices match FlyWire IDs",
                   "nonnegative contact counts", "signed weights match source columns"],
        "files": {},
    }
    for path in (comp, con):
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        record["files"][path.name] = {"sha256": checksum, "bytes": path.stat().st_size}
    records.append(record)
    print(json.dumps(record, indent=2), flush=True)
out = ROOT / "data" / "manifests"
out.mkdir(parents=True, exist_ok=True)
(out / "brain-reference.json").write_text(json.dumps({
    "source": "https://github.com/philshiu/Drosophila_brain_model",
    "commit": "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960",
    "retrieved": "2026-09-08", "code_license": "MIT",
    "dataset_redistribution": "Reference repository license recorded; separate dataset terms still to verify before packaging brain data.",
    "snapshots": records,
}, indent=2))
