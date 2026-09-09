"""Select exact v630 labels; preserve the distinction between circuits and proxies."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'vendor/flywire-annotations/supplemental_files/Supplemental_file1_annotations.tsv'
model_ids = set(np.load(ROOT / 'data/brain-v630/ids.npy', allow_pickle=False).tolist())
with source.open() as stream:
    annotations = list(csv.DictReader(stream, delimiter='\t'))
definitions = {
    'looming': ('hemibrain_type', 'LPLC2', 210, 'Looming-threat pathway',
                'https://doi.org/10.1016/j.cub.2019.01.079', 'Direct population input; no visual receptive-field or subjective fear model.'),
    'warmth': ('hemibrain_type', 'TRN_VP2', 7, 'Antennal warmth input',
               'https://doi.org/10.1038/nature12390', 'Temperature-to-rate curve is authored; not tissue heating or damage.'),
    'aversion_proxy': ('cell_type', 'CB0059', 2, 'Central aversion candidate',
                       'https://doi.org/10.1101/2025.10.28.684868',
                       'Type-based candidate from Figure S6I. Ascending nociceptor IDs are unresolved; not validated nociception or pain.'),
    'giant_fiber': ('cell_type', 'DNp01', 2, 'Giant fiber escape readout',
                    'https://doi.org/10.1016/j.cub.2019.01.079', 'Brain readout only; VNC and jump/flight muscles are not simulated.'),
}
registry = {'format': 'nexus-intervention-circuits-1', 'snapshot': '630',
            'annotation_commit': 'df6bb136f5b3d91c3992df4e8de2642329e2a384',
            'annotation_source': 'https://github.com/flyconnectome/flywire_annotations/tree/v1.1.0',
            'annotation_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'circuits': {}}
for key, (field, label, count, title, evidence, limitation) in definitions.items():
    rows = sorted([r for r in annotations if r[field] == label], key=lambda r: (r['side'], int(r['root_id'])))
    assert len(rows) == count and all(int(r['root_id']) in model_ids for r in rows)
    registry['circuits'][key] = {'title': title, 'selection': {field: label}, 'source': evidence,
                                 'limitation': limitation, 'ids': [r['root_id'] for r in rows],
                                 'sides': [r['side'] for r in rows]}
path = ROOT / 'src/nexus/brain/intervention-circuits.json'
path.write_text(json.dumps(registry, indent=2)+'\n')
print({key: len(value['ids']) for key,value in registry['circuits'].items()})
