"""Verify the MDN motor readout against source annotations and synaptic edges."""
import hashlib
import json
from pathlib import Path
import sys
import os
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT/'.runtime/numba'))
import numpy as np
import pandas as pd
from nexus.brain.runtime import Brain, Connectome
from nexus.brain.motor_effects import MotorEffects


def main():
    pack = ROOT/'data/brain-male-cns-v1.0-lif'
    annotation = ROOT/'data/raw/male-cns-v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = json.loads((pack/'manifest.json').read_text())
    source = next(x for x in manifest['source_structural_manifest']['sources']['files'] if x['file']==annotation.name)
    assert sha(annotation)==source['sha256']
    graph = Connectome.load(pack, allow_experimental=True)
    b = Brain(graph)
    effects = MotorEffects(b)
    table = pd.read_feather(annotation)
    mdn = table.loc[table.type.eq('MDN') & table.bodyId.isin(graph.ids)]
    assert set(map(int, mdn.bodyId))==set(map(int,graph.ids[effects.mdn]))
    assert mdn.somaSide.value_counts().to_dict()=={'R':2,'L':2}
    lbl = table.loc[table.type.eq('LBL40') & table.bodyId.isin(graph.ids)]
    assert len(lbl)==2
    edges=[]
    for pre in mdn.bodyId:
        i=b.lookup[int(pre)];start,end=graph.offsets[i:i+2]
        for post in lbl.bodyId:
            indices=np.flatnonzero(graph.posts[start:end]==b.lookup[int(post)])
            for j in indices:
                weight=float(graph.weights[start+j]);assert weight>0
                edges.append({'pre':int(pre),'post':int(post),'contacts':round(weight/.275),'model_weight_mv':weight})
    assert len(edges)==4 and {e['pre'] for e in edges}==set(map(int,mdn.bodyId))
    fields=['bodyId','type','somaSide','mancType','flywireType']
    report={'format':'nexus-descending-evidence-1','success':True,'dataset':graph.snapshot,
            'annotation_sha256':sha(annotation),'manifest_sha256':sha(pack/'manifest.json'),
            'mdn_annotations':json.loads(mdn[fields].to_json(orient='records')),
            'lbl40_annotations':json.loads(lbl[fields].to_json(orient='records')),
            'mdn_to_lbl40_edges':edges,
            'sources':[{'url':'https://www.nature.com/articles/s41467-020-19936-x',
                        'supports':'MDN command role in backward walking and LBL40 leg premotor involvement.'},
                       {'url':'https://pmc.ncbi.nlm.nih.gov/articles/PMC12636578/',
                        'supports':'Abdominal md stimulation elicits rapid forward locomotion without consistent directional turning; current model retreat is a mismatch.'}],
            'limitations':['Type correspondence and synapses do not validate firing dynamics.',
                           'Leg trajectories and rate-to-command mapping remain engineered.',
                           'No subjective-state inference.']}
    (ROOT/'experiments/descending-evidence-v1.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'matched_mdn':len(mdn),'matched_lbl40':len(lbl),'verified_edges':edges},indent=2))


if __name__=='__main__':
    main()
