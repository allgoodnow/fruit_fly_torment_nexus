"""Install exact annotation-bound visual cohorts without changing synaptic weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from nexus.brain.runtime import Connectome,Brain,weights_filename
from nexus.datasets.male_cns import FILES,digest
from nexus.datasets.graded_relays import attach_graded_relays


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--lamina-only',action='store_true',help='Install the earlier L1/L2-only model')
    args=parser.parse_args()
    pack=ROOT/'data/brain-male-cns-v1.0-lif'
    original=(pack/'manifest.json').read_bytes();manifest=json.loads(original)
    raw=ROOT/'data/raw/male-cns-v1.0'/FILES['annotations']
    record=next(r for r in manifest['source_structural_manifest']['sources']['files'] if r['file']==FILES['annotations'])
    if digest(raw)!=record['sha256']:raise ValueError('Annotations differ from this pack')
    graph=Connectome.load(pack,allow_experimental=True)
    registry=attach_graded_relays(graph.circuits,pd.read_feather(raw),graph.ids,include_medulla=not args.lamina_only)
    content=(json.dumps(registry,indent=2)+'\n').encode();checksum=hashlib.sha256(content).hexdigest()
    name='circuits-'+checksum[:16]+'.json'
    manifest['circuit_registry']=name;manifest['files'][name]=checksum
    with tempfile.TemporaryDirectory(prefix='.graded-',dir=pack) as directory:
        staging=Path(directory)
        for array in ['ids.npy','offsets.npy','posts.npy',weights_filename(manifest)]:
            (staging/array).hardlink_to(pack/array)
        (staging/name).write_bytes(content)
        (staging/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        brain=Brain(Connectome.load(staging,allow_experimental=True));brain.set_graded_relays(True)
        if (pack/'manifest.json').read_bytes()!=original:raise ValueError('Pack changed during preparation')
        if (pack/name).exists() and (pack/name).read_bytes()!=content:raise ValueError('Registry checksum collision')
        (staging/name).replace(pack/name)
        (staging/'manifest.json').replace(pack/'manifest.json')
    version=1 if args.lamina_only else 2
    report={'format':f'nexus-graded-relay-evidence-{version}','annotations':record,'cohort':registry['graded_relays'],
            'manifest_sha256':digest(pack/'manifest.json'),'weights_unchanged':True}
    (ROOT/f'experiments/graded-relay-evidence-v{version}.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/f'data/manifests/male-cns-runtime-v{9 if args.lamina_only else 10}.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:len(v) for k,v in registry['graded_relays']['groups'].items()}))


if __name__=='__main__':main()
