"""Equal-mean visual patterns, inferred columns, and downstream output blockade."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
for name, folder in [('NUMBA_CACHE_DIR','numba'), ('MPLCONFIGDIR','matplotlib'), ('FLYGYM_ASSET_CACHE_DIR','assets')]:
    os.environ.setdefault(name, str(ROOT / '.runtime' / folder))
import imageio_ffmpeg
import numpy as np
from nexus.brain.runtime import Brain, Connectome
from nexus.body import FlyBody
from nexus.coupled import CoupledSession
from nexus.retina import VisualColumns
from nexus.video import VideoSource


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    movies={}
    for pattern in ['left','right']:
        path=args.output_dir/f'{pattern}.mp4'
        frame=np.zeros((180,320,3), dtype=np.uint8)
        frame[:, :160] = 255
        if pattern=='right': frame=frame[:,::-1].copy()
        writer=imageio_ffmpeg.write_frames(str(path),(320,180),fps=20,codec='libx264rgb',
                                          pix_fmt_out='rgb24',quality=10,macro_block_size=1,
                                          ffmpeg_log_level='error')
        writer.send(None)
        for _ in range(6): writer.send(frame)
        writer.close()
        movies[pattern]=path
    pack=ROOT/'data/brain-male-cns-v1.0-lif'
    graph=Connectome.load(pack,allow_experimental=True)
    mapping=VisualColumns(graph.circuits)
    means={}
    for name,path in movies.items():
        video=VideoSource(path)
        means[name]=float(video.frame.mean())
        video.close()
    assert abs(means['left']-means['right']) < 1e-9
    body=FlyBody(render=False)
    session=None
    rows=[]
    comparisons=[]
    try:
        for seed in [73100,73101,73102]:
            saved={}
            for mode, pattern, blocked in [('pooled','left',False),('pooled','right',False),
                                           ('spatial','left',False),('spatial','right',False),
                                           ('spatial','left',True),('spatial','right',True)]:
                brain=Brain(graph,seed=seed)
                body.reset()
                session=CoupledSession(brain,body,autonomous=False)
                session.command('vision_video',str(movies[pattern]))
                session.command('vision_mapping',mode)
                session.command('eye_feedback',True)
                eye_ids=brain.circuit_ids('eye_left')+brain.circuit_ids('eye_right')
                eyes=brain.resolve(eye_ids)
                if blocked: brain.silence(eye_ids)
                session.advance(2500)
                packet=session.snapshot()['brain']
                assert abs(brain.time-body.time)<1e-9
                assert not brain.circuit_inputs and brain.inhibition_gain==1
                targets=np.setdiff1d(np.flatnonzero(brain.v < -52.5),eyes)
                row={'seed':seed,'mode':mode,'pattern':pattern,'output_blocked':blocked,
                     'total_spikes':int(brain.counts.sum()),'inhibited_cells':len(targets),
                     'minimum_mv':float(brain.v.min()),'video_frame':session.eyes.video.index,
                     'shared_clock':True}
                if blocked: assert len(targets)==0
                else: assert len(targets)>0
                if mode=='spatial':
                    values=mapping.sample(session.eyes.preview)
                    dark=np.concatenate([brain.resolve(mapping.ids[s])[values[s]==0] for s in ['L','R']])
                    bright=np.concatenate([brain.resolve(mapping.ids[s])[values[s]>.9] for s in ['L','R']])
                    assert len(dark)>0 and len(bright)>0
                    assert brain.counts[dark].sum()==0 and brain.counts[bright].sum()>0
                    mapped=np.concatenate([brain.resolve(mapping.ids[s]) for s in ['L','R']])
                    excluded=np.setdiff1d(eyes,mapped)
                    assert brain.counts[excluded].sum()==0
                    row.update(dark_input_spikes=0,bright_input_spikes=int(brain.counts[bright].sum()),
                               excluded_input_spikes=0)
                key=(mode,pattern,blocked)
                saved[key]=(brain.counts.copy(),brain.v.copy())
                np.savez_compressed(args.output_dir/f'{seed}-{mode}-{pattern}-{blocked}.npz',
                                     counts=brain.counts,voltage=brain.v)
                session.advance(1000)
                assert not brain.eye_inputs and not session.eyes.enabled and session.eyes.video.ended
                session.eyes.close()
                row['eof_released']=True
                rows.append(row)
                print(json.dumps(row),flush=True)
            for index in [0,1]:
                np.testing.assert_array_equal(saved['pooled','left',False][index],saved['pooled','right',False][index])
            for pattern in ['left','right']:
                np.testing.assert_array_equal(saved['spatial',pattern,False][0][eyes],saved['spatial',pattern,True][0][eyes])
            left,right=saved['spatial','left',False],saved['spatial','right',False]
            changed=int(np.count_nonzero(np.abs(left[1]-right[1])>.5))
            assert changed>0 and not np.array_equal(left[0],right[0])
            comparisons.append({'seed':seed,'pooled_patterns_identical':True,
                                'spatial_patterns_distinct':True,'cells_with_voltage_difference_over_0_5_mv':changed,
                                'blockade_preserves_input_spikes':True})
        report={'format':'nexus-spatial-vision-results-1','success':True,
                'manifest_sha256':hashlib.sha256((pack/'manifest.json').read_bytes()).hexdigest(),
                'image_means':means,'trials':rows,'comparisons':comparisons,
                'limits':['Column identities are inferred from L1/L2 agreement.',
                          'Video projection is uncalibrated; the physical eye cameras retain pooled input.',
                          'These tests establish spatial input routing, not object recognition or guided exploration.',
                          'Generic spiking photoreceptors and unbounded inhibitory current remain unfitted.']}
        (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:
        if session is not None: session.eyes.close()
        body.close()


if __name__=='__main__': main()
