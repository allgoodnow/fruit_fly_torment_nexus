"""Sustained light, dark recovery, and full-network photoreceptor blockade."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
for name,folder in [('NUMBA_CACHE_DIR','numba'),('MPLCONFIGDIR','matplotlib'),('FLYGYM_ASSET_CACHE_DIR','assets')]:
    os.environ.setdefault(name,str(ROOT/'.runtime'/folder))
import imageio_ffmpeg
import numpy as np
from nexus.brain.runtime import Brain,Connectome
from nexus.body import FlyBody
from nexus.coupled import CoupledSession


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    movie=args.output_dir/'light-dark-light.mp4'
    writer=imageio_ffmpeg.write_frames(str(movie),(320,180),fps=20,codec='libx264rgb',
                                      pix_fmt_out='rgb24',quality=10,macro_block_size=1,ffmpeg_log_level='error')
    writer.send(None)
    for brightness in [255]*14+[0]*14+[255]*4:
        writer.send(np.full((180,320,3),brightness,dtype=np.uint8))
    writer.close()
    pack=ROOT/'data/brain-male-cns-v1.0-lif'
    graph=Connectome.load(pack,allow_experimental=True)
    body=FlyBody(render=False)
    session=None
    rows=[]
    try:
        for seed in [73100,73101,73102]:
            reference={}
            for condition in ['static','adaptive','adaptive_blocked']:
                brain=Brain(graph,seed=seed)
                body.reset()
                session=CoupledSession(brain,body,autonomous=False)
                session.command('vision_video',str(movie))
                session.command('vision_mapping','spatial')
                session.command('vision_adaptation',condition!='static')
                session.command('eye_feedback',True)
                eye_ids=brain.circuit_ids('eye_left')+brain.circuit_ids('eye_right')
                eyes=brain.resolve(eye_ids)
                if condition=='adaptive_blocked':brain.silence(eye_ids)
                targets=np.unique(np.concatenate([graph.posts[graph.offsets[i]:graph.offsets[i+1]][
                    graph.weights[graph.offsets[i]:graph.offsets[i+1]]<0] for i in eyes]))
                trace=[]
                for _ in range(33):
                    session.advance(500)
                    state=session.eyes.snapshot()
                    trace.append({'time_ms':brain.step/10,'input_hz':state['rates_hz'],
                                  'background':state['adaptation']['mean_background'],
                                  'minimum_target_mv':float(brain.v[targets].min()),
                                  'mean_target_mv':float(brain.v[targets].mean()),
                                  'eye_spikes':int(brain.counts[eyes].sum())})
                assert not session.eyes.enabled and not brain.eye_inputs and session.eyes.video.ended
                assert abs(brain.time-body.time)<1e-9
                assert trace[13]['input_hz']['L']>0 and not trace[14]['input_hz']
                row={'seed':seed,'condition':condition,'initial_hz':trace[0]['input_hz']['L'],
                     'late_bright_hz':trace[13]['input_hz']['L'],
                     'after_dark_hz':trace[28]['input_hz']['L'],
                     'late_bright_minimum_target_mv':trace[13]['minimum_target_mv'],
                     'late_bright_mean_target_mv':trace[13]['mean_target_mv'],
                     'minimum_target_mv':min(t['minimum_target_mv'] for t in trace),
                     'late_bright_spikes':trace[13]['eye_spikes']-trace[11]['eye_spikes'],
                     'shared_clock':True,'eof_released':True}
                if condition=='static':
                    assert np.isclose(row['initial_hz'],row['late_bright_hz'])
                else:
                    assert row['initial_hz']>95 and 25<row['late_bright_hz']<28
                    assert row['after_dark_hz']>80
                if condition=='adaptive_blocked':assert row['minimum_target_mv']==-52.
                rows.append(row)
                reference[condition]=(row,brain.counts[eyes].copy(),brain.rng.bit_generator.state)
                (args.output_dir/f'{seed}-{condition}.json').write_text(json.dumps(trace,indent=2)+'\n')
                print(json.dumps(row),flush=True)
                session.eyes.close()
            static,adaptive=reference['static'][0],reference['adaptive'][0]
            assert adaptive['late_bright_spikes']<static['late_bright_spikes']
            assert adaptive['late_bright_mean_target_mv']>static['late_bright_mean_target_mv']
            np.testing.assert_array_equal(reference['adaptive'][1],reference['adaptive_blocked'][1])
            assert reference['adaptive'][2]==reference['adaptive_blocked'][2]
        report={'format':'nexus-light-adaptation-results-1','success':True,'trials':rows,
                'manifest_sha256':hashlib.sha256((pack/'manifest.json').read_bytes()).hexdigest(),
                'mapping':'spatial','parameters':{'tau_ms':250.,'strength':3.,'fitted':False},
                'limits':['Divisive input adaptation is an authored approximation, not a fitted retinal model.',
                          'No graded transmission, ON/OFF circuit reconstruction, chloride reversal, or object recognition.',
                          'Initial bright input can still over-hyperpolarize the unbounded-current neural model.']}
        (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:
        if session is not None:session.eyes.close()
        body.close()


if __name__=='__main__':main()
