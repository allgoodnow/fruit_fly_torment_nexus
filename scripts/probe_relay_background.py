"""Paired full-network/body test of a tonic visual-relay drive hypothesis."""
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
import pandas as pd
from nexus.brain.runtime import Brain,Connectome
from nexus.body import FlyBody
from nexus.coupled import CoupledSession


def movie(path, levels):
    writer=imageio_ffmpeg.write_frames(str(path),(320,180),fps=20,codec='libx264rgb',pix_fmt_out='rgb24',quality=10,macro_block_size=1,ffmpeg_log_level='error')
    writer.send(None)
    for level in levels:writer.send(np.full((180,320,3),level,dtype=np.uint8))
    writer.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    movie(args.output_dir/'dark.mp4',[0]*18)
    movie(args.output_dir/'pulse.mp4',[0]*6+[128]*6+[0]*6)
    pack=ROOT/'data/brain-male-cns-v1.0-lif'
    graph=Connectome.load(pack,allow_experimental=True)
    annotation_path=ROOT/'data/raw/male-cns-v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather'
    annotation=pd.read_feather(annotation_path).set_index('bodyId').reindex(graph.ids)
    classes=annotation['superclass'].fillna('unknown').to_numpy()
    body=FlyBody(render=False);session=None;rows=[];comparisons=[]
    try:
        for seed in [73100,73101,73102]:
            runs={}
            for condition in ['baseline_dark','video','photoreceptors_blocked','relays_blocked']:
                brain=Brain(graph,seed=seed);body.reset()
                session=CoupledSession(brain,body,autonomous=False)
                session.command('relay_background',True)
                session.command('vision_video',str(args.output_dir/('pulse.mp4' if condition in ['video','photoreceptors_blocked'] else 'dark.mp4')))
                session.command('vision_mapping','spatial')
                session.command('eye_feedback',True)
                eyes=brain.resolve(brain.circuit_ids('eye_left')+brain.circuit_ids('eye_right'))
                relay=brain.background_targets
                downstream=np.ones(len(brain.v),bool);downstream[eyes]=False;downstream[relay]=False
                if condition=='photoreceptors_blocked':brain.silence(graph.ids[eyes].tolist())
                if condition=='relays_blocked':brain.silence(graph.ids[relay].tolist())
                phases=[];trace=[];previous=brain.counts.copy()
                for sample in range(19):
                    session.advance(500)
                    counts=brain.counts-previous;previous=brain.counts.copy()
                    trace.append({'time_ms':brain.step/10,'relay_spikes':int(counts[relay].sum()),'downstream_spikes':int(counts[downstream].sum()),'minimum_mv':float(brain.v.min())})
                    if sample in (5,11,17):phases.append((brain.counts.copy(),brain.v.copy()))
                assert abs(brain.time-body.time)<1e-9 and not brain.eye_inputs and not session.eyes.enabled
                row={'seed':seed,'condition':condition,'relay_cells':len(relay),'total_spikes':int(brain.counts.sum()),'downstream_spikes':int(brain.counts[downstream].sum()),'downstream_spiking_cells':int((brain.counts[downstream]>0).sum()),'minimum_sampled_mv':min(t['minimum_mv'] for t in trace),'shared_clock':True,'eof_released_visual_input':True,'baseline_remains_at_eof':brain.background_enabled}
                runs[condition]=(phases,brain.counts.copy(),brain.v.copy(),brain.rng.bit_generator.state)
                (args.output_dir/f'{seed}-{condition}.json').write_text(json.dumps(trace,indent=2)+'\n')
                rows.append(row);print(json.dumps(row),flush=True)
                before=brain.v.copy();session.command('brain',{'kind':'release'})
                assert not brain.background_enabled and not brain.eye_inputs
                np.testing.assert_array_equal(before,brain.v)
                session.eyes.close()
            base,video,blocked,relay_block=[runs[key] for key in ['baseline_dark','video','photoreceptors_blocked','relays_blocked']]
            excluded=np.ones(len(graph.ids),bool);excluded[eyes]=False
            np.testing.assert_array_equal(base[1][excluded],blocked[1][excluded])
            np.testing.assert_array_equal(base[2][excluded],blocked[2][excluded])
            assert video[3]==blocked[3]
            np.testing.assert_array_equal(video[1][eyes],blocked[1][eyes])
            assert not relay_block[1][downstream].any()
            # Compare the same 300-600 ms window, not a different baseline period.
            base_window=base[0][1][0]-base[0][0][0]
            video_window=video[0][1][0]-video[0][0][0]
            changed=(base_window!=video_window)&downstream
            assert base_window[relay].sum()>video_window[relay].sum()
            assert base_window[downstream].sum()>video_window[downstream].sum()
            assert changed.any()
            comparison={'seed':seed,'window_ms':[300,600],'baseline_relay_spikes':int(base_window[relay].sum()),'video_relay_spikes':int(video_window[relay].sum()),'baseline_downstream_spikes':int(base_window[downstream].sum()),'video_downstream_spikes':int(video_window[downstream].sum()),'downstream_cells_with_changed_spike_count':int(changed.sum()),'changed_cells_by_superclass':{str(c):int((changed&(classes==c)).sum()) for c in sorted(set(classes[changed]))},'photoreceptor_block_restores_nonphotoreceptor_state_exactly':True,'photoreceptor_block_preserves_input_spikes_and_rng':True,'relay_block_removes_downstream_spikes':True}
            comparisons.append(comparison);print(json.dumps(comparison),flush=True)
        report={'format':'nexus-relay-background-results-1','success':True,'trials':rows,'comparisons':comparisons,'manifest_sha256':hashlib.sha256((pack/'manifest.json').read_bytes()).hexdigest(),'annotation_sha256':hashlib.sha256(annotation_path.read_bytes()).hexdigest(),'parameters':{'bias_mv_range':[8,10],'fitted':False,'mapping':'spatial','light_adaptation':False,'input_gray_rgb':128,'trial_ms':950},'limits':['Tonic drive is applied to negative-weight R1-R6 targets only, not the whole brain.','Relay neurons still use a spiking proxy; no graded release or reversal potentials were added.','These are causal routing checks, not biological calibration, visual recognition or spontaneous exploration.']}
        (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:
        if session is not None:session.eyes.close()
        body.close()


if __name__=='__main__':main()
