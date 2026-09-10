"""Sixty simulated seconds: repeated inputs, explicit posture assistance and resets."""
import copy, hashlib, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
for name,path in [('NUMBA_CACHE_DIR','.runtime/numba'),('MPLCONFIGDIR','.runtime/matplotlib'),('FLYGYM_ASSET_CACHE_DIR','.runtime/assets')]:
    os.environ.setdefault(name,str(ROOT/path))
import numpy as np
from nexus import __version__
from nexus.body import FlyBody
from nexus.brain.runtime import Brain,Connectome
from nexus.brain.scenarios import sequence_protocol
from nexus.coupled import CoupledSession


def rss():
    return int(next(x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')).split()[1])*1024


def neural_state(b):
    h=hashlib.sha256()
    for field in ['v','g','counts','queue','queue_size','inputs','rates','output_gain','last_spike','refractory']:
        h.update(getattr(b,field).tobytes())
    h.update(json.dumps(b.rng.bit_generator.state,sort_keys=True).encode())
    h.update(str((b.step,b.inhibition_gain,b.nominal_temperature)).encode())
    return h.hexdigest()


def main():
    output=ROOT/'runs/final_endurance';output.mkdir(parents=True,exist_ok=False)
    graph=Connectome.load(ROOT/'data/brain-male-cns-v1.0-lif',allow_experimental=True)
    b=Brain(graph);body=FlyBody(render=False);s=CoupledSession(b,body,autonomous=True)
    s.command('resume_after_protocol',False)
    plan=sequence_protocol([{'name':name,'baseline_ms':100,'stimulus_ms':300,'recovery_ms':600}
                            for _ in range(5) for name in ['defensive','aversion','heat','seizure','heat_overload']],
                           dataset=graph.snapshot,pain_circuit='nociception_proxy')
    plan['duration_ms']=48000
    (output/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    started=time.perf_counter();assist=[];rows=[];reset_counts=[]
    try:
        s.command('protocol',plan)
        with (output/'trace.jsonl').open('x') as stream:
            while s.running:
                s.advance(1000)
                assert abs(b.time-body.time)<1e-9
                assert all(np.isfinite(x).all() for x in [b.v,b.g,body.sim.mj_data.qpos,body.sim.mj_data.qvel])
                packet=s.snapshot()
                row={'sim_seconds':b.time,'rss_bytes':rss(),'upright':body.upright(),
                     'spikes':int(b.counts.sum()),'population_hz_per_neuron':packet['brain']['population_hz_per_neuron'],
                     'recovery':s.recovery.status(),'inputs':len(b.inputs),'history':len(b.history),'path':len(body.path)}
                rows.append(row);stream.write(json.dumps(row)+'\n')
                if b.step%100000==0:
                    before=neural_state(b);cursor=s.protocol.cursor;clock=b.time;position=body.position()[:2].copy()
                    s.command('reposition_body')
                    assert not s.running and b.time==body.time==clock and neural_state(b)==before
                    assert s.protocol.cursor==cursor and body.upright()>.95
                    np.testing.assert_allclose(body.position()[:2],position,atol=1e-9)
                    assist.append({'at_ms':b.step/10,'neural_state_preserved':True,'upright_after':body.upright()})
                    s.command('running',True)
                if b.step%50000==0:
                    print(json.dumps({'sim_seconds':b.time,'wall_seconds':round(time.perf_counter()-started,1),'rss_mib':round(rss()/2**20,1),'upright':body.upright()}),flush=True)
        assert b.time==48 and not b.inputs.size and b.inhibition_gain==1
        recovery_before_reset=s.recovery.snapshot()
        isolated=json.loads((ROOT/'experiments/male-cns-nociception-recovery.json').read_text())
        for cycle in range(5):
            s.command('reset')
            assert b.time==body.time==0 and int(b.counts.sum())==0 and not b.inputs.size and s.reposition_count==0
            s.command('protocol',isolated)
            while s.running:s.advance(1000)
            assert b.time==body.time==2.4 and body.upright()>.95 and not b.inputs.size
            digest=hashlib.sha256(b.counts.tobytes()).hexdigest();reset_counts.append(digest)
            print(json.dumps({'reset_cycle':cycle+1,'spikes':int(b.counts.sum()),'upright':body.upright(),'rss_mib':round(rss()/2**20,1)}),flush=True)
        assert len(set(reset_counts))==1
        warm=[x['rss_bytes'] for x in rows if x['sim_seconds']>=5]
        growth=warm[-1]-warm[0];assert growth<128*2**20
        assert len(body.path)<=1200 and len(b.history)<=b.history.maxlen and len(s.recovery.episodes)<=128
        report={'version':__version__,'success':True,'simulated_seconds':60,'continuous_seconds':48,
                'repeated_stimulations':25,'reset_cycles':5,'identical_neural_counts_after_every_reset':True,
                'manual_repositions':assist,'rss_growth_after_warmup_bytes':growth,'rss_peak_bytes':max(warm),
                'wall_seconds':time.perf_counter()-started,'post_release_before_reset':recovery_before_reset,
                'final_population_hz_per_neuron_before_reset':rows[-1]['population_hz_per_neuron'],
                'limitations':['One neural seed on this host; not an hours-long or cross-hardware endurance certification.',
                               'Posture assistance is explicit and does not normalize persistent neural activity.']}
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='post_release_before_reset'}),flush=True)
    finally:body.close()

if __name__=='__main__':main()
