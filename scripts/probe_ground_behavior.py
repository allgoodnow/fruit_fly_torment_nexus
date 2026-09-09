"""Physical ground-behavior trials, including real-pathway interruption and ablation."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT/'.runtime/numba'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.runtime/matplotlib'))
import numpy as np
from nexus.body import FlyBody
from nexus.brain.runtime import Brain, Connectome
from nexus.coupled import CoupledSession


def main():
    out = ROOT/'runs/ground_behavior_probe'
    out.mkdir(parents=True, exist_ok=True)
    brain = Brain(Connectome.load(ROOT/'data/brain-v630'))
    body = FlyBody(render=False)
    reports = []
    try:
        for name in ['free', 'threat', 'bridge_blocked']:
            brain.reset()
            body.reset()
            s = CoupledSession(brain, body, autonomous=True)
            if name == 'bridge_blocked':
                s.command('bridge_enabled', False)
            events = [] if name == 'free' else [
                {'at_ms': 1750, 'action': 'circuit', 'name': 'looming', 'rate_hz': 200},
                {'at_ms': 2000, 'action': 'release'}]
            s.command('protocol', {'format': 'nexus-protocol-1', 'duration_ms': 2200, 'events': events})
            trace = []
            for _ in range(400):
                s.advance(100)
                trace.append({'time': brain.time, 'body_time': body.time,
                              'behavior': s.behavior.output(1.), 'position_mm': body.position().tolist(),
                              'upright': float(body.sim.mj_data.xmat[body.thorax_id].reshape(3,3)[2,2]),
                              'joint_offset': body.motor_offset_rms,
                              'spikes': int(brain.counts.sum())})
            assert brain.time == 4. and s.running and s.protocol is None
            assert all(abs(t['time']-t['body_time']) < 1e-9 for t in trace)
            assert all(t['position_mm'][2] > .1 for t in trace)
            assert np.isfinite(body.sim.mj_data.qpos).all()
            states = {t['behavior']['state'] for t in trace}
            assert {'exploring', 'turning', 'resting'} <= states
            row = {'case': name, 'states': sorted(states), 'spikes': int(brain.counts.sum()),
                   'minimum_upright': min(t['upright'] for t in trace),
                   'final_position_mm': body.position().tolist(), 'trace': trace}
            reports.append(row)
            (out/'report.json').write_text(json.dumps(reports, indent=2, allow_nan=False))
            print({k:v for k,v in row.items() if k != 'trace'}, flush=True)
        free, threat, blocked = reports
        assert free['minimum_upright'] > .8
        assert 'interrupted' in threat['states'] and 'interrupted' not in blocked['states']
        assert threat['spikes'] == blocked['spikes'] > 0
        np.testing.assert_array_equal(free['final_position_mm'], blocked['final_position_mm'])
        (out/'acceptance.json').write_text(json.dumps({'success': True,
            'checks': ['upright free exploration and rest', 'shared clocks', 'no reset at protocol end',
                       'neural interruption', 'motor bridge ablation preserves spikes and restores free trajectory']}, indent=2))
    finally:
        body.close()


if __name__ == '__main__':
    main()
