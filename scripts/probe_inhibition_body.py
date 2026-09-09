"""Shared-clock perturbation, decoder ablation, coupling sensitivity and recovery."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / '.runtime/numba'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.runtime/matplotlib'))
import numpy as np
from nexus.body import FlyBody
from nexus.brain.runtime import Brain, Connectome
from nexus.brain.targets import SUGAR
from nexus.coupled import CoupledSession


def main():
    out = ROOT / 'runs/inhibition_body_probe'
    out.mkdir(parents=True, exist_ok=True)
    brain = Brain(Connectome.load(ROOT / 'data/brain-v630'))
    brain.advance(.0001)
    brain.reset()
    body = FlyBody(render=False)
    reports = []
    try:
        for name, gain, enabled, interval in [('sham', 1., True, 100),
                                              ('reduced', .25, True, 100),
                                              ('decoder_blocked', .25, False, 100),
                                              ('coupling_5ms', .25, True, 50)]:
            brain.reset()
            body.reset()
            session = CoupledSession(brain, body, coupling_ticks=interval)
            session.decoder.enabled = enabled
            session.command('protocol', {'format': 'nexus-protocol-1', 'duration_ms': 1400,
                'events': [{'at_ms': 100, 'action': 'inhibition_gain', 'gain': gain},
                           {'at_ms': 100, 'action': 'stimulate', 'ids': SUGAR, 'rate_hz': 200},
                           {'at_ms': 400, 'action': 'release'}]})
            start, trace = time.perf_counter(), []
            previous = 0
            for _ in range(140):
                session.advance(100)
                total = int(brain.counts.sum())
                pose = body.sim.mj_data.xmat[body.thorax_id].reshape(3, 3)
                trace.append({'time': brain.time, 'body_time': body.time, 'spikes': total,
                              'interval_spikes': total-previous, 'inhibition_gain': brain.inhibition_gain,
                              'position_mm': body.position().tolist(), 'upright': float(pose[2, 2]),
                              'yaw': float(np.arctan2(pose[1, 0], pose[0, 0])),
                              'motor': session.decoder.output(session.baseline)})
                previous = total
            row = {'name': name, 'wall_seconds': time.perf_counter()-start,
                   'total_spikes': total, 'last_100ms_spikes': sum(t['interval_spikes'] for t in trace[-10:]),
                   'minimum_upright': min(t['upright'] for t in trace), 'final_yaw': trace[-1]['yaw'],
                   'final_position_mm': trace[-1]['position_mm'], 'trace': trace}
            reports.append(row)
            (out / 'report.json').write_text(json.dumps(reports, indent=2, allow_nan=False))
            assert session.protocol.completed and not session.running and brain.inhibition_gain == 1
            assert all(abs(t['time']-t['body_time']) < 1e-9 for t in trace)
            assert np.isfinite(body.sim.mj_data.qpos).all()
            print(json.dumps({k:v for k,v in row.items() if k != 'trace'}), flush=True)
        sham, active, blocked, shorter = reports
        assert active['total_spikes'] == blocked['total_spikes'] == shorter['total_spikes']
        assert all(t['motor']['left_drive'] == t['motor']['right_drive'] for t in blocked['trace'])
        assert any(abs(t['motor']['left_drive']-t['motor']['right_drive']) > .01 for t in active['trace'])
        assert not np.allclose(active['final_position_mm'], blocked['final_position_mm'], atol=1e-3)
        (out / 'acceptance.json').write_text(json.dumps({'success': True,
            'checks': ['shared clocks', 'finite physics', 'release restores gain',
                       'decoder ablation preserves neural counts and removes motor bias',
                       'coupling interval preserves neural counts']}, indent=2))
    finally:
        body.close()


if __name__ == '__main__':
    main()
