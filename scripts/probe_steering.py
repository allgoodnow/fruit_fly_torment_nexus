"""Measure causal neural steering against matched no-input/blocked controls."""
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / '.runtime/numba'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.runtime/matplotlib'))
from nexus.body import FlyBody
from nexus.brain.runtime import Brain, Connectome
from nexus.brain.motor import DNA02_LEFT, DNA02_RIGHT
from nexus.coupled import CoupledSession


def main():
    brain = Brain(Connectome.load(ROOT / 'data/brain-v630'))
    brain.advance(.0001)
    brain.reset()
    body = FlyBody(render=False)
    session = CoupledSession(brain, body)
    out = ROOT / 'runs/steering_probe'
    out.mkdir(parents=True, exist_ok=True)
    result = []
    try:
        for name, target, blocked, silenced, interval in [
            ('baseline', None, False, False, 100),
            ('left', DNA02_LEFT, False, False, 100),
            ('right', DNA02_RIGHT, False, False, 100),
            ('left_bridge_blocked', DNA02_LEFT, True, False, 100),
            ('left_output_silenced', DNA02_LEFT, False, True, 100),
            ('left_5ms', DNA02_LEFT, False, False, 50),
        ]:
            session.command('reset')
            session.coupling_ticks = interval
            session.decoder.enabled = not blocked
            if target:
                brain.stimulate([target], 200)
            if silenced:
                brain.silence([target])
            angles, trace = [], []
            start = time.perf_counter()
            for _ in range(60):
                session.advance(100)
                rotation = body.sim.mj_data.xmat[body.thorax_id].reshape(3, 3)
                angles.append(math.atan2(rotation[1, 0], rotation[0, 0]))
                trace.append({'time': brain.time, 'position_mm': body.position().tolist(),
                              'upright': float(rotation[2, 2]), **session.decoder.output(session.baseline)})
            import numpy as np
            row = {'name': name, 'brain_time': brain.time, 'body_time': body.time,
                   'wall_seconds': time.perf_counter()-start,
                   'yaw_change_rad': float(np.unwrap(angles)[-1]-np.unwrap(angles)[0]),
                   'position_mm': body.position().tolist(), 'spikes': int(brain.counts.sum()),
                   'minimum_upright': min(t['upright'] for t in trace),
                   'target_spikes': brain.counts[session.decoder.indices].tolist(), 'trace': trace}
            result.append(row)
            print(json.dumps({k:v for k,v in row.items() if k != 'trace'}), flush=True)
            (out / 'report.json').write_text(json.dumps(result, indent=2))
    finally:
        body.close()


if __name__ == '__main__':
    main()
