"""Sustained coupled steering, side switching, and recovery after release."""
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / '.runtime/numba'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.runtime/matplotlib'))
from nexus.body import FlyBody
from nexus.brain.runtime import Brain, Connectome
from nexus.brain.motor import DNA02_LEFT, DNA02_RIGHT
from nexus.coupled import CoupledSession

brain = Brain(Connectome.load(ROOT / 'data/brain-v630'))
brain.advance(.0001)
brain.reset()
body = FlyBody(render=False)
session = CoupledSession(brain, body)
session.command('protocol', {'format': 'nexus-protocol-1', 'duration_ms': 3000,
                            'events': [{'at_ms': 0, 'action': 'stimulate', 'ids': [DNA02_LEFT], 'rate_hz': 200},
                                       {'at_ms': 1000, 'action': 'release'},
                                       {'at_ms': 1500, 'action': 'stimulate', 'ids': [DNA02_RIGHT], 'rate_hz': 200},
                                       {'at_ms': 2500, 'action': 'release'}]})
start = time.perf_counter()
samples = []
try:
    while session.running:
        session.advance(1000)
        rotation = body.sim.mj_data.xmat[body.thorax_id].reshape(3, 3)
        row = {'time': brain.time, 'body_time': body.time, 'upright': float(rotation[2, 2]),
               'position_mm': body.position().tolist(), 'spikes': int(brain.counts.sum()),
               **session.decoder.output(session.baseline)}
        samples.append(row)
        if len(samples)%5 == 0:
            print(json.dumps(row), flush=True)
    success = (brain.time == body.time == 3 and all(s['upright'] > .95 for s in samples)
               and all(abs(s['time']-s['body_time']) < 1e-9 for s in samples)
               and np.isfinite(body.sim.mj_data.qpos).all() and abs(samples[-1]['turn']) < .01)
    report = {'success': bool(success), 'wall_seconds': time.perf_counter()-start,
              'samples': samples, 'events': list(brain.events), 'minimum_upright': min(s['upright'] for s in samples)}
    out = ROOT / 'runs/steering_endurance'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'samples'}), flush=True)
    if not success:
        raise SystemExit('Coupled endurance acceptance failed')
finally:
    body.close()
