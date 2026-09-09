"""Real-body food crossings with sensory-input and outgoing-silencing controls."""
import json
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
from nexus.brain.targets import SUGAR, MN9
from nexus.coupled import CoupledSession
from nexus.environment import FoodEnvironment

brain = Brain(Connectome.load(ROOT / 'data/brain-v630'))
brain.advance(.0001)
brain.reset()
body = FlyBody(render=False)
world = FoodEnvironment()
session = CoupledSession(brain, body, environment=world)
out = ROOT / 'runs/food_probe'
out.mkdir(parents=True, exist_ok=True)
reports = []
try:
    for name, enabled, present, silence in [('food', True, True, False),
                                           ('feedback_blocked', False, True, False),
                                           ('food_removed', True, False, False),
                                           ('sugar_outputs_silenced', True, True, True)]:
        session.command('food_config', {'enabled': enabled, 'present': present})
        session.command('reset')
        if silence:
            brain.silence(SUGAR)
        start = time.perf_counter()
        trace = []
        for _ in range(120):
            session.advance(100)
            trace.append({'time': brain.time, 'body_time': body.time, 'contact': world.contact,
                          'active': world.active, 'sensory_cells': len(brain.sensory_inputs),
                          'mn9_spikes': int(brain.counts[brain.lookup[int(MN9)]]),
                          'upright': float(body.sim.mj_data.xmat[body.thorax_id].reshape(3, 3)[2, 2]),
                          'position_mm': body.position().tolist()})
        row = {'name': name, 'wall_seconds': time.perf_counter()-start,
               'active_seconds': world.active_seconds, 'spikes': int(brain.counts.sum()),
               'mn9_spikes': trace[-1]['mn9_spikes'], 'contact_seen': any(t['contact'] for t in trace),
               'final_sensory_cells': len(brain.sensory_inputs), 'minimum_upright': min(t['upright'] for t in trace),
               'events': list(world.events), 'trace': trace}
        reports.append(row)
        (out / 'report.json').write_text(json.dumps(reports, indent=2))
        print(json.dumps({k:v for k,v in row.items() if k not in ('trace','events')}), flush=True)
    a, blocked, removed, silenced = reports
    assert a['contact_seen'] and a['active_seconds'] > .1 and a['mn9_spikes'] > 0
    assert a['final_sensory_cells'] == 0
    assert blocked['contact_seen'] and blocked['spikes'] == 0
    assert not removed['contact_seen'] and removed['spikes'] == 0
    assert silenced['contact_seen'] and silenced['spikes'] > 0 and silenced['mn9_spikes'] == 0
    assert all(r['minimum_upright'] > .95 for r in reports)
    assert all(abs(t['time']-t['body_time']) < 1e-9 for r in reports for t in r['trace'])
    (out / 'acceptance.json').write_text(json.dumps({'success': True, 'conditions': [r['name'] for r in reports]}, indent=2))
    print('Food crossing and causal controls passed.', flush=True)
finally:
    body.close()
