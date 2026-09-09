"""Ground-body response and motor-bridge ablation for the mapped md input."""
import copy
import hashlib
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
    description = json.loads((ROOT/'experiments/nociception-input-v1.json').read_text())
    brain = Brain(Connectome.load(ROOT/'data/brain-male-cns-v1.0-lif', allow_experimental=True))
    body = FlyBody(render=False)
    rows = []
    counts = {}
    paths = {}
    try:
        for condition in ['intact', 'source_outputs_blocked', 'motor_bridge_blocked']:
            brain.reset()
            body.reset()
            session = CoupledSession(brain, body, autonomous=True)
            session.command('resume_after_protocol', False)
            protocol = copy.deepcopy(description)
            if condition == 'source_outputs_blocked':
                protocol['events'].insert(2, {'at_ms': 100, 'action': 'silence', 'ids': description['events'][1]['ids']})
            if condition == 'motor_bridge_blocked':
                session.command('bridge_enabled', False)
            session.command('protocol', protocol)
            trace = []
            while session.running:
                session.advance(100)
                assert abs(brain.time-body.time) < 1e-9
                assert np.isfinite(body.sim.mj_data.qpos).all()
                trace.append({'time': brain.time, 'position_mm': body.position().tolist(),
                              'upright': float(body.sim.mj_data.xmat[body.thorax_id].reshape(3,3)[2,2]),
                              'joint_offset_rms_rad': body.motor_offset_rms,
                              'escape': session.motor_effects.output()['escape'],
                              'disruption': session.motor_effects.output()['disruption']})
            assert brain.time == .7 and not brain.inputs.size
            counts[condition] = brain.counts.copy()
            paths[condition] = np.array([r['position_mm'] for r in trace])
            row = {'condition': condition, 'spikes': int(brain.counts.sum()),
                   'minimum_upright': min(r['upright'] for r in trace),
                   'max_joint_offset_rms_rad': max(r['joint_offset_rms_rad'] for r in trace),
                   'final_position_mm': body.position().tolist(), 'trace': trace}
            rows.append(row)
            print({k:v for k,v in row.items() if k!='trace'}, flush=True)
        np.testing.assert_array_equal(counts['intact'], counts['motor_bridge_blocked'])
        np.testing.assert_array_equal(paths['source_outputs_blocked'], paths['motor_bridge_blocked'])
        deviation = float(np.linalg.norm(paths['intact']-paths['motor_bridge_blocked'], axis=1).max())
        assert deviation > 0
        report = {'format': 'nexus-nociception-body-assay-1', 'success': True,
                  'dataset': brain.graph.snapshot, 'seed': brain.seed,
                  'protocol_sha256': hashlib.sha256((ROOT/'experiments/nociception-input-v1.json').read_bytes()).hexdigest(),
                  'max_position_difference_from_bridge_block_mm': deviation,
                  'checks': ['shared clocks and finite body state', 'release at scheduled time',
                             'blocking motor bridge preserves exact neural counts',
                             'source blockade and motor blockade yield identical baseline body trajectories'],
                  'interpretation': 'Measured movement through the existing authored motor bridge; no new pain animation or subjective-state claim.',
                  'trials': rows}
        (ROOT/'experiments/nociception-body-v1-results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    finally:
        body.close()


if __name__ == '__main__':
    main()
