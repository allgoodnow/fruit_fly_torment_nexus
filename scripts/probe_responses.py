"""Full brain/body experiments with pathway, decoder, and heat-mapping controls."""
import json
import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT/'.runtime/numba'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.runtime/matplotlib'))
import numpy as np
from nexus.brain.runtime import Brain, Connectome
from nexus.brain.circuits import circuit_ids
from nexus.brain.scenarios import scenario_protocol
from nexus.body import FlyBody
from nexus.coupled import CoupledSession


def main():
    out = ROOT/'runs/response_probe'
    out.mkdir(parents=True, exist_ok=True)
    brain = Brain(Connectome.load(ROOT/'data/brain-v630'))
    brain.advance(.0001)
    brain.reset()
    body = FlyBody(render=False)
    cases = [('baseline', None, None), ('threat', 'defensive', None),
             ('threat_outputs_blocked', 'defensive', 'looming'),
             ('aversion_proxy', 'aversion', None), ('heat40', 'heat', None),
             ('heat100', 'heat', None), ('heat_outputs_blocked', 'heat', 'warmth'),
             ('seizure_like', 'seizure', None), ('seizure_motor_blocked', 'seizure', 'motor'),
             ('seizure_bridge_blocked', 'seizure', 'bridge'),
             ('heat_overload', 'heat_overload', None)]
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', nargs='+', choices=[c[0] for c in cases])
    args = parser.parse_args()
    reports = ([r for r in json.loads((out/'report.json').read_text()) if r['name'] not in args.cases]
               if args.cases else [])
    if args.cases:
        cases = [case for case in cases if case[0] in args.cases]
    try:
        for name, scenario, block in cases:
            brain.reset()
            body.reset()
            session = CoupledSession(brain, body)
            celsius = 100 if name in ('heat100', 'heat_overload') else 40
            protocol = (scenario_protocol(scenario, stimulus_ms=500, celsius=celsius) if scenario else
                        {'format': 'nexus-protocol-1', 'duration_ms': 1100, 'events': []})
            if block == 'motor':
                session.command('motor_effects_enabled', False)
            elif block == 'bridge':
                session.command('bridge_enabled', False)
            elif block:
                protocol['events'].insert(-1, {'at_ms': 100, 'action': 'silence', 'ids': circuit_ids(block)})
                # Release restores outgoing gain, including for queued spikes.
                # Keep the causal block during the post-input window as well.
                protocol['events'].append({'at_ms': 600, 'action': 'silence', 'ids': circuit_ids(block)})
                protocol['events'].append({'at_ms': 1100, 'action': 'release'})
            session.command('protocol', protocol)
            start, trace = time.perf_counter(), []
            for _ in range(110):
                session.advance(100)
                pose = body.sim.mj_data.xmat[body.thorax_id].reshape(3, 3)
                trace.append({'time': brain.time, 'body_time': body.time,
                              'position_mm': body.position().tolist(), 'upright': float(pose[2, 2]),
                              'spikes': int(brain.counts.sum()), 'effects': session.motor_effects.output(),
                              'joint_offset_rms_rad': body.motor_offset_rms,
                              'gf_counts': brain.counts[session.motor_effects.gf].tolist()})
            row = {'name': name, 'protocol': protocol, 'wall_seconds': time.perf_counter()-start,
                   'total_spikes': int(brain.counts.sum()), 'giant_fiber_spikes': trace[-1]['gf_counts'],
                   'minimum_upright': min(t['upright'] for t in trace),
                   'minimum_height_mm': min(t['position_mm'][2] for t in trace),
                   'max_height_mm': max(t['position_mm'][2] for t in trace),
                   'max_disruption': max(t['effects']['disruption'] for t in trace),
                   'max_joint_offset_rms_rad': max(t['joint_offset_rms_rad'] for t in trace),
                   'final_position_mm': body.position().tolist(), 'events': list(brain.events), 'trace': trace}
            reports.append(row)
            (out/'report.json').write_text(json.dumps(reports, indent=2, allow_nan=False))
            print(json.dumps({k:v for k,v in row.items() if k not in ('protocol','events','trace')}), flush=True)
            assert session.protocol.completed and not session.running and brain.inhibition_gain == 1
            assert not brain.circuit_inputs and brain.nominal_temperature is None
            assert all(abs(t['time']-t['body_time']) < 1e-9 for t in trace)
            assert np.isfinite(body.sim.mj_data.qpos).all()
        r = {row['name']: row for row in reports}
        assert sum(r['threat']['giant_fiber_spikes']) > 0
        assert sum(r['threat_outputs_blocked']['giant_fiber_spikes']) == 0
        assert r['heat40']['total_spikes'] == r['heat100']['total_spikes']
        np.testing.assert_array_equal(r['heat40']['final_position_mm'], r['heat100']['final_position_mm'])
        assert r['seizure_like']['max_disruption'] > .1
        assert r['seizure_like']['max_joint_offset_rms_rad'] > .05
        assert r['seizure_motor_blocked']['max_joint_offset_rms_rad'] == 0
        assert r['seizure_like']['total_spikes'] == r['seizure_motor_blocked']['total_spikes'] == r['seizure_bridge_blocked']['total_spikes']
        np.testing.assert_array_equal(r['baseline']['final_position_mm'], r['seizure_bridge_blocked']['final_position_mm'])
        assert r['heat_overload']['max_disruption'] > .1
        # Overturned bodies must be supported by torso/head ground contacts.
        assert all(t['position_mm'][2] > .1 for row in reports for t in row['trace'])
        (out/'acceptance.json').write_text(json.dumps({'success': True, 'cases': list(r),
            'limits': ['CB0059 input is an inhibitory central candidate; no complete nociceptive pathway or aversive learning.',
                       'Heat conversion and motor effects are authored approximations, not biological damage or felt states.']}, indent=2))
    finally:
        body.close()


if __name__ == '__main__':
    main()
