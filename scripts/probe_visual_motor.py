"""Measure a prepared visual response through the actual brain/body controller.

The explicit walking-input trials are positive controls, not visual responses.
No gain or neural parameter is fitted by this experiment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
for name, folder in [('NUMBA_CACHE_DIR', 'numba'), ('MPLCONFIGDIR', 'matplotlib'),
                     ('FLYGYM_ASSET_CACHE_DIR', 'assets')]:
    os.environ.setdefault(name, str(ROOT/'.runtime'/folder))
import numpy as np

from nexus.body import FlyBody
from nexus.brain.runtime import Brain, Connectome
from nexus.coupled import CoupledSession
from nexus.receptor_playback import file_hash
from nexus.sequence_run import write_json


def trial(graph, body, args, condition):
    body.reset()
    brain = Brain(graph)
    session = CoupledSession(brain, body)
    session.command('neural_walking', True)
    session.command('graded_relays', True)
    session.command('vision_video', str(args.video))
    rows = []
    try:
        session.command('vision_receptors', {'report_path': str(args.response/'report.json'),
                                           'curve_path': str(args.release_curve)})
        recording = session.eyes.receptors
        if recording.end < 5000:
            raise ValueError('Use at least 500 ms of recorded brain time for the motor check')
        session.command('eye_feedback', True)
        if condition == 'receptor_output_blocked':
            session.command('silence', {'ids': recording.ids})
        if condition == 'walking_input_motor_blocked':
            session.command('bridge_enabled', False)
        session.command('running', True)
        motor_input = condition in ('walking_input', 'walking_input_motor_blocked')
        while not recording.ended:
            # A separate, declared 100–400 ms positive control, never enabled
            # automatically by a video or installed as a behavioral fallback.
            if motor_input and brain.step == 1000:
                session.command('stimulate', {'ids': session.walking_decoder.ids, 'rate_hz': 100.})
            elif motor_input and brain.step == 4000:
                # Release imposed neural input without pausing the externally
                # supplied receptor recording (the GUI's Release pauses both).
                brain.release()
            session.advance(min(100, recording.end-brain.step))
            if not session.eyes.enabled and not recording.ended:
                raise RuntimeError(session.eyes.error or 'Visual playback stopped early')
            behavior = session.ground_output()
            rows.append({'time_ms': brain.step/10, 'readouts': session.motor_readouts(),
                         'walking': session.walking_decoder.output(session.baseline),
                         'effects': session.motor_effects.output(),
                         'command': session.motor_output(behavior), 'resting': behavior['resting'],
                         'position_mm': body.position().tolist(), 'upright': body.upright(),
                         'physics_qpos': body.sim.mj_data.qpos.copy().tolist()})
        with (args.output_dir/f'{condition}.jsonl').open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
        groups = {}
        for name, values in rows[-1]['readouts'].items():
            voltage = np.asarray([row['readouts'][name]['voltage_mv'] for row in rows])
            groups[name] = {**values, 'sampled_min_voltage_mv': voltage.min(axis=0).tolist(),
                            'sampled_max_voltage_mv': voltage.max(axis=0).tolist()}
        positions = np.asarray([row['position_mm'] for row in rows])
        summary = {'duration_ms': brain.step/10, 'body_time_ms': body.time*1000,
                   'receptor_cells': len(recording.ids), 'video_start_ms': recording.video_start/10,
                   'readouts': groups,
                   'maximum_forward_drive': max(row['walking']['drive'] for row in rows),
                   'maximum_absolute_turn': max(abs(row['command']['turn']) for row in rows),
                   'maximum_retreat': max(row['effects']['retreat'] for row in rows),
                   'maximum_escape': max(row['effects']['escape'] for row in rows),
                   'maximum_disruption': max(row['effects']['disruption'] for row in rows),
                   'resting_intervals': sum(row['resting'] for row in rows), 'intervals': len(rows),
                   'final_displacement_mm': float(np.linalg.norm((positions[-1]-body.origin)[:2])),
                   'minimum_upright': min(row['upright'] for row in rows),
                   'voltage_sha256': hashlib.sha256(brain.v.astype('<f8').tobytes()).hexdigest(),
                   'spike_counts_sha256': hashlib.sha256(brain.counts.astype('<i8').tobytes()).hexdigest(),
                   'walking_input': {'rate_hz': 100., 'start_ms': 100, 'end_ms': 400} if motor_input else None}
        return summary, rows
    finally:
        session.eyes.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--response', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--release-curve', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pack', type=Path, default=ROOT/'data/brain-male-cns-v1.0-lif')
    args = parser.parse_args()
    graph = Connectome.load(args.pack, allow_experimental=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    body = FlyBody(render=False)
    conditions, traces = {}, {}
    started = time.perf_counter()
    try:
        for condition in ('recording', 'receptor_output_blocked', 'walking_input', 'walking_input_motor_blocked'):
            conditions[condition], traces[condition] = trial(graph, body, args, condition)
            print(json.dumps({'condition': condition, 'maximum_forward_drive': conditions[condition]['maximum_forward_drive'],
                              'displacement_mm': conditions[condition]['final_displacement_mm']}), flush=True)
    finally:
        body.close()
    active, blocked = conditions['walking_input'], conditions['walking_input_motor_blocked']
    body_difference = float(np.max(np.linalg.norm(
        np.asarray([row['position_mm'] for row in traces['walking_input']])-
        np.asarray([row['position_mm'] for row in traces['walking_input_motor_blocked']]), axis=1)))
    checks = {'shared_clock': all(abs(c['duration_ms']-c['body_time_ms']) < 1e-6 for c in conditions.values()),
              'walking_input_generates_drive': active['maximum_forward_drive'] > 0,
              'motor_blockade_removes_drive': blocked['maximum_forward_drive'] == 0,
              'motor_blockade_preserves_neural_state': all(active[k] == blocked[k]
                                                         for k in ('voltage_sha256', 'spike_counts_sha256')),
              'walking_input_moves_body_through_controller': body_difference > .01}
    result = {'format': 'nexus-visual-motor-check-1', 'success': all(checks.values()),
              'dataset': graph.snapshot, 'neurons': len(graph.ids), 'edges': len(graph.posts),
              'sample_interval_ms': 10, 'conditions': conditions, 'checks': checks,
              'positive_control_maximum_body_difference_mm': body_difference,
              'wall_seconds': time.perf_counter()-started,
              'input_sha256': {name: file_hash(path) for name, path in
                               [('report', args.response/'report.json'), ('response', args.response/'response.npz'),
                                ('video', args.video), ('release_curve', args.release_curve),
                                ('pack_manifest', args.pack/'manifest.json')]},
              'limits': ['Movement is not a required pass condition for visual input.',
                         'Walking input is a separate positive control, not evidence of visual locomotion.',
                         'Receptor-output blockade includes tonic output; it does not isolate light from darkness.',
                         'Readout voltages and conductances are sampled every 10 ms, not continuously.',
                         'Neural parameters and controller gains remain unfitted; no natural behavior is established.']}
    write_json(args.output_dir/'report.json', result)
    if not result['success']:
        raise RuntimeError('Motor integration check failed; inspect report.json')


if __name__ == '__main__':
    main()
