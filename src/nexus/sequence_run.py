"""Unattended runs of the same coupled neural and 3D body simulation as the GUI."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from . import __version__
from .brain.protocol import Protocol


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def record_sequence(session, description, output, *, provenance=None, progress=None):
    """Record a complete run without resetting either model or modifying the input.

    Ten-millisecond observations are summaries, not a complete spike raster or a
    resumable checkpoint. Event times retain the model's 0.1 ms resolution.
    """
    Protocol(description, session.brain)  # Reject the entire plan before any mutation.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    brain, body = session.brain, session.body
    started = time.perf_counter()
    start_step = brain.step
    starting_counts = brain.counts.copy()
    earlier_events = tuple(brain.events)
    earlier_event_ids = {id(event) for event in earlier_events}
    report = {
        'format': 'nexus-sequence-run-1', 'version': __version__, 'success': False,
        'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
        'dataset': brain.graph.snapshot, 'model': brain.graph.model, 'seed': brain.seed,
        'neurons': len(brain.graph.ids), 'edges': len(brain.graph.posts),
        'origin_ms': start_step / 10, 'protocol': description,
        'protocol_canonical_sha256': hashlib.sha256(
            json.dumps(description, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'provenance': provenance or {}, 'sample_interval_ms': 10,
        'motor_bridge_enabled': session.decoder.enabled and session.motor_effects.enabled,
        'environment': session.environment.snapshot() if session.environment else None,
        'limitations': [
            'Experimental neural dynamics and authored motor decoder; subjective states are unverified.',
            'Completion and removal of inputs do not establish neural or behavioral recovery.',
            'Trace rows summarize intervals; counts.npz contains per-neuron counts, not every spike time.',
        ],
    }
    write_json(output/'report.json', report)
    previous_counts = starting_counts.copy()
    previous_step = start_step
    samples = 0
    try:
        session.command('resume_after_protocol', False)
        session.command('protocol', description)
        with (output/'trace.jsonl').open('x') as stream:
            while session.running:
                before_position = body.position()
                forward = body.sim.mj_data.xmat[body.thorax_id].reshape(3, 3)[:2, 0].copy()
                forward /= max(float(np.linalg.norm(forward)), 1e-9)
                session.advance(100)
                if (not np.isfinite(brain.v).all() or not np.isfinite(brain.g).all()
                        or not np.isfinite(body.sim.mj_data.qpos).all()
                        or not np.isfinite(body.sim.mj_data.qvel).all()):
                    raise RuntimeError('Non-finite neural or body state')
                if abs(brain.time-body.time) > 1e-9:
                    raise RuntimeError('Body and brain clocks diverged')
                delta = brain.counts-previous_counts
                seconds = (brain.step-previous_step)*.0001
                row = {
                    'from_ms': previous_step/10, 'to_ms': brain.step/10,
                    'spikes': int(delta.sum()), 'active_cells': int(np.count_nonzero(delta)),
                    'population_hz_per_neuron': float(delta.sum()/seconds/len(delta)),
                    'position_mm': body.position().tolist(),
                    'longitudinal_delta_mm': float(np.dot((body.position()-before_position)[:2], forward)),
                    'upright': float(body.sim.mj_data.xmat[body.thorax_id].reshape(3, 3)[2, 2]),
                    'joint_offset_rms_rad': body.motor_offset_rms,
                    'behavior': session.behavior.output(session.baseline),
                    'motor_effects': session.motor_effects.output(),
                    'motor_command': session.motor_output(session.behavior.output(session.baseline)),
                    'recovery': session.recovery.status(),
                }
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                previous_counts[:] = brain.counts
                previous_step = brain.step
                samples += 1
                if samples % 100 == 0:
                    stream.flush()
                    if progress:
                        progress({'elapsed_ms': (brain.step-start_step)/10,
                                  'duration_ms': description['duration_ms'],
                                  'spikes': int((brain.counts-starting_counts).sum())})
        if not session.completed_protocol or not session.completed_protocol.completed:
            raise RuntimeError('Sequence stopped before its endpoint')
        np.savez_compressed(output/'counts.npz', ids=brain.graph.ids,
                            start_counts=starting_counts, end_counts=brain.counts)
        report.update(success=True, status='completed', samples=samples,
                      elapsed_ms=(brain.step-start_step)/10,
                      total_spikes=int((brain.counts-starting_counts).sum()),
                      end_counts_sha256=hashlib.sha256(brain.counts.astype('<i8').tobytes()).hexdigest(),
                      final_position_mm=body.position().tolist(),
                      inputs_released=not bool(brain.inputs.size),
                      network_overlays_released=bool(brain.inhibition_gain == 1 and np.all(brain.output_gain == 1)),
                      events=[e for e in brain.events if id(e) not in earlier_event_ids],
                      recovery=session.recovery.snapshot())
    except BaseException as error:
        session.running = False
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error=f'{type(error).__name__}: {error}', samples=samples,
                      elapsed_ms=(brain.step-start_step)/10)
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        write_json(output/'report.json', report)
    return report


def run_file(protocol_path, output, *, dataset='male-cns', seed=73100, motor_bridge=True):
    from .body import FlyBody
    from .brain.config import default_pack, model_config
    from .brain.runtime import Brain, Connectome
    from .coupled import CoupledSession

    contents = Path(protocol_path).read_bytes()
    if len(contents) > 1_000_000:
        raise ValueError('Sequence file is too large')
    description = json.loads(contents)
    config = model_config(default_pack(dataset))
    brain = Brain(Connectome.load(config.directory, allow_experimental=config.experimental), seed=seed)
    Protocol(description, brain)
    manifest_bytes = (config.directory/'manifest.json').read_bytes()
    body = FlyBody(render=False)
    try:
        session = CoupledSession(brain, body, autonomous=True)
        session.command('bridge_enabled', motor_bridge)
        return record_sequence(session, description, output,
                               provenance={'pack_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
                                           'input_file_sha256': hashlib.sha256(contents).hexdigest()},
                               progress=lambda row: print(json.dumps(row), flush=True))
    finally:
        body.close()
