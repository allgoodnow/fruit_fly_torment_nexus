"""MP4 -> R1-R6 -> scanned synapses, plus real eye-camera/body sampling."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
for name, folder in [('NUMBA_CACHE_DIR', 'numba'), ('MPLCONFIGDIR', 'matplotlib'), ('FLYGYM_ASSET_CACHE_DIR', 'assets')]:
    os.environ.setdefault(name, str(ROOT / '.runtime' / folder))
import imageio_ffmpeg
import numpy as np
from nexus.brain.runtime import Brain, Connectome
from nexus.body import FlyBody
from nexus.coupled import CoupledSession


def make_movie(path):
    writer = imageio_ffmpeg.write_frames(str(path), (320, 180), fps=20, codec='libx264', quality=10,
                                        macro_block_size=1, ffmpeg_log_level='error')
    writer.send(None)
    for intensity in [0, 0, 255, 255, 0, 0]:
        writer.send(np.full((180, 320, 3), intensity, dtype=np.uint8))
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    movie = args.output_dir / 'black-white.mp4'
    make_movie(movie)
    pack = ROOT / 'data/brain-male-cns-v1.0-lif'
    graph = Connectome.load(pack, allow_experimental=True)
    body = FlyBody(render=False)
    rows = []
    session = None
    try:
        for seed in [73100, 73101, 73102]:
            trials = {}
            for condition in ['video', 'photoreceptor_blocked', 'input_off']:
                brain = Brain(graph, seed=seed)
                body.reset()
                session = CoupledSession(brain, body, autonomous=False)
                session.command('vision_video', str(movie))
                eye_ids = brain.circuit_ids('eye_left') + brain.circuit_ids('eye_right')
                eyes = brain.resolve(eye_ids)
                target_parts = [graph.posts[graph.offsets[i]:graph.offsets[i+1]][
                    graph.weights[graph.offsets[i]:graph.offsets[i+1]] < 0] for i in eyes]
                targets = np.unique(np.concatenate(target_parts))
                if condition != 'input_off':
                    session.command('eye_feedback', True)
                if condition == 'photoreceptor_blocked':
                    brain.silence(eye_ids)
                trace = []
                for _ in range(35):
                    session.advance(100)
                    trace.append({'time_ms': brain.step / 10, 'vision': session.eyes.snapshot(),
                                  'eye_spikes': int(brain.counts[eyes].sum()),
                                  'minimum_target_mv': float(brain.v[targets].min()),
                                  'position_mm': body.position().tolist()})
                row = {'seed': seed, 'condition': condition,
                       'black_baseline_eye_spikes': trace[9]['eye_spikes'],
                       'white_phase_eye_spikes': trace[19]['eye_spikes'] - trace[9]['eye_spikes'],
                       'minimum_target_mv': min(t['minimum_target_mv'] for t in trace),
                       'total_spikes': int(brain.counts.sum()),
                       'ended': bool(session.eyes.video.ended), 'released': not brain.eye_inputs,
                       'shared_clock': abs(body.time - brain.time) < 1e-9}
                assert row['black_baseline_eye_spikes'] == 0 and row['shared_clock']
                assert row['released']
                if condition != 'input_off':
                    assert row['white_phase_eye_spikes'] > 0 and row['ended']
                else:
                    assert row['total_spikes'] == 0
                if condition == 'video':
                    assert row['minimum_target_mv'] < -52.1
                else:
                    assert row['minimum_target_mv'] == -52.
                (args.output_dir / f'{seed}-{condition}.json').write_text(json.dumps(trace, indent=2) + '\n')
                np.savez_compressed(args.output_dir / f'{seed}-{condition}.npz', counts=brain.counts, voltage=brain.v)
                trials[condition] = brain.counts[eyes].copy(), brain.rng.bit_generator.state
                rows.append(row)
                session.eyes.close()
                print(json.dumps(row), flush=True)
            np.testing.assert_array_equal(trials['video'][0], trials['photoreceptor_blocked'][0])
            assert trials['video'][1] == trials['photoreceptor_blocked'][1]
        body.reset()
        brain = Brain(graph)
        session = CoupledSession(brain, body)
        session.command('eye_feedback', True)
        samples = []
        for _ in range(15):
            session.advance(100)
            if not samples or samples[-1]['sample_ms'] != session.eyes.snapshot()['sample_ms']:
                samples.append(session.eyes.snapshot())
        assert len(samples) == 3 and brain.counts.sum() > 0
        np.save(args.output_dir / 'eyes-preview.npy', body.eye_preview)
        report = {'format': 'nexus-vision-results-1', 'success': True,
                  'manifest_sha256': hashlib.sha256((pack / 'manifest.json').read_bytes()).hexdigest(),
                  'movie_sha256': hashlib.sha256(movie.read_bytes()).hexdigest(),
                  'video_trials': rows, 'camera_samples': samples,
                  'camera_total_spikes': int(brain.counts.sum()),
                  'limits': ['Brightness-only visual input, not retinotopy, motion perception, or object recognition.',
                             'Videos are resampled to 20 Hz without audio and supplied equally to both eyes.',
                             '100 Hz at full white is unfitted; photoreceptors retain the generic spike model.',
                             'No chloride reversal potential: full-field input produces uncalibrated target voltages below -110 mV.']}
        (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    finally:
        if session is not None:
            session.eyes.close()
        body.close()


if __name__ == '__main__':
    main()
