"""Run local video through explicit absorbed-photon and molecular eye models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT/'.runtime/numba'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.runtime/matplotlib'))
import numpy as np
from nexus.photon_input import VideoPhotonInput
from nexus.retina import VisualColumns
from nexus.video import VIDEO_FPS


def select_locations(registry):
    """Four distinct mapped receptors near quadrant centers per eye."""
    mapping = VisualColumns(registry)
    selected = []
    for side in ('L', 'R'):
        for center in ((.25, .25), (.75, .25), (.25, .75), (.75, .75)):
            order = np.argsort(((mapping.uv[side]-center)**2).sum(axis=1), kind='stable')
            root = next((mapping.ids[side][i] for i in order if mapping.ids[side][i] not in selected), None)
            if root is None:
                raise ValueError('Supply explicit IDs when fewer than four mapped receptors exist per eye')
            selected.append(root)
    return selected


def plot_response(path, source, intensity, channels, voltage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout='constrained')
    colors = ('#e84500', '#bc2400', '#bd8d00', '#6c4300')
    for i, (side, root) in enumerate(zip(source.sides, source.ids)):
        color, style = colors[i % 4], '-' if side == 'L' else '--'
        axes[0].plot(np.arange(len(intensity)), intensity[:, i], color=color, linestyle=style)
        axes[1].plot(np.arange(len(channels))*.1, channels[:, i], color=color, linestyle=style)
        axes[2].plot((np.arange(len(voltage))+1)*.1, voltage[:, i], color=color,
                     linestyle=style, label=f'{side} · {root}')
    for ax, ylabel in zip(axes, ('Relative input', 'Open TRP channels', 'Voltage (mV)')):
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylim(-.03, 1.03)
    axes[2].set_xlabel('Video exposure time (ms)')
    axes[2].legend(ncol=4, fontsize=8, loc='lower right')
    fig.suptitle('Video-driven photoreceptor responses\n'
                 f'Assumed white: {source.exposure.white_rate_hz:g} absorbed photons/s; '
                 f'black: {source.exposure.black_rate_hz:g}; {source.transfer} input')
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pack', type=Path, default=ROOT/'data/brain-male-cns-v1.0-lif')
    parser.add_argument('--receptor-ids', nargs='+')
    parser.add_argument('--white-rate-hz', type=float, required=True,
                        help='Assumed absorbed photons/s per receptor at white; not measured from video')
    parser.add_argument('--black-rate-hz', type=float, default=0.)
    parser.add_argument('--transfer', choices=('linear', 'srgb'), required=True)
    parser.add_argument('--duration-ms', type=int, default=300)
    parser.add_argument('--seed', type=int, default=88300)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--batch-cells', type=int, default=8)
    args = parser.parse_args()
    if args.duration_ms < 1:
        parser.error('Duration must be a positive number of milliseconds')
    manifest_path = args.pack/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    name = manifest['circuit_registry']
    if Path(name).name != name:
        raise ValueError('Registry must be a file within the brain pack')
    registry_bytes = (args.pack/name).read_bytes()
    registry_hash = hashlib.sha256(registry_bytes).hexdigest()
    if registry_hash != manifest['files'][name]:
        raise ValueError('Brain-pack registry checksum mismatch')
    registry = json.loads(registry_bytes)
    ids = args.receptor_ids or select_locations(registry)
    photon_seed, cascade_seed = np.random.SeedSequence(args.seed).spawn(2)
    photon_seeds = photon_seed.generate_state(len(ids), dtype=np.uint64).tolist()
    cascade_seeds = cascade_seed.generate_state(len(ids), dtype=np.uint64).tolist()
    source = VideoPhotonInput(args.video, registry, ids, photon_seeds,
                             white_rate_hz=args.white_rate_hz, black_rate_hz=args.black_rate_hz,
                             transfer=args.transfer)
    try:
        if args.backend == 'cuda':
            from nexus.brain.phototransduction_batch import BatchedPhototransduction
            model = BatchedPhototransduction(cascade_seeds, batch_cells=args.batch_cells)
        else:
            from nexus.brain.phototransduction_parallel import ParallelPhototransduction
            models = [ParallelPhototransduction(seed=seed) for seed in cascade_seeds]
        args.output_dir.mkdir(parents=True, exist_ok=False)
        input_parts, channel_parts, voltage_parts = [], [], []
        wall = time.perf_counter()
        elapsed = 0
        while elapsed < args.duration_ms:
            inputs = source.read(min(50, args.duration_ms-elapsed))
            counts = inputs['absorbed_photons']
            if not len(counts):
                break
            if args.backend == 'cuda':
                result = model.advance(counts)
            else:
                separate = [m.advance(counts[:, i]) for i, m in enumerate(models)]
                result = {key: np.column_stack([r[key] for r in separate]) for key in separate[0]}
            input_parts.append(inputs)
            channel_parts.append(result['open_channels'])
            voltage_parts.append(result['voltage_mv'])
            elapsed += len(counts)
            print(f'{elapsed} ms of video exposure processed', flush=True)
        seconds = time.perf_counter()-wall
        intensity = np.concatenate([p['intensity'] for p in input_parts])
        photons = np.concatenate([p['absorbed_photons'] for p in input_parts])
        channels, voltage = np.concatenate(channel_parts), np.concatenate(voltage_parts)
        np.savez_compressed(args.output_dir/'response.npz', ids=np.asarray(ids), intensity=intensity,
                            absorbed_photons=photons, open_channels=channels, voltage_mv=voltage)
        plot_response(args.output_dir/'response.png', source, intensity, channels, voltage)
        events = model.events.tolist() if args.backend == 'cuda' else [m.events for m in models]
        with args.video.open('rb') as stream:
            source_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        report = {'format':'nexus-video-phototransduction-1','success':True,
                  'source_name':args.video.name,'source_sha256':source_hash,
                  'registry_sha256':registry_hash,'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  'backend':args.backend,'batch_cells':args.batch_cells if args.backend == 'cuda' else None,
                  'microvilli_per_receptor':30000,'exposure_ms':elapsed,'wall_seconds':seconds,
                  'video_input_fps':VIDEO_FPS,'absorbed_photon_bin_ms':1.,'membrane_dt_ms':.1,
                  'transfer':args.transfer,'white_rate_hz':args.white_rate_hz,'black_rate_hz':args.black_rate_hz,
                  'calibrated':False,'root_seed':args.seed,'photon_seeds':photon_seeds,'cascade_seeds':cascade_seeds,
                  'receptors':[{'id':root,'side':side,'uv':uv.tolist(),'absorbed_photons':int(photons[:, i].sum()),
                                'expected_absorbed_photons':float(((args.black_rate_hz+
                                  (args.white_rate_hz-args.black_rate_hz)*intensity[:, i])/1000.).sum()),
                                'maximum_open_channels':int(channels[:, i].max()),'molecular_events':int(events[i]),
                                'voltage_range_mv':[float(voltage[:, i].min()),float(voltage[:, i].max())]}
                               for i,(root,side,uv) in enumerate(zip(ids,source.sides,source.uv))],
                  'implementation_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [ROOT/'src/nexus/photon_input.py', ROOT/'src/nexus/retina.py', ROOT/'src/nexus/video.py',
                                ROOT/'src/nexus/brain/phototransduction_parallel.py',
                                ROOT/'src/nexus/brain/phototransduction_batch.py',
                                ROOT/'src/nexus/brain/phototransduction_cuda.py',
                                ROOT/'src/nexus/brain/phototransduction.py',
                                ROOT/'src/nexus/brain/photoreceptor.py', Path(__file__).resolve()]},
                  'limits':['RGB intensity, white/black rates and video projection are explicit uncalibrated assumptions.',
                            'Poisson arrival counts are a new controlled exposure input, separate from molecular event randomness.',
                            'Video is held at 20 fps, counts are binned at 1 ms; sub-bin arrival times are not modeled.',
                            'Selected receptors only; no shared-population scale-up, transmitter release or live brain input.',
                            'Timing includes decoding and possible JIT on the first advance; excludes setup, plotting and report export.']}
        (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:
        source.close()


if __name__ == '__main__':
    main()
