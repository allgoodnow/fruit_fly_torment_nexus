"""Run local video through explicit absorbed-photon and molecular eye models."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT/'.runtime/numba'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.runtime/matplotlib'))
import numpy as np
from nexus.photon_input import VideoPhotonInput
from nexus.receptor_preparation import prepare_response


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
    report = prepare_response(**vars(args), progress=lambda p: print(
        f"{p['elapsed_ms']} ms of video exposure processed", flush=True))
    # Plotting stays in this optional command-line presentation, outside the GUI.
    with np.load(args.output_dir/'response.npz', allow_pickle=False) as data:
        photon_seeds = report['photon_seeds']
        manifest = json.loads((args.pack/'manifest.json').read_text())
        registry = json.loads((args.pack/manifest['circuit_registry']).read_text())
        source = VideoPhotonInput(args.video, registry, data['ids'].tolist(), photon_seeds,
                                 white_rate_hz=args.white_rate_hz, black_rate_hz=args.black_rate_hz,
                                 transfer=args.transfer)
        try:
            plot_response(args.output_dir/'response.png', source, data['intensity'], data['open_channels'], data['voltage_mv'])
        finally:
            source.close()

if __name__ == '__main__':
    main()
