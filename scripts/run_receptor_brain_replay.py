"""Replay selected molecular receptor voltages through the MaleCNS connectome.

Requires an explicit, unfitted release curve. Compares matched video/dark
traces with and without receptor output; never inserts photoreceptor spikes.
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
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT/'.runtime/numba'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.runtime/matplotlib'))
import numpy as np
from nexus.brain.receptor_release import ReleaseCurve
from nexus.brain.runtime import Brain, Connectome, DT_MS


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_response(directory, registry_hash):
    report = json.loads((directory/'report.json').read_text())
    if (report.get('format') != 'nexus-video-phototransduction-1'
            or report.get('membrane_dt_ms') != DT_MS
            or report.get('registry_sha256') != registry_hash):
        raise ValueError('Replay needs a matching, 0.1 ms video-phototransduction response')
    with np.load(directory/'response.npz', allow_pickle=False) as data:
        ids = data['ids'].tolist()
        voltage = data['voltage_mv'].copy()
    if (ids != [r['id'] for r in report['receptors']]
            or voltage.shape != (report['exposure_ms'] * 10, len(ids))
            or not np.isfinite(voltage).all()):
        raise ValueError('Response IDs or voltage dimensions disagree with their report')
    return ids, voltage, report


def trace_selection(brain, source_ids):
    """Select anatomical stages by edges, without looking at their responses."""
    registry = brain.graph.circuits['graded_relays']['groups']
    lamina = {brain.lookup[int(root)] for k, ids in registry.items()
              if k.startswith(('L1_', 'L2_')) for root in ids}
    medulla = set(brain.graded_indices) - lamina
    selected = list(brain.resolve(source_ids))
    labels = ['receptor'] * len(selected)
    if len(selected) > 8:
        selected, labels = selected[:8], labels[:8]
    previous = list(brain.receptor_indices)
    for label, eligible in [('lamina', lamina), ('medulla', medulla), ('downstream', None)]:
        scores = {}
        for pre in previous:
            a, b = brain.graph.offsets[pre:pre+2]
            for post, weight in zip(brain.graph.posts[a:b], brain.graph.weights[a:b]):
                post = int(post)
                if weight and post not in selected and (eligible is None or post in eligible):
                    scores[post] = scores.get(post, 0.) + abs(float(weight))
        previous = sorted(scores, key=lambda i: (-scores[i], i))[:8]
        selected.extend(previous)
        labels.extend([label] * len(previous))
    return [str(brain.graph.ids[i]) for i in selected], labels


def replay(graph, ids, voltage, curve, trace_ids, *, blocked, name):
    brain = Brain(graph)
    brain.set_graded_relays(True)
    brain.configure_receptor_replay(ids, reference_mv=curve.reference_mv)
    if blocked:
        brain.silence(ids)
    rate = curve.evaluate(voltage)
    traces, snapshots, counts, ticks = [], [], [], []
    wall = time.perf_counter()
    for start in range(0, len(voltage), 500):
        stop = min(start + 500, len(voltage))
        result = brain.advance((stop-start)*.0001, trace_ids=trace_ids,
                               receptor_voltage_mv=voltage[start:stop], receptor_release_hz=rate[start:stop])
        traces.append(result['voltage_mv'])
        snapshots.append(brain.v.copy())
        counts.append(brain.counts.copy())
        ticks.append(brain.step)
        print(f'{name}: {brain.time*1000:g} ms replayed', flush=True)
    summary = {'wall_seconds': time.perf_counter()-wall,
               'source_spikes': int(brain.counts[brain.receptor_indices].sum()),
               'graded_relay_spikes': int(brain.counts[brain.graded_indices].sum()),
               'total_spikes': int(brain.counts.sum()),
               'release_equivalent_hz_range': [float(rate.min()), float(rate.max())]}
    return {'trace': np.concatenate(traces), 'voltage': np.asarray(snapshots),
            'counts': np.asarray(counts), 'ticks': np.asarray(ticks), 'summary': summary}


def comparison(video, dark, mask):
    delta = video['voltage'][:, mask] - dark['voltage'][:, mask]
    maximum = np.max(np.abs(delta), axis=0)
    spike_delta = video['counts'][-1, mask] - dark['counts'][-1, mask]
    return {'cells': int(mask.sum()), 'sampled_maximum_absolute_voltage_delta_mv': float(maximum.max(initial=0.)),
            'cells_at_least_0_01_mv': int((maximum >= .01).sum()),
            'cells_at_least_0_5_mv': int((maximum >= .5).sum()),
            'cells_with_changed_spike_count': int(np.count_nonzero(spike_delta)),
            'net_spike_count_difference': int(spike_delta.sum())}


def plot_response(path, results, trace_ids, labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True, layout='constrained')
    colors = ('#e84500', '#bc2400', '#bd8d00', '#6c4300')
    times = (np.arange(len(results['video']['trace'])) + 1) * DT_MS
    delta = results['video']['trace'] - results['dark']['trace']
    blocked_delta = results['video_blocked']['trace'] - results['dark_blocked']['trace']
    for ax, stage in zip(axes, ('receptor', 'lamina', 'medulla', 'downstream')):
        group = [i for i, label in enumerate(labels) if label == stage]
        # Show four preselected connections per stage, even when nearly silent.
        for j, i in enumerate(group[:4]):
            ax.plot(times, delta[:, i], color=colors[j], label=trace_ids[i])
            ax.plot(times, blocked_delta[:, i], color=colors[j], linestyle=':', alpha=.6)
        ax.axhline(0., color='black', linewidth=.5)
        ax.set_ylabel(f'{stage.capitalize()}\nΔ mV')
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        if group:
            ax.legend(ncol=4, fontsize=8, loc='upper right')
    axes[-1].set_xlabel('Replay time (ms)')
    fig.suptitle('Video minus matched darkness through connectome edges\n'
                 'Solid: receptor output enabled · dotted: blocked · release curve unfitted')
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-response', type=Path, required=True)
    parser.add_argument('--dark-response', type=Path, required=True)
    parser.add_argument('--release-curve', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pack', type=Path, default=ROOT/'data/brain-male-cns-v1.0-lif')
    args = parser.parse_args()
    manifest = json.loads((args.pack/'manifest.json').read_text())
    graph = Connectome.load(args.pack, allow_experimental=True)
    registry_hash = manifest['files'][manifest['circuit_registry']]
    ids, voltage, video_report = load_response(args.video_response, registry_hash)
    dark_ids, dark_voltage, dark_report = load_response(args.dark_response, registry_hash)
    if ids != dark_ids or voltage.shape != dark_voltage.shape:
        raise ValueError('Video and darkness responses must share IDs and duration')
    matched = ('source_sha256', 'root_seed', 'photon_seeds', 'cascade_seeds', 'transfer', 'video_input_fps')
    if (any(video_report[k] != dark_report[k] for k in matched)
            or dark_report['white_rate_hz'] != 0 or dark_report['black_rate_hz'] != 0):
        raise ValueError('Use matched seeds and stimulus, with zero absorbed photons for the dark control')
    curve = ReleaseCurve.load(args.release_curve)
    # Validate the complete input before beginning any of the experiments.
    curve.evaluate(voltage)
    curve.evaluate(dark_voltage)
    preview = Brain(graph)
    preview.set_graded_relays(True)
    preview.configure_receptor_replay(ids, reference_mv=curve.reference_mv)
    source_mask = preview.receptor_mask.copy()
    relay_mask = preview.graded_mask.copy()
    direct_mask = np.zeros(len(graph.ids), dtype=bool)
    for pre in preview.receptor_indices:
        a, b = graph.offsets[pre:pre+2]
        direct_mask[graph.posts[a:b][graph.weights[a:b] != 0]] = True
    trace_ids, labels = trace_selection(preview, ids)
    del preview
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = {}
    for name, values, blocked in [('dark', dark_voltage, False), ('video', voltage, False),
                                 ('dark_blocked', dark_voltage, True), ('video_blocked', voltage, True)]:
        results[name] = replay(graph, ids, values, curve, trace_ids, blocked=blocked, name=name)
    masks = {'non_source': ~source_mask, 'direct_targets': direct_mask & ~source_mask,
             'beyond_direct_targets': ~source_mask & ~direct_mask,
             'graded_relays': relay_mask, 'other_neurons': ~source_mask & ~relay_mask}
    comparisons = {name: comparison(results['video'], results['dark'], mask) for name, mask in masks.items()}
    blocked = comparison(results['video_blocked'], results['dark_blocked'], ~source_mask)
    first_difference = np.flatnonzero(np.any(voltage != dark_voltage, axis=1))
    first_tick = int(first_difference[0] + 1) if len(first_difference) else None
    ticks = results['dark']['ticks']
    before = ticks < first_tick if first_tick is not None else np.ones(len(ticks), dtype=bool)
    checks = {'source_spikes_zero': all(r['summary']['source_spikes'] == 0 for r in results.values()),
              'graded_relay_spikes_zero': all(r['summary']['graded_relay_spikes'] == 0 for r in results.values()),
              'pre_input_sampled_brain_state_identical': bool(
                  np.array_equal(results['video']['voltage'][before], results['dark']['voltage'][before])
                  and np.array_equal(results['video']['counts'][before], results['dark']['counts'][before])),
              'blocked_non_source_state_identical': bool(
                  np.array_equal(results['video_blocked']['voltage'][:, ~source_mask],
                                 results['dark_blocked']['voltage'][:, ~source_mask])
                  and np.array_equal(results['video_blocked']['counts'], results['dark_blocked']['counts']))}
    np.savez_compressed(args.output_dir/'response.npz', trace_ids=np.asarray(trace_ids),
                        trace_labels=np.asarray(labels), snapshot_ticks=ticks,
                        **{f'{k}_trace_mv': r['trace'] for k, r in results.items()},
                        **{f'{k}_sample_voltage_mv': r['voltage'] for k, r in results.items()},
                        **{f'{k}_sample_spike_counts': r['counts'] for k, r in results.items()})
    plot_response(args.output_dir/'response.png', results, trace_ids, labels)
    report = {'format': 'nexus-receptor-brain-replay-1', 'success': all(checks.values()),
              'dataset': graph.snapshot, 'neurons': len(graph.ids), 'edges': len(graph.posts),
              'duration_ms': len(voltage)*DT_MS, 'receptor_ids': ids, 'receptor_count': len(ids),
              'registry_sha256': registry_hash, 'manifest_sha256': file_hash(args.pack/'manifest.json'),
              'input_hashes': {kind: {name: file_hash(directory/name) for name in ('response.npz', 'report.json')}
                               for kind, directory in [('video', args.video_response), ('dark', args.dark_response)]},
              'release_curve': curve.snapshot(), 'release_curve_sha256': file_hash(args.release_curve),
              'release_sample_ms': 1., 'delivery_delay_ms': 1.8, 'voltage_sample_ms': DT_MS,
              'first_receptor_input_difference_ms': first_tick*DT_MS if first_tick is not None else None,
              'brain_snapshot_times_ms': (ticks*DT_MS).tolist(),
              'trace_selection': [{'id': root, 'stage': label} for root, label in zip(trace_ids, labels)],
              'conditions': {k: r['summary'] for k, r in results.items()}, 'checks': checks,
              'video_minus_dark': comparisons, 'blocked_video_minus_dark': blocked,
              'implementation_sha256': {str(p.relative_to(ROOT)): file_hash(p) for p in
                  [ROOT/'src/nexus/brain/runtime.py', ROOT/'src/nexus/brain/receptor_release.py',
                   ROOT/'src/nexus/brain/telemetry.py', Path(__file__).resolve()]},
              'limits': ['Selected explicit receptors only; the remaining eye cells have no external light drive.',
                         'The release curve is an unfitted connectivity-assay assumption, not measured histamine release.',
                         'Equivalent release packets preserve connectome weights; no receptor spikes or forced downstream activation.',
                         'Voltage replay excludes synaptic feedback into receptor sources and vesicle depletion or terminal calcium.',
                         'The existing graded relay law, conductance scales, reversals and delays remain unfitted.',
                         'Full-network differences are measured at the listed snapshots, not continuously.',
                         'Other cells begin at model rest; tonic output comes from the selected release/relay laws, not validated whole-brain spontaneous activity.',
                         'Offline experiment only; live GUI, whole-eye scaling and physiological calibration remain unfinished.',
                         'Timing includes any first-call JIT and trace/state copying; excludes loading, plotting and export.']}
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'success': report['success'], 'checks': checks, 'video_minus_dark': comparisons}), flush=True)
    if not report['success']:
        raise RuntimeError('Receptor replay control checks failed; inspect the saved report')


if __name__ == '__main__':
    main()
