import json
from pathlib import Path

import numpy as np
import pytest

from nexus.receptor_playback import ReceptorPlayback, file_hash
from nexus.receptor_preparation import PreparationCancelled, prepare_response, select_coverage
from test_receptor_replay import replay_model
from test_vision import movie

CURVE = Path(__file__).resolve().parents[1]/'experiments/receptor-release-assay-curve.json'


@pytest.fixture
def preparation_pack(tmp_path):
    pack = tmp_path/'pack'
    pack.mkdir()
    registry = {'snapshot': 'male-cns:v1.0',
                'circuits': {'eye_left': {'ids': ['13']}, 'eye_right': {'ids': ['14']}},
                'visual_columns': {'format': 'nexus-visual-columns-1', 'eyes': {
                    'L': {'cells': [{'id': '13', 'uv': [.25, .5]}]},
                    'R': {'cells': [{'id': '14', 'uv': [.75, .5]}]}}}}
    path = pack/'circuits.json'
    path.write_text(json.dumps(registry))
    (pack/'manifest.json').write_text(json.dumps({'circuit_registry': path.name,
                                                'files': {path.name: file_hash(path)}}))
    return pack


def test_cpu_preparation_publishes_a_loadable_molecular_recording(preparation_pack, movie, tmp_path):
    output = tmp_path/'complete'
    progress = []
    report = prepare_response(video=movie, pack=preparation_pack, output_dir=output,
                              white_rate_hz=30000., transfer='srgb', duration_ms=150,
                              receptor_ids=['13'], release_curve=CURVE, progress=progress.append)
    assert report['microvilli_per_receptor'] == 30000 and report['backend'] == 'cpu'
    assert [p['elapsed_ms'] for p in progress if p['phase'] == 'computing'] == [50, 100, 150]
    assert [p['work_done'] for p in progress if p['phase'] == 'computing'] == [50, 100, 150]
    assert report['storage']['maximum_resident_receptors'] == 1
    assert report['coverage']['selected_cells'] == 1
    assert report['prepared_release_curve_sha256'] == file_hash(CURVE)
    assert (output/'release-curve.json').read_bytes() == CURVE.read_bytes()
    assert report['source_hashes_complete'] and report['calibrated'] is False
    with np.load(output/'response.npz') as data:
        assert data['voltage_mv'].shape == (1500, 1)
        assert not data['absorbed_photons'][:100].any()
        assert not data['open_channels'][:1000].any()
        assert data['open_channels'][1000:].max() > 0
        voltage = data['voltage_mv'].copy()
    brain = replay_model()
    brain.graph.registry_sha256 = report['registry_sha256']
    prepared = ReceptorPlayback(output/'report.json', output/'release-curve.json', brain, movie)
    brain.configure_receptor_replay(prepared.ids, reference_mv=prepared.curve.reference_mv)
    brain.advance(.15, **prepared.input(0, 1500))
    prepared.after_step(brain)
    assert prepared.ended and not brain.counts[brain.receptor_indices].any()
    np.testing.assert_array_equal(brain.v[brain.receptor_indices], voltage[-1])


def test_cancel_between_normal_batches_does_not_publish_a_ready_response(preparation_pack, movie, tmp_path):
    output = tmp_path/'cancelled'
    cancel = False
    progress = []

    def observe(event):
        nonlocal cancel
        if event['phase'] == 'computing':
            progress.append(event['elapsed_ms'])
            cancel = True

    with pytest.raises(PreparationCancelled):
        prepare_response(video=movie, pack=preparation_pack, output_dir=output,
                         white_rate_hz=30000., transfer='srgb', duration_ms=150,
                         receptor_ids=['13'], release_curve=CURVE, progress=observe,
                         cancelled=lambda: cancel)
    assert progress == [50]
    assert not (output/'report.json').exists() and not (output/'response.npz').exists()
    assert not list(output.glob('.receptor-work-*'))


def test_coverage_preserves_anchor_order_and_unique_spatial_targets():
    eyes = {side: [{'id': f'{side}{i}', 'uv': [x, y]}
                   for i, (x, y) in enumerate((x, y) for x in np.linspace(0, 1, 8)
                                              for y in np.linspace(0, 1, 5))]
            for side in ('L', 'R')}
    registry = {'snapshot': 'male-cns:v1.0',
                'circuits': {name: {'ids': [c['id'] for c in eyes[side]]}
                             for side, name in [('L', 'eye_left'), ('R', 'eye_right')]},
                'visual_columns': {'format': 'nexus-visual-columns-1', 'eyes': {
                    side: {'cells': eyes[side]} for side in ('L', 'R')}}}
    anchors = select_coverage(registry, 'sample8')
    spread = select_coverage(registry, 'sample64')
    mapped = select_coverage(registry, 'mapped')
    assert len(set(anchors)) == 8 and len(set(spread)) == 64 and len(set(mapped)) == 80
    assert anchors == spread[:8] == mapped[:8]
    assert sum(root.startswith('L') for root in spread) == 32
    assert sum(root.startswith('R') for root in spread) == 32


def test_population_plot_reads_only_displayed_columns(tmp_path):
    from zipfile import ZipFile
    from scripts.run_video_phototransduction import read_plot_array
    values = np.arange(100*16, dtype=np.float64).reshape(100, 16)
    path = tmp_path/'traces.npz'
    # Full disk arrays are column ordered; EOF-truncated input views can be row ordered.
    np.savez_compressed(path, voltage_mv=np.asfortranarray(values), intensity=values)
    with ZipFile(path) as archive:
        for name in ('voltage_mv', 'intensity'):
            np.testing.assert_array_equal(read_plot_array(archive, name), values[:, :8])
