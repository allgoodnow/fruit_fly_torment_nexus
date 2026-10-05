"""Prepare a local video's explicit molecular receptors for native playback.

The command-line runner and GUI worker share this operation. No neural state is
advanced here. A successful report is published only after the response is ready.
"""
import hashlib
import json
from pathlib import Path
from queue import Full
import traceback
import sys
import time

import numpy as np
from .brain.receptor_release import ReleaseCurve
from .photon_input import VideoPhotonInput
from .retina import VisualColumns
from .video import VIDEO_FPS

ROOT = Path(__file__).resolve().parent
IMPLEMENTATION_FILES = ('receptor_preparation.py', 'photon_input.py', 'retina.py', 'video.py',
                        'brain/phototransduction_parallel.py', 'brain/phototransduction_batch.py',
                        'brain/phototransduction_cuda.py', 'brain/phototransduction.py',
                        'brain/photoreceptor.py')


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class PreparationCancelled(Exception):
    """A requested cancellation at a batch boundary, never a ready response."""


def implementation_hashes():
    if getattr(sys, 'frozen', False):
        bundled = Path(sys._MEIPASS)/'nexus/source-hashes.json'
        available = json.loads(bundled.read_text()) if bundled.is_file() else {}
        return {f'src/nexus/{name}': available[f'src/nexus/{name}'] for name in IMPLEMENTATION_FILES
                if f'src/nexus/{name}' in available}
    return {f'src/nexus/{name}': file_hash(ROOT/name) for name in IMPLEMENTATION_FILES}


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


def prepare_response(*, video, output_dir, pack, white_rate_hz, transfer,
                     black_rate_hz=0., duration_ms=300, seed=88300, backend='cpu',
                     batch_cells=8, receptor_ids=None, release_curve=None,
                     progress=None, cancelled=None):
    if (isinstance(duration_ms, bool) or not isinstance(duration_ms, int) or duration_ms < 1):
        raise ValueError('Duration must be a positive number of milliseconds')
    if backend not in ('cpu', 'cuda'):
        raise ValueError('Choose cpu or cuda explicitly')

    def check_cancelled():
        if cancelled is not None and cancelled():
            raise PreparationCancelled()

    video, output_dir, pack = Path(video), Path(output_dir), Path(pack)
    check_cancelled()
    manifest_path = pack/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    name = manifest['circuit_registry']
    if Path(name).name != name:
        raise ValueError('Registry must be a file within the brain pack')
    registry_bytes = (pack/name).read_bytes()
    registry_hash = hashlib.sha256(registry_bytes).hexdigest()
    if registry_hash != manifest['files'][name]:
        raise ValueError('Brain-pack registry checksum mismatch')
    registry = json.loads(registry_bytes)
    ids = list(receptor_ids) if receptor_ids is not None else select_locations(registry)
    photon_seed, cascade_seed = np.random.SeedSequence(seed).spawn(2)
    photon_seeds = photon_seed.generate_state(len(ids), dtype=np.uint64).tolist()
    cascade_seeds = cascade_seed.generate_state(len(ids), dtype=np.uint64).tolist()
    source_hash = file_hash(video)
    curve = ReleaseCurve.load(release_curve) if release_curve is not None else None
    curve_bytes = Path(release_curve).read_bytes() if curve is not None else None
    source = VideoPhotonInput(video, registry, ids, photon_seeds,
                             white_rate_hz=white_rate_hz, black_rate_hz=black_rate_hz,
                             transfer=transfer)
    try:
        check_cancelled()
        if progress:
            progress({'kind': 'progress', 'elapsed_ms': 0, 'duration_ms': duration_ms, 'phase': 'initializing'})
        if backend == 'cuda':
            from nexus.brain.phototransduction_batch import BatchedPhototransduction
            model = BatchedPhototransduction(cascade_seeds, batch_cells=batch_cells)
        else:
            from nexus.brain.phototransduction_parallel import ParallelPhototransduction
            models = [ParallelPhototransduction(seed=seed) for seed in cascade_seeds]
        output_dir.mkdir(parents=True, exist_ok=False)
        input_parts, channel_parts, voltage_parts = [], [], []
        wall = time.perf_counter()
        elapsed = 0
        while elapsed < duration_ms:
            check_cancelled()
            inputs = source.read(min(50, duration_ms-elapsed))
            counts = inputs['absorbed_photons']
            if not len(counts):
                break
            if backend == 'cuda':
                result = model.advance(counts)
            else:
                separate = [m.advance(counts[:, i]) for i, m in enumerate(models)]
                result = {key: np.column_stack([r[key] for r in separate]) for key in separate[0]}
            input_parts.append(inputs)
            channel_parts.append(result['open_channels'])
            voltage_parts.append(result['voltage_mv'])
            elapsed += len(counts)
            if progress:
                progress({'kind': 'progress', 'elapsed_ms': elapsed, 'duration_ms': duration_ms, 'phase': 'computing'})
        seconds = time.perf_counter()-wall
        check_cancelled()
        if not input_parts:
            raise ValueError('The video contains no input frames at 20 fps')
        intensity = np.concatenate([p['intensity'] for p in input_parts])
        photons = np.concatenate([p['absorbed_photons'] for p in input_parts])
        channels, voltage = np.concatenate(channel_parts), np.concatenate(voltage_parts)
        if curve is not None:
            curve.evaluate(voltage)
        if file_hash(video) != source_hash:
            raise ValueError('The source video changed during preparation')
        np.savez_compressed(output_dir/'response.npz', ids=np.asarray(ids), intensity=intensity,
                            absorbed_photons=photons, open_channels=channels, voltage_mv=voltage)
        events = model.events.tolist() if backend == 'cuda' else [m.events for m in models]
        report = {'format':'nexus-video-phototransduction-1','success':True,
                  'source_name':video.name,'source_sha256':source_hash,
                  'registry_sha256':registry_hash,'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  'backend':backend,'batch_cells':batch_cells if backend == 'cuda' else None,
                  'microvilli_per_receptor':30000,'exposure_ms':elapsed,'wall_seconds':seconds,
                  'video_input_fps':VIDEO_FPS,'absorbed_photon_bin_ms':1.,'membrane_dt_ms':.1,
                  'transfer':transfer,'white_rate_hz':white_rate_hz,'black_rate_hz':black_rate_hz,
                  'calibrated':False,'root_seed':seed,'photon_seeds':photon_seeds,'cascade_seeds':cascade_seeds,
                  'receptors':[{'id':root,'side':side,'uv':uv.tolist(),'absorbed_photons':int(photons[:, i].sum()),
                                'expected_absorbed_photons':float(((black_rate_hz+
                                  (white_rate_hz-black_rate_hz)*intensity[:, i])/1000.).sum()),
                                'maximum_open_channels':int(channels[:, i].max()),'molecular_events':int(events[i]),
                                'voltage_range_mv':[float(voltage[:, i].min()),float(voltage[:, i].max())]}
                               for i,(root,side,uv) in enumerate(zip(ids,source.sides,source.uv))],
                  'implementation_sha256': implementation_hashes(),
                  'limits':['RGB intensity, white/black rates and video projection are explicit uncalibrated assumptions.',
                            'Poisson arrival counts are a new controlled exposure input, separate from molecular event randomness.',
                            'Video is held at 20 fps, counts are binned at 1 ms; sub-bin arrival times are not modeled.',
                            'Selected receptors only; no population multiplication, synaptic feedback or simultaneous live molecular integration.',
                            'Timing includes decoding and possible JIT on the first advance; excludes setup, plotting and report export.']}
        report['source_hashes_complete'] = len(report['implementation_sha256']) == len(IMPLEMENTATION_FILES)
        check_cancelled()
        if curve_bytes is not None:
            (output_dir/'release-curve.json').write_bytes(curve_bytes)
            report['prepared_release_curve_sha256'] = hashlib.sha256(curve_bytes).hexdigest()
        (output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        return report
    finally:
        source.close()


def preparation_worker(settings, events, cancel):
    """Spawned worker; only small progress messages cross the GUI boundary."""
    def progress(value):
        try:
            events.put_nowait(value)
        except Full:
            pass

    try:
        report = prepare_response(**settings, progress=progress, cancelled=cancel.is_set)
        events.put({'kind': 'complete', 'report_path': str(Path(settings['output_dir'])/'report.json'),
                    'curve_path': str(Path(settings['output_dir'])/'release-curve.json'),
                    'exposure_ms': report['exposure_ms'], 'cells': len(report['receptors'])})
    except PreparationCancelled:
        events.put({'kind': 'cancelled'})
    except Exception as error:
        events.put({'kind': 'failed', 'message': str(error), 'detail': traceback.format_exc()})
