"""Prepare a local video's explicit molecular receptors for native playback.

The command-line runner and GUI worker share this operation. No neural state is
advanced here. A successful report is published only after the response is ready.
"""
import hashlib
import json
from pathlib import Path
from queue import Full
from tempfile import TemporaryDirectory
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


def select_coverage(registry, coverage):
    """Keep the eight anchor identities/seeds, then extend anatomical coverage."""
    selected = select_locations(registry)
    if coverage == 'sample8':
        return selected
    mapping = VisualColumns(registry)
    if coverage == 'mapped':
        return selected + [root for side in ('L', 'R') for root in mapping.ids[side]
                           if root not in selected]
    if coverage != 'sample64':
        raise ValueError('Choose sample8, sample64 or mapped coverage')
    for side in ('L', 'R'):
        ids, uv = mapping.ids[side], mapping.uv[side]
        if len(ids) < 32:
            raise ValueError('64-cell coverage needs at least 32 mapped receptors per eye')
        chosen = [i for i, root in enumerate(ids) if root in selected]
        nearest = np.min(((uv[:, None, :] - uv[chosen][None, :, :])**2).sum(axis=2), axis=1)
        nearest[chosen] = -1.
        while len(chosen) < 32:
            i = int(np.argmax(nearest))
            chosen.append(i)
            selected.append(ids[i])
            nearest = np.minimum(nearest, ((uv - uv[i])**2).sum(axis=1))
            nearest[chosen] = -1.
    return selected


class _ResponseWorkspace:
    """Column-contiguous disk arrays; only a small molecular group lives in RAM."""
    def __init__(self, directory):
        self.temporary = TemporaryDirectory(prefix='.receptor-work-', dir=directory)
        self.arrays = []

    def array(self, name, shape, dtype):
        array = np.lib.format.open_memmap(Path(self.temporary.name)/f'{name}.npy', mode='w+',
                                        dtype=dtype, shape=shape, fortran_order=True)
        self.arrays.append(array)
        return array

    def close(self):
        for array in self.arrays:
            array._mmap.close()
        self.temporary.cleanup()


def prepare_response(*, video, output_dir, pack, white_rate_hz, transfer,
                     black_rate_hz=0., duration_ms=300, seed=88300, backend='cpu',
                     batch_cells=8, receptor_ids=None, release_curve=None,
                     coverage='sample8', progress=None, cancelled=None):
    if (isinstance(duration_ms, bool) or not isinstance(duration_ms, int) or duration_ms < 1):
        raise ValueError('Duration must be a positive number of milliseconds')
    if backend not in ('cpu', 'cuda'):
        raise ValueError('Choose cpu or cuda explicitly')
    if isinstance(batch_cells, bool) or not isinstance(batch_cells, int) or not 1 <= batch_cells <= 8:
        raise ValueError('Preparation groups must contain between 1 and 8 receptors')
    if receptor_ids is not None and coverage != 'sample8':
        raise ValueError('Choose explicit IDs or a coverage preset, not both')

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
    ids = list(receptor_ids) if receptor_ids is not None else select_coverage(registry, coverage)
    photon_seed, cascade_seed = np.random.SeedSequence(seed).spawn(2)
    photon_seeds = photon_seed.generate_state(len(ids), dtype=np.uint64).tolist()
    cascade_seeds = cascade_seed.generate_state(len(ids), dtype=np.uint64).tolist()
    source_hash = file_hash(video)
    curve = ReleaseCurve.load(release_curve) if release_curve is not None else None
    curve_bytes = Path(release_curve).read_bytes() if curve is not None else None
    source = VideoPhotonInput(video, registry, ids, photon_seeds,
                             white_rate_hz=white_rate_hz, black_rate_hz=black_rate_hz,
                             transfer=transfer)
    workspace = None
    try:
        check_cancelled()
        output_dir.mkdir(parents=True, exist_ok=False)
        workspace = _ResponseWorkspace(output_dir)
        intensity_store = workspace.array('intensity', (duration_ms, len(ids)), np.float64)
        photon_store = workspace.array('photons', (duration_ms, len(ids)), np.int64)
        wall = time.perf_counter()
        elapsed = 0
        if progress:
            progress({'kind': 'progress', 'elapsed_ms': 0, 'duration_ms': duration_ms,
                      'phase': 'decoding', 'cells': len(ids)})
        # Decode/absorb once, preserving each cell's input RNG and pixel sample.
        while elapsed < duration_ms:
            check_cancelled()
            inputs = source.read(min(50, duration_ms-elapsed))
            counts = inputs['absorbed_photons']
            if not len(counts):
                break
            stop = elapsed + len(counts)
            intensity_store[elapsed:stop] = inputs['intensity']
            photon_store[elapsed:stop] = counts
            elapsed += len(counts)
            if progress:
                progress({'kind': 'progress', 'elapsed_ms': elapsed, 'duration_ms': duration_ms,
                          'phase': 'decoding', 'cells': len(ids)})
        source.close()
        check_cancelled()
        if not elapsed:
            raise ValueError('The video contains no input frames at 20 fps')
        intensity, photons = intensity_store[:elapsed], photon_store[:elapsed]
        channels = workspace.array('channels', (elapsed*10, len(ids)), np.int64)
        voltage = workspace.array('voltage', (elapsed*10, len(ids)), np.float64)
        events = np.zeros(len(ids), dtype=np.int64)
        # With no receptor feedback in this model, an entire group's recording
        # can finish before constructing the next group's independent states.
        for first in range(0, len(ids), batch_cells):
            check_cancelled()
            last = min(first+batch_cells, len(ids))
            if progress:
                progress({'kind': 'progress', 'elapsed_ms': 0, 'duration_ms': elapsed,
                          'phase': 'initializing', 'cells': len(ids), 'first_cell': first,
                          'last_cell': last, 'work_done': first*elapsed, 'work_total': len(ids)*elapsed})
            if backend == 'cuda':
                from nexus.brain.phototransduction_batch import BatchedPhototransduction
                model = BatchedPhototransduction(cascade_seeds[first:last], batch_cells=batch_cells)
            else:
                from nexus.brain.phototransduction_parallel import ParallelPhototransduction
                models = [ParallelPhototransduction(seed=s) for s in cascade_seeds[first:last]]
            for start in range(0, elapsed, 50):
                check_cancelled()
                stop = min(start+50, elapsed)
                counts = photons[start:stop, first:last]
                if backend == 'cuda':
                    result = model.advance(counts)
                else:
                    separate = [m.advance(counts[:, i]) for i, m in enumerate(models)]
                    result = {key: np.column_stack([r[key] for r in separate]) for key in separate[0]}
                if curve is not None:
                    curve.evaluate(result['voltage_mv'])
                channels[start*10:stop*10, first:last] = result['open_channels']
                voltage[start*10:stop*10, first:last] = result['voltage_mv']
                if progress:
                    progress({'kind': 'progress', 'elapsed_ms': stop, 'duration_ms': elapsed,
                              'phase': 'computing', 'cells': len(ids), 'first_cell': first, 'last_cell': last,
                              'work_done': first*elapsed+stop*(last-first), 'work_total': len(ids)*elapsed})
            if backend == 'cuda':
                events[first:last] = model.events
                del model
            else:
                events[first:last] = [m.events for m in models]
                del models, separate
            del result
        seconds = time.perf_counter()-wall
        check_cancelled()
        if file_hash(video) != source_hash:
            raise ValueError('The source video changed during preparation')
        if progress:
            progress({'kind': 'progress', 'phase': 'exporting', 'elapsed_ms': elapsed,
                      'duration_ms': elapsed, 'cells': len(ids),
                      'work_done': len(ids)*elapsed, 'work_total': len(ids)*elapsed})
        np.savez_compressed(output_dir/'response.npz', ids=np.asarray(ids), intensity=intensity,
                            absorbed_photons=photons, open_channels=channels, voltage_mv=voltage)
        mapping = VisualColumns(registry)
        eye_ids = {str(root) for name in ('eye_left', 'eye_right')
                   for root in registry['circuits'][name]['ids']}
        mapped_ids = set(mapping.ids['L']) | set(mapping.ids['R'])
        report = {'format':'nexus-video-phototransduction-1','success':True,
                  'source_name':video.name,'source_sha256':source_hash,
                  'registry_sha256':registry_hash,'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  'backend':backend,'batch_cells':batch_cells if backend == 'cuda' else None,
                  'microvilli_per_receptor':30000,'exposure_ms':elapsed,'wall_seconds':seconds,
                  'coverage': {'mode': 'explicit' if receptor_ids is not None else coverage,
                               'mapped_cells': len(mapped_ids), 'eye_cohort_cells': len(eye_ids),
                               'cells_without_coordinates': len(eye_ids-mapped_ids),
                               'selected_cells': len(ids)},
                  'storage': {'method': 'disk arrays and sequential independent receptor groups',
                              'maximum_resident_receptors': min(batch_cells, len(ids)),
                              'molecular_state_bytes': min(batch_cells, len(ids))*30000*104,
                              'trace_disk_array_bytes': sum(a.nbytes for a in workspace.arrays)},
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
                            'Timing includes decoding, group initialization, possible JIT and disk-array writes; excludes final archive/report export.',
                            'All mapped coverage excludes receptors without image coordinates; it is not a complete eye or calibrated optics.',
                            'Sequential group processing is valid only for the present independent receptors without synaptic feedback.']}
        report['source_hashes_complete'] = len(report['implementation_sha256']) == len(IMPLEMENTATION_FILES)
        check_cancelled()
        if curve_bytes is not None:
            (output_dir/'release-curve.json').write_bytes(curve_bytes)
            report['prepared_release_curve_sha256'] = hashlib.sha256(curve_bytes).hexdigest()
        (output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        return report
    finally:
        source.close()
        if workspace is not None:
            workspace.close()


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
