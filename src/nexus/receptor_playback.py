"""Prepared receptor voltages on the shared brain/body/video clock.

This imports an offline exposure response, not an on-the-fly molecular model.
Playback ends at the last recorded voltage; no continuation is invented.
"""
import hashlib
import json
import mmap
from pathlib import Path
from tempfile import TemporaryFile
from zipfile import ZipFile

import numpy as np

from .brain.receptor_release import ReleaseCurve
from .brain.runtime import DT_MS
from .video import VIDEO_FPS

PLAYBACK_WINDOW_TICKS = 500  # 50 ms, matching the video sampling interval.
COPY_BYTES = 1024**2
RESPONSE_FORMATS = ('nexus-video-phototransduction-1', 'nexus-video-phototransduction-2')


def recording_durations(report):
    """Video duration and optional connected pre-exposure, in integer ms."""
    duration = report.get('exposure_ms')
    if (report.get('format') not in RESPONSE_FORMATS or isinstance(duration, bool)
            or not isinstance(duration, int) or duration < 1):
        raise ValueError('Invalid receptor recording format or video duration')
    prefix = 0
    if report['format'] == RESPONSE_FORMATS[1]:
        pre = report.get('pre_exposure', {})
        prefix = pre.get('duration_ms')
        if (isinstance(prefix, bool) or not isinstance(prefix, int) or prefix < 1
                or pre.get('feeds_brain') is not True
                or report.get('playback_duration_ms') != prefix+duration):
            raise ValueError('Connected pre-exposure needs a complete recorded prefix')
    elif report.get('pre_exposure', {}).get('feeds_brain', False):
        raise ValueError('Connected pre-exposure requires a version 2 recording')
    return duration, prefix


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class _VoltageRecording:
    """Validated numeric samples in an automatically removed temporary file.

    Read both earlier row-ordered and newer column-ordered recordings. A mapping
    exists only while copying a playback window, so visited pages do not remain
    mapped for the lifetime of a long recording. Returned arrays own their data.
    """
    def __init__(self, response_path, shape, curve, name='voltage_mv'):
        self.file = TemporaryFile(prefix='nexus-receptor-playback-')
        try:
            with ZipFile(response_path) as archive, archive.open(name+'.npy') as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    read_header = np.lib.format.read_array_header_1_0
                elif version == (2, 0):
                    read_header = np.lib.format.read_array_header_2_0
                else:
                    raise ValueError('Unsupported receptor voltage array format')
                actual_shape, fortran, self.dtype = read_header(stream)
                if actual_shape != shape or self.dtype.kind not in 'fiu':
                    raise ValueError('Response dimensions or voltage type disagree with their report')
                self.shape = shape
                self.order = 'F' if fortran else 'C'
                self.nbytes = shape[0]*shape[1]*self.dtype.itemsize
                remaining = self.nbytes
                chunk_bytes = max(1, COPY_BYTES//self.dtype.itemsize)*self.dtype.itemsize
                while remaining:
                    raw = stream.read(min(chunk_bytes, remaining))
                    if not raw or len(raw) % self.dtype.itemsize:
                        raise ValueError('Incomplete receptor voltage recording')
                    # Check every sample before enabling playback, in bounded
                    # blocks. Release is recomputed only for the current window.
                    curve.evaluate(np.frombuffer(raw, dtype=self.dtype))
                    self.file.write(raw)
                    remaining -= len(raw)
                if stream.read(1):
                    raise ValueError('Unexpected data after the receptor voltage recording')
            self.file.flush()
        except BaseException:
            self.close()
            raise

    def read(self, start, stop):
        if self.file.closed:
            raise ValueError('Receptor recording is closed')
        with mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            return np.array(np.ndarray(self.shape, dtype=self.dtype, buffer=mapped,
                                       order=self.order)[start:stop],
                            dtype=np.float64, order='C', copy=True)

    def close(self):
        self.file.close()


class ReceptorPlayback:
    def __init__(self, report_path, curve_path, brain, video_path):
        self.report_path = Path(report_path).expanduser().resolve()
        self.curve_path = Path(curve_path).expanduser().resolve()
        self.response_path = self.report_path.parent/'response.npz'
        report = json.loads(self.report_path.read_text())
        if (report.get('format') not in RESPONSE_FORMATS
                or report.get('success') is not True
                or report.get('membrane_dt_ms') != DT_MS
                or report.get('video_input_fps') != VIDEO_FPS
                or not brain.graph.registry_sha256
                or report.get('registry_sha256') != brain.graph.registry_sha256):
            raise ValueError('Choose a receptor response for this brain pack and its 0.1 ms / 20 fps clocks')
        if file_hash(video_path) != report.get('source_sha256'):
            raise ValueError('Load the same video that generated this receptor response')
        with np.load(self.response_path, allow_pickle=False) as data:
            self.ids = data['ids'].tolist()
        duration, prefix = recording_durations(report)
        if (self.ids != [r['id'] for r in report['receptors']]
                or not len(self.ids)):
            raise ValueError('Response dimensions or neuron IDs disagree with their report')
        self.curve = ReleaseCurve.load(self.curve_path)
        self.initial_voltage_mv = np.asarray(report.get('initial_voltage_mv',
                                             [self.curve.reference_mv]*len(self.ids)), dtype=np.float64)
        if self.initial_voltage_mv.shape != (len(self.ids),):
            raise ValueError('Supply one initial voltage per recorded receptor')
        self.curve.evaluate(self.initial_voltage_mv)
        self.initial_voltage_mv.setflags(write=False)
        self.pre_exposure = report.get('pre_exposure', {'duration_ms': 0})
        self.report_hash = file_hash(self.report_path)
        self.response_hash = file_hash(self.response_path)
        self.curve_hash = file_hash(self.curve_path)
        self.source_hash = report['source_sha256']
        self.exposure = {k: report[k] for k in ('white_rate_hz', 'black_rate_hz', 'transfer', 'root_seed')}
        self.video_start = prefix*10
        self._end = (prefix+duration)*10
        self._window_start = 0
        self._window_voltage = self._window_release = None
        self._recording = _VoltageRecording(self.response_path, (duration*10, len(self.ids)), self.curve)
        self._pre_recording = None
        try:
            if prefix:
                self._pre_recording = _VoltageRecording(self.response_path, (self.video_start, len(self.ids)),
                                                        self.curve, 'pre_exposure_voltage_mv')
        except BaseException:
            self.close()
            raise
        self.cursor = 0
        self.end_state = None

    @property
    def end(self):
        return self._end

    @property
    def ended(self):
        return self.cursor == self.end

    @property
    def in_pre_exposure(self):
        return self.cursor < self.video_start

    def _read_window(self, start, stop):
        if start >= self.video_start:
            return self._recording.read(start-self.video_start, stop-self.video_start)
        before = self._pre_recording.read(start, min(stop, self.video_start))
        if stop <= self.video_start:
            return before
        return np.concatenate((before, self._recording.read(0, stop-self.video_start)))

    def input(self, step, ticks):
        if step != self.cursor or ticks < 1 or step + ticks > self.end:
            raise ValueError('Prepared receptor input must follow its recorded clock')
        if (self._window_voltage is None or step < self._window_start
                or step+ticks > self._window_start+len(self._window_voltage)):
            stop = min(self.end, step+max(ticks, PLAYBACK_WINDOW_TICKS))
            voltage = self._read_window(step, stop)
            release = self.curve.evaluate(voltage)
            voltage.setflags(write=False)
            release.setflags(write=False)
            self._window_start = step
            self._window_voltage, self._window_release = voltage, release
        offset = step-self._window_start
        return {'receptor_voltage_mv': self._window_voltage[offset:offset+ticks],
                'receptor_release_hz': self._window_release[offset:offset+ticks]}

    def close(self):
        self._window_voltage = self._window_release = None
        self._recording.close()
        if self._pre_recording is not None:
            self._pre_recording.close()

    def after_step(self, brain):
        self.cursor = brain.step
        if self.ended:
            self.end_state = {'voltage_sha256': hashlib.sha256(brain.v.astype('<f8').tobytes()).hexdigest(),
                              'spike_counts_sha256': hashlib.sha256(brain.counts.astype('<i8').tobytes()).hexdigest()}

    def snapshot(self):
        return {'loaded': True, 'ended': self.ended, 'position_ms': self.cursor * DT_MS,
                'duration_ms': self.end * DT_MS, 'cells': len(self.ids), 'ids': self.ids,
                'phase': 'ended' if self.ended else 'pre_exposure' if self.in_pre_exposure else 'video',
                'video_start_ms': self.video_start * DT_MS,
                'video_position_ms': max(0, self.cursor-self.video_start) * DT_MS,
                'video_duration_ms': (self.end-self.video_start) * DT_MS,
                'report_sha256': self.report_hash, 'response_sha256': self.response_hash,
                'release_curve_sha256': self.curve_hash, 'source_sha256': self.source_hash,
                'exposure': self.exposure, 'release_curve': self.curve.snapshot(),
                'pre_exposure': self.pre_exposure,
                'storage': {'mode': 'temporary disk recording; windowed voltage and release',
                            'voltage_disk_bytes': self._recording.nbytes + (self._pre_recording.nbytes if self._pre_recording else 0),
                            'window_ms': PLAYBACK_WINDOW_TICKS * DT_MS},
                'end_state': self.end_state,
                'mode': 'prepared voltage replay; no live molecular integration or source feedback'}
