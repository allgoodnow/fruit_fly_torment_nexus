"""Prepared receptor voltages on the shared brain/body/video clock.

This imports an offline exposure response, not an on-the-fly molecular model.
Playback ends at the last recorded voltage; no continuation is invented.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .brain.receptor_release import ReleaseCurve
from .brain.runtime import DT_MS
from .video import VIDEO_FPS


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class ReceptorPlayback:
    def __init__(self, report_path, curve_path, brain, video_path):
        self.report_path = Path(report_path).expanduser().resolve()
        self.curve_path = Path(curve_path).expanduser().resolve()
        self.response_path = self.report_path.parent/'response.npz'
        report = json.loads(self.report_path.read_text())
        if (report.get('format') != 'nexus-video-phototransduction-1'
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
            self.voltage_mv = data['voltage_mv'].copy()
        duration = report.get('exposure_ms')
        if (isinstance(duration, bool) or not isinstance(duration, int) or duration < 1
                or self.ids != [r['id'] for r in report['receptors']]
                or self.voltage_mv.shape != (duration * 10, len(self.ids))
                or not len(self.ids) or not np.isfinite(self.voltage_mv).all()):
            raise ValueError('Response dimensions or neuron IDs disagree with their report')
        self.curve = ReleaseCurve.load(self.curve_path)
        self.release_hz = self.curve.evaluate(self.voltage_mv)
        self.report_hash = file_hash(self.report_path)
        self.response_hash = file_hash(self.response_path)
        self.curve_hash = file_hash(self.curve_path)
        self.source_hash = report['source_sha256']
        self.exposure = {k: report[k] for k in ('white_rate_hz', 'black_rate_hz', 'transfer', 'root_seed')}
        self.voltage_mv.setflags(write=False)
        self.release_hz.setflags(write=False)
        self.cursor = 0
        self.end_state = None

    @property
    def end(self):
        return len(self.voltage_mv)

    @property
    def ended(self):
        return self.cursor == self.end

    def input(self, step, ticks):
        if step != self.cursor or ticks < 1 or step + ticks > self.end:
            raise ValueError('Prepared receptor input must follow its recorded clock')
        return {'receptor_voltage_mv': self.voltage_mv[step:step+ticks],
                'receptor_release_hz': self.release_hz[step:step+ticks]}

    def after_step(self, brain):
        self.cursor = brain.step
        if self.ended:
            self.end_state = {'voltage_sha256': hashlib.sha256(brain.v.astype('<f8').tobytes()).hexdigest(),
                              'spike_counts_sha256': hashlib.sha256(brain.counts.astype('<i8').tobytes()).hexdigest()}

    def snapshot(self):
        return {'loaded': True, 'ended': self.ended, 'position_ms': self.cursor * DT_MS,
                'duration_ms': self.end * DT_MS, 'cells': len(self.ids), 'ids': self.ids,
                'report_sha256': self.report_hash, 'response_sha256': self.response_hash,
                'release_curve_sha256': self.curve_hash, 'source_sha256': self.source_hash,
                'exposure': self.exposure, 'release_curve': self.curve.snapshot(),
                'end_state': self.end_state,
                'mode': 'prepared voltage replay; no live molecular integration or source feedback'}
