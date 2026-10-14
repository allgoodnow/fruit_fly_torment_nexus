"""Native preparation controls; molecular computation stays in a spawn worker."""
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time
from uuid import uuid4

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QProgressBar, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from .brain.receptor_release import ReleaseCurve
from .receptor_preparation import preparation_worker
from .retina import VisualColumns


class ReceptorPreparationDialog(QDialog):
    def __init__(self, video, pack, data_dir, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Prepare receptor response')
        self.setMinimumWidth(460)
        self.video, self.pack = Path(video), Path(pack)
        self.example_curve = self.pack.parent.parent/'experiments/receptor-release-assay-curve.json'
        self.output_dir = Path(data_dir)/'receptor-responses'/uuid4().hex
        self.status_path = self.output_dir.with_suffix('.status.json')
        self.process = None
        self.result_files = None
        self.progress_messages = []
        self.poll_count = 0
        self.cancel_requested = False
        self.failure = None
        self._terminal = False
        self._ready_status = None
        self._status_sequence = 0
        self._last_progress_text = ''
        self._last_status_at = None
        self._cancel_at = None
        self._terminate_requested = False
        layout = QVBoxLayout(self)
        title = QLabel(self.video.name)
        title.setWordWrap(True)
        layout.addWidget(title)
        self.settings = QWidget()
        form = QFormLayout(self.settings)
        form.setContentsMargins(0, 0, 0, 0)
        manifest = json.loads((self.pack/'manifest.json').read_text())
        registry = json.loads((self.pack/manifest['circuit_registry']).read_text())
        mapping = VisualColumns(registry)
        self.mapped_cells = sum(len(mapping.ids[side]) for side in ('L', 'R'))
        self.coverage = QComboBox()
        self.coverage.addItem('8 receptors · quadrant sample', 'sample8')
        self.coverage.addItem('64 receptors · spatial sample', 'sample64')
        self.coverage.addItem(f'All mapped receptors ({self.mapped_cells:,})', 'mapped')
        form.addRow('Receptor coverage', self.coverage)
        self.duration = QSpinBox()
        self.duration.setRange(1, 60000)
        self.duration.setValue(500)
        self.duration.setSuffix(' ms')
        form.addRow('Recording duration', self.duration)
        self.white = QDoubleSpinBox()
        self.white.setRange(0., 1000000.)
        self.white.setDecimals(2)
        self.white.setValue(30000.)
        self.black = QDoubleSpinBox()
        self.black.setRange(0., 1000000.)
        self.black.setDecimals(2)
        form.addRow('White exposure (photons/s)', self.white)
        form.addRow('Black exposure (photons/s)', self.black)
        self.transfer = QComboBox()
        self.transfer.addItem('sRGB assumption', 'srgb')
        self.transfer.addItem('Linear RGB assumption', 'linear')
        form.addRow('Video intensity', self.transfer)
        self.backend = QComboBox()
        self.backend.addItem('CPU', 'cpu')
        self.backend.addItem('NVIDIA GPU (CUDA)', 'cuda')
        form.addRow('Prepare using', self.backend)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(88300)
        form.addRow('Repeatable seed', self.seed)
        curve_row = QHBoxLayout()
        self.curve = QLineEdit()
        self.curve.setReadOnly(True)
        self.curve.setPlaceholderText('Choose an explicit release curve')
        choose = QPushButton('Choose…')
        choose.clicked.connect(self.choose_curve)
        curve_row.addWidget(self.curve, 1)
        curve_row.addWidget(choose)
        form.addRow('Release curve', curve_row)
        layout.addWidget(self.settings)
        note = QLabel('30,000 microvilli per cell · experimental exposure and release')
        note.setToolTip('Each cell retains 30,000 microvilli. The selected subset is not the whole eye. '
                        'Exposure and release are uncalibrated assumptions. See Guide.')
        layout.addWidget(note)
        self.coverage.currentIndexChanged.connect(self.update_duration_limit)
        self.update_duration_limit()
        self.progress = QProgressBar()
        self.progress.setRange(0, self.duration.value())
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.status = QLabel('Preparation keeps the simulation paused at time zero.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.start_button = QPushButton('Prepare')
        self.start_button.clicked.connect(self.start)
        self.load_button = QPushButton('Load response')
        self.load_button.setEnabled(False)
        self.load_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.reject)
        for button in (self.start_button, self.load_button, self.cancel_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.poll)

    def update_duration_limit(self):
        cells = {'sample8': 8, 'sample64': 64, 'mapped': self.mapped_cells}[self.coverage.currentData()]
        # Playback retains both voltage and release arrays, each float64 at 0.1 ms.
        # Keep their combined payload below 1 GiB; preparation uses disk arrays.
        self.duration.setMaximum(min(60000, (1024**3)//(cells*10*8*2)))
        self.duration.setToolTip('Duration is limited to keep playback voltage and release arrays '
                                 'within 1 GiB. More receptors require more preparation time. See Guide.')

    def choose_curve(self):
        initial = str(self.example_curve) if self.example_curve.is_file() else ''
        path, _ = QFileDialog.getOpenFileName(self, 'Choose the explicit release curve (example is unfitted)', initial, 'JSON (*.json)')
        if path:
            self.curve.setText(path)

    def start(self):
        if self.process is not None:
            return
        try:
            if not self.curve.text():
                raise ValueError('Choose an explicit release curve before preparing')
            ReleaseCurve.load(self.curve.text())
            if self.black.value() > self.white.value():
                raise ValueError('Black exposure must not exceed white exposure')
        except (OSError, ValueError, KeyError) as error:
            self.status.setText(str(error))
            self.status.setStyleSheet('color: #c00000')
            return
        settings = {'video': str(self.video), 'pack': str(self.pack), 'output_dir': str(self.output_dir),
                    'white_rate_hz': self.white.value(), 'black_rate_hz': self.black.value(),
                    'transfer': self.transfer.currentData(), 'duration_ms': self.duration.value(),
                    'coverage': self.coverage.currentData(),
                    'backend': self.backend.currentData(), 'seed': self.seed.value(),
                    'release_curve': self.curve.text()}
        ctx = mp.get_context('spawn')
        self.failure = None
        self.cancel = ctx.Event()
        self.process = ctx.Process(target=preparation_worker, args=(settings, str(self.status_path), self.cancel),
                                   name='Fruit Fly Torment Nexus receptor preparation')
        try:
            self.process.start()
        except Exception as error:
            self.failure = str(error)
            self.process = None
            self.status.setText(self.failure)
            self.status.setStyleSheet('color: #c00000')
            return
        self.settings.setEnabled(False)
        self.start_button.setEnabled(False)
        self.progress.setRange(0, 0)
        self.status.setStyleSheet('')
        self._last_progress_text = 'Starting preparation…'
        self._last_status_at = time.monotonic()
        self.status.setText(self._last_progress_text)
        self.status.setToolTip(f'Job status: {self.status_path}')
        self.timer.start()

    def poll(self):
        self.poll_count += 1
        finished = not self.process.is_alive()
        if finished:
            self.process.join(timeout=0)
        try:
            event = json.loads(self.status_path.read_text())
            if (not isinstance(event, dict) or event.get('format') != 'nexus-receptor-preparation-status-1'
                    or not isinstance(event.get('sequence'), int)
                    or event.get('kind') not in ('progress', 'complete', 'cancelled', 'failed')):
                raise ValueError('Invalid preparation status')
        except (OSError, ValueError) as error:
            event = None
            if finished:
                self.failure = f'Cannot read preparation outcome: {error}'
        if event is not None and event['sequence'] > self._status_sequence:
            self._status_sequence = event['sequence']
            self._last_status_at = time.monotonic()
            kind = event['kind']
            if kind == 'progress':
                self.progress_messages.append(event)
                if not self.cancel_requested:
                    if event['phase'] == 'starting':
                        self.progress.setRange(0, 0)
                        self.status.setText('Starting preparation…')
                    elif event['phase'] == 'decoding':
                        self.progress.setRange(0, 0)
                        self.status.setText(f"Reading video · {event['elapsed_ms']} / {event['duration_ms']} ms")
                    else:
                        self.progress.setRange(0, 1000)
                        self.progress.setValue(event['work_done']*1000//event['work_total'])
                        if event['phase'] == 'exporting':
                            self.status.setText('Saving the response…')
                        else:
                            action = 'Initializing' if event['phase'] == 'initializing' else 'Computing'
                            self.status.setText(f"{action} receptors {event['first_cell']+1}–{event['last_cell']} "
                                                f"/ {event['cells']} · {event['elapsed_ms']} / {event['duration_ms']} ms")
                    self._last_progress_text = self.status.text()
            elif kind == 'complete':
                self._terminal = True
                if not self.cancel_requested:
                    self.result_files = {k: event[k] for k in ('report_path', 'curve_path')}
                    self.progress.setRange(0, event['exposure_ms'])
                    self.progress.setValue(event['exposure_ms'])
                    self._ready_status = f"Ready · {event['exposure_ms']} ms · {event['cells']} receptors"
                    self.status.setText('Response saved · waiting for the worker to finish…')
            elif kind == 'cancelled':
                self._terminal = True
                self.status.setText('Cancelled. No response loaded.')
            elif kind == 'failed':
                self._terminal = True
                self.failure = event['message']
                self.status.setText(self.failure)
                self.status.setToolTip(event.get('detail', ''))
                self.status.setStyleSheet('color: #c00000')
        if (not finished and self.cancel_requested and not self._terminate_requested
                and time.monotonic()-self._cancel_at >= 2.):
            self._terminate_requested = True
            self.process.terminate()
            self.status.setText('Stopping preparation…')
        elif (not finished and not self.cancel_requested and not self._terminal
              and self._last_status_at is not None and time.monotonic()-self._last_status_at >= 30.):
            quiet_seconds = int(time.monotonic()-self._last_status_at)
            self.status.setText(f'{self._last_progress_text} · no update for {quiet_seconds} s (Cancel is available)')
        if finished:
            self.timer.stop()
            if self.cancel_requested:
                self.result_files = None
                self.status.setText('Cancelled. No response loaded.')
                self.cleanup_interrupted_work()
            elif self.failure is not None or not self._terminal or self.process.exitcode != 0:
                self.result_files = None
                self.failure = self.failure or f'Preparation worker exited without a ready outcome (code {self.process.exitcode})'
                self.status.setText(self.failure)
                self.status.setStyleSheet('color: #c00000')
                if not self.status.toolTip():
                    self.status.setToolTip(f'Job status: {self.status_path}')
            elif self.result_files is not None:
                self.status.setText(self._ready_status)
            self.cancel_button.setText('Close')
            self.load_button.setEnabled(self.result_files is not None and not self.cancel_requested)
            if self.failure is not None:
                self.cleanup_interrupted_work()
            if self.cancel_requested:
                super().reject()

    def accept(self):
        if self.result_files is not None and not self.process.is_alive():
            super().accept()

    def reject(self):
        if self.process is not None and self.process.is_alive():
            self.cancel_requested = True
            if self._cancel_at is None:
                self._cancel_at = time.monotonic()
            self.cancel.set()
            self.load_button.setEnabled(False)
            self.cancel_button.setEnabled(False)
            self.status.setText('Cancelling after the current batch…')
        else:
            super().reject()

    def closeEvent(self, event):
        if self.process is not None and self.process.is_alive():
            self.reject()
            event.ignore()
        else:
            event.accept()

    def shutdown(self):
        self.timer.stop()
        if self.process is None:
            return
        if self.process.is_alive():
            self.cancel.set()
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
        if not self.process.is_alive() and (self.cancel_requested or self.process.exitcode != 0):
            self.cleanup_interrupted_work()

    def cleanup_interrupted_work(self):
        # Only this dialog's unique job directory, after its worker has exited.
        for path in self.output_dir.glob('.receptor-work-*'):
            if path.is_dir() and not path.is_symlink():
                try:
                    shutil.rmtree(path)
                except OSError:
                    pass  # Preserve an inaccessible scratch directory; never affect readiness.
