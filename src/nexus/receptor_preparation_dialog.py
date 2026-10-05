"""Native preparation controls; molecular computation stays in a spawn worker."""
import multiprocessing as mp
from pathlib import Path
from queue import Empty
from uuid import uuid4

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QProgressBar, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from .brain.receptor_release import ReleaseCurve
from .receptor_preparation import preparation_worker


class ReceptorPreparationDialog(QDialog):
    def __init__(self, video, pack, data_dir, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Prepare receptor response')
        self.setMinimumWidth(460)
        self.video, self.pack = Path(video), Path(pack)
        self.example_curve = self.pack.parent.parent/'experiments/receptor-release-assay-curve.json'
        self.output_dir = Path(data_dir)/'receptor-responses'/uuid4().hex
        self.process = None
        self.result_files = None
        self.progress_messages = []
        self.poll_count = 0
        self.cancel_requested = False
        self.failure = None
        self._terminal = False
        self._queue_closed = False
        layout = QVBoxLayout(self)
        title = QLabel(self.video.name)
        title.setWordWrap(True)
        layout.addWidget(title)
        self.settings = QWidget()
        form = QFormLayout(self.settings)
        form.setContentsMargins(0, 0, 0, 0)
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
        note = QLabel('8 selected receptors · experimental exposure and release settings')
        note.setToolTip('Each cell retains 30,000 microvilli. The selected subset is not the whole eye. '
                        'Exposure and release are uncalibrated assumptions. See Guide.')
        layout.addWidget(note)
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
                    'backend': self.backend.currentData(), 'seed': self.seed.value(),
                    'release_curve': self.curve.text()}
        ctx = mp.get_context('spawn')
        self.events, self.cancel = ctx.Queue(maxsize=64), ctx.Event()
        self.process = ctx.Process(target=preparation_worker, args=(settings, self.events, self.cancel),
                                   name='Fruit Fly Torment Nexus receptor preparation')
        try:
            self.process.start()
        except Exception as error:
            self.failure = str(error)
            self.process = None
            self.events.close()
            self._queue_closed = True
            self.status.setText(self.failure)
            self.status.setStyleSheet('color: #c00000')
            return
        self.settings.setEnabled(False)
        self.start_button.setEnabled(False)
        self.progress.setRange(0, self.duration.value())
        self.status.setStyleSheet('')
        self.status.setText('Preparing the first batch…')
        self.timer.start()

    def poll(self):
        self.poll_count += 1
        finished = not self.process.is_alive()
        if finished:
            self.process.join(timeout=0)
        for _ in range(64):
            try:
                event = self.events.get_nowait()
            except Empty:
                break
            kind = event['kind']
            if kind == 'progress':
                self.progress_messages.append(event)
                self.progress.setValue(event['elapsed_ms'])
                if not self.cancel_requested:
                    self.status.setText('Preparing the first batch…' if not event['elapsed_ms'] else
                                        f"Prepared {event['elapsed_ms']} / {event['duration_ms']} ms")
            elif kind == 'complete':
                self._terminal = True
                if not self.cancel_requested:
                    self.result_files = {k: event[k] for k in ('report_path', 'curve_path')}
                    self.progress.setRange(0, event['exposure_ms'])
                    self.progress.setValue(event['exposure_ms'])
                    self.status.setText(f"Ready · {event['exposure_ms']} ms · {event['cells']} receptors")
            elif kind == 'cancelled':
                self._terminal = True
                self.status.setText('Cancelled. No response loaded.')
            elif kind == 'failed':
                self._terminal = True
                self.failure = event['message']
                self.status.setText(self.failure)
                self.status.setToolTip(event.get('detail', ''))
                self.status.setStyleSheet('color: #c00000')
        if finished:
            self.timer.stop()
            self.events.close()
            self.events.join_thread()
            self._queue_closed = True
            if not self._terminal:
                self.failure = f'Preparation worker exited (code {self.process.exitcode})'
                self.status.setText(self.failure)
                self.status.setStyleSheet('color: #c00000')
            self.cancel_button.setText('Close')
            self.load_button.setEnabled(self.result_files is not None and not self.cancel_requested)
            if self.cancel_requested:
                super().reject()

    def accept(self):
        if self.result_files is not None and not self.process.is_alive():
            super().accept()

    def reject(self):
        if self.process is not None and self.process.is_alive():
            self.cancel_requested = True
            self.cancel.set()
            self.load_button.setEnabled(False)
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
        if not self._queue_closed:
            self.events.cancel_join_thread()
            self.events.close()
            self._queue_closed = True
