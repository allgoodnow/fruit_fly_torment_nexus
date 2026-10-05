"""Compact display of the visual frame actually supplied to the brain."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QCheckBox, QComboBox


class VisionPreview(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.setMinimumSize(240, 135)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.white)
        if self.image is None:
            painter.setPen(Qt.GlobalColor.black)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'No visual input')
        else:
            size = self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
            x, y = (self.width() - size.width()) // 2, (self.height() - size.height()) // 2
            painter.drawImage(x, y, self.image.scaled(size, Qt.AspectRatioMode.KeepAspectRatio,
                                                    Qt.TransformationMode.SmoothTransformation))


class VisionPanel(QWidget):
    command = Signal(str, object)

    def __init__(self):
        super().__init__()
        self.setMinimumWidth(395)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(3)
        title = QLabel("FLY’S VISION")
        title.setStyleSheet('font-weight: 600')
        layout.addWidget(title)
        content = QHBoxLayout()
        content.setSpacing(8)
        controls = QVBoxLayout()
        controls.setSpacing(4)
        content.addLayout(controls)
        self.preview = VisionPreview()
        content.addWidget(self.preview, 1)
        layout.addLayout(content, 1)
        self.load = QPushButton('Load video…')
        self.load.clicked.connect(self.pick_video)
        self.eyes = QPushButton('Use eyes')
        self.eyes.clicked.connect(lambda: self.command.emit('vision_eyes', None))
        self.restart = QPushButton('Restart')
        self.restart.clicked.connect(lambda: self.command.emit('vision_restart', None))
        for button in [self.load, self.eyes, self.restart]:
            controls.addWidget(button)
        self.feed = QCheckBox('Feed to brain')
        self.feed.setToolTip('Visual input to R1-R6 photoreceptors. Enable, then run the simulation. See Guide.')
        self.feed.toggled.connect(lambda checked: self.command.emit('eye_feedback', checked))
        controls.addWidget(self.feed)
        self.mapping = QComboBox()
        self.mapping.addItem('Brightness', 'pooled')
        self.mapping.addItem('Spatial (exp.)', 'spatial')
        self.mapping.addItem('Receptor replay (exp.)', 'receptor_replay')
        # This mode is loaded from Brain controls; selecting it alone cannot
        # supply a recorded voltage waveform or its release assumption.
        self.mapping.model().item(2).setEnabled(False)
        self.mapping.setToolTip('Spatial video input uses inferred columns and an uncalibrated image projection. Changing mode pauses and releases visual input. See Guide.')
        self.mapping.currentIndexChanged.connect(lambda: self.command.emit('vision_mapping', self.mapping.currentData()))
        controls.addWidget(self.mapping)
        self.adaptation = QCheckBox('Adapt to light')
        self.adaptation.setToolTip('Experimental sensitivity reduction with sustained light. Unfitted parameters; see Guide. Changing this pauses and releases visual input.')
        self.adaptation.toggled.connect(lambda checked: self.command.emit('vision_adaptation', checked))
        controls.addWidget(self.adaptation)
        controls.addStretch(1)
        self.status = QLabel('Disabled')
        self.status.setToolTip('Playback follows simulation time, without audio.')
        layout.addWidget(self.status)
        self.setEnabled(False)

    def pick_video(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Choose what the fly sees', '',
                                              'Video (*.mp4 *.m4v *.mov *.mkv *.avi *.webm);;All files (*)')
        if path:
            self.command.emit('vision_video', path)

    def update_snapshot(self, state, pixels, running):
        self.setEnabled(state.get('available', False))
        self.feed.blockSignals(True)
        self.feed.setChecked(state.get('enabled', False))
        self.feed.blockSignals(False)
        video = state.get('source') == 'video'
        replay = state.get('receptor_playback', {})
        prepared = replay.get('loaded', False)
        self.mapping.blockSignals(True)
        self.mapping.setCurrentIndex(2 if prepared else 1 if state.get('mapping_mode') == 'spatial' else 0)
        self.mapping.blockSignals(False)
        self.mapping.setEnabled(video and state.get('spatial_available', False) and not prepared)
        self.adaptation.blockSignals(True)
        self.adaptation.setChecked(state.get('adaptation', {}).get('selected', False))
        self.adaptation.blockSignals(False)
        self.adaptation.setEnabled(not prepared)
        self.load.setEnabled(not prepared)
        self.eyes.setEnabled(not prepared)
        self.restart.setEnabled(video and not prepared)
        self.feed.setEnabled(not replay.get('ended', False))
        self.restart.setToolTip('Reset body + brain to rewind a prepared receptor recording.' if prepared else '')
        if pixels is not None:
            height, width, _ = pixels.shape
            self.preview.image = QImage(pixels.data, width, height, pixels.strides[0],
                                        QImage.Format.Format_RGB888).copy()
        else:
            self.preview.image = None
        self.preview.update()
        name = state.get('video_name') if video else 'Left / right eyes'
        status = 'Ended' if state.get('video_ended') or replay.get('ended') else 'Playing' if state.get('enabled') and running else 'Paused'
        if state.get('error'):
            status = 'Video error'
        position = f" · {state.get('video_frame', 0) / 20:.2f} s" if video else ''
        text = status + position + (' · ' + name if video and name else '')
        if prepared:
            text = f"{status} · {replay['position_ms']/1000:.2f} / {replay['duration_ms']/1000:.2f} s · {replay['cells']} receptors"
        self.status.setText(self.status.fontMetrics().elidedText(text, Qt.TextElideMode.ElideMiddle,
                                                               max(180, self.width() - 16)))
        self.status.setToolTip(state.get('error') or ('Prepared receptor recording · ' + (name or '') +
                               '\nUnfitted release curve; pauses at its endpoint. See Guide.' if prepared else name or ''))
