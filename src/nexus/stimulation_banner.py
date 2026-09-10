from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel

from .stimulation import active_stimulation


class StimulationBanner(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName('stimulationBanner')
        self.setStyleSheet('#stimulationBanner { border-top: 1px solid #d9dfe3; border-bottom: 1px solid #d9dfe3; }')
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 10, 0, 10)
        heading = QLabel('STIMULATION')
        row.addWidget(heading)
        row.addSpacing(14)
        self.readout = QLabel('LOADING')
        self.readout.setStyleSheet('font-size: 24px; font-weight: 700; color: #c00000;')
        self.readout.setWordWrap(True)
        row.addWidget(self.readout, 1)
        self.clock = QLabel('')
        self.clock.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.clock)
        self.labels = ()

    def update_snapshot(self, packet):
        self.labels = active_stimulation(packet)
        self.readout.setText(' + '.join(self.labels) or 'NONE')
        state = 'RUNNING' if packet['running'] else 'PAUSED'
        self.clock.setText(f"{state}  ·  {packet['sim_time']:.3f} s")

    def unavailable(self):
        self.labels = ()
        self.readout.setText('UNAVAILABLE')
        self.clock.setText('STOPPED')
