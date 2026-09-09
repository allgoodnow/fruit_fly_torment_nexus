"""Native OpenGL view of specimen-matched cell anchors and simulated spike activity."""
import numpy as np
from PySide6.QtGui import QVector3D
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from pyqtgraph.opengl import GLViewWidget, GLScatterPlotItem
from OpenGL import GL

from .anatomy import Anatomy


class BrainView(QWidget):
    def __init__(self, directory):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.heading = QLabel('NEURAL ANATOMY')
        row.addWidget(self.heading, 1)
        home = QPushButton('Home')
        fit = QPushButton('Fit all')
        row.addWidget(home)
        row.addWidget(fit)
        layout.addLayout(row)
        self.gl = GLViewWidget()
        self.gl.setMinimumSize(300, 280)
        self.gl.setBackgroundColor('white')
        layout.addWidget(self.gl, 1)
        self.caption = QLabel('Loading cell positions…')
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
        hint = QLabel('Drag to orbit · Scroll to zoom · Red: spikes in last 150 ms\nBlack: stimulated targets · Each point is a cell anchor')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.anatomy = None
        self.active_count = 0
        self.unlocated_active = 0
        self.tick = 0
        try:
            self.anatomy = Anatomy(directory)
        except (OSError, ValueError, KeyError) as error:
            self.caption.setText(f'Anatomy unavailable: {error}')
            home.setEnabled(False)
            fit.setEnabled(False)
            return
        a = self.anatomy
        self.heading.setText('BRAIN + VNC / MALECNS v1.0' if a.manifest['snapshot'] == 'male-cns:v1.0' else 'BRAIN / FLYWIRE v630')
        self.cloud = GLScatterPlotItem(pos=a.positions[a.valid], color=(.22, .24, .25, .12), size=1.25, pxMode=True, glOptions='translucent')
        self.gl.addItem(self.cloud)
        self.targets = GLScatterPlotItem(pos=np.empty((0, 3)), color=(0, 0, 0, .9), size=9, glOptions='translucent')
        self.gl.addItem(self.targets)
        self.spikes = GLScatterPlotItem(pos=np.empty((0, 3)), color=(.72, .025, .04, 1), size=5, glOptions='translucent')
        self.gl.addItem(self.spikes)
        # The black target marker must not occlude its red spike centre at the
        # exact same coordinate. Inner activity remains visible as an overlay.
        self.targets.updateGLOptions({GL.GL_DEPTH_TEST: False})
        self.spikes.updateGLOptions({GL.GL_DEPTH_TEST: False})
        home.clicked.connect(self.home)
        fit.clicked.connect(self.fit_all)
        self.home()
        self.caption.setText(f'{a.valid.sum():,} positioned cells · {(~a.valid).sum()} unlocated\nBrain not loaded · no simulated activity yet')

    def home(self):
        male = self.anatomy.manifest['snapshot'] == 'male-cns:v1.0'
        self.gl.setCameraPosition(pos=QVector3D(0, 0, 0), distance=self.anatomy.span*(1.5 if male else 1.15),
                                  elevation=60 if male else 8, azimuth=90 if male else -90)

    def fit_all(self):
        points = self.anatomy.positions[self.anatomy.valid]
        low, high = points.min(axis=0), points.max(axis=0)
        centre = (low + high) / 2
        self.gl.setCameraPosition(pos=QVector3D(*map(float, centre)), distance=float(np.linalg.norm(high-low))*1.3)

    def update_snapshot(self, packet):
        if self.anatomy is None:
            return
        if (packet.get('dataset', self.anatomy.manifest['snapshot']) != self.anatomy.manifest['snapshot']
                or packet['neuron_order_sha256'] != self.anatomy.neuron_order_sha256):
            self.spikes.setData(pos=np.empty((0, 3)))
            self.targets.setData(pos=np.empty((0, 3)))
            self.active_count = self.unlocated_active = 0
            self.caption.setText('Activity unavailable: loaded network uses a different neuron order')
            return
        self.tick = packet['tick']
        positions, colors, sizes, self.unlocated_active = self.anatomy.activity(
            self.tick, packet['active_indices'], packet['active_steps'])
        self.active_count = len(positions)
        self.spikes.setData(pos=positions, color=colors, size=sizes)
        indices = [self.anatomy.lookup[int(root)] for root in packet['stimulated_ids']]
        indices = [i for i in indices if self.anatomy.valid[i]]
        self.targets.setData(pos=self.anatomy.positions[indices])
        self.caption.setText(f"{'Running' if packet['running'] else 'Paused'} · brain {packet['sim_time']:.3f} s · {self.active_count:,} recently active\n"
                             f"{self.anatomy.valid.sum():,} positioned · {self.unlocated_active} active cells unlocated")

    def diagnostics(self):
        return {'loaded': self.anatomy is not None, 'mapped': int(self.anatomy.valid.sum()) if self.anatomy else 0,
                'active_count': self.active_count, 'unlocated_active': self.unlocated_active, 'tick': self.tick,
                'dataset': self.anatomy.manifest['snapshot'] if self.anatomy else None,
                'representation': self.anatomy.manifest.get('representation', 'v630 cell anchors, not neuron morphology') if self.anatomy else None, 'fade_sim_ms': 150}
