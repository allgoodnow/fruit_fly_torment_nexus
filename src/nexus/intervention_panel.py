"""Native experiment controls; all effects use the existing brain command queue."""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QGroupBox,
                              QLabel, QPushButton, QSpinBox, QCheckBox)
from .brain.scenarios import scenario_protocol


class InterventionPanel(QWidget):
    def __init__(self, brain_panel, *, coupled=True):
        super().__init__()
        self.brain_panel = brain_panel
        self.coupled = coupled
        layout = QVBoxLayout(self)
        self.status = QLabel('Waiting for brain…' if coupled else 'Load the brain in the Brain tab first.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.controls = QWidget()
        controls = QVBoxLayout(self.controls)
        controls.setContentsMargins(0, 0, 0, 0)
        phases = QGroupBox('Prepare, then run')
        form = QFormLayout(phases)
        self.durations = []
        for label, value in [('Baseline', 100), ('Stimulus', 500), ('After release', 500)]:
            spin = QSpinBox()
            spin.setRange(1, 60000)
            spin.setValue(value)
            spin.setSuffix(' ms')
            form.addRow(label, spin)
            self.durations.append(spin)
        controls.addWidget(phases)
        self.buttons = {}
        definitions = [('defensive', 'Fear'), ('aversion', 'Pain'), ('seizure', 'Seizure')]
        for name, title in definitions:
            button = QPushButton('Run '+title.lower())
            button.setMinimumHeight(36)
            button.clicked.connect(lambda checked=False, name=name: self.run(name))
            self.buttons[name] = button
            controls.addWidget(button)
        heat = QGroupBox('Heat')
        form = QFormLayout(heat)
        self.temperature = QSpinBox()
        self.temperature.setRange(20, 100)
        self.temperature.setValue(40)
        self.temperature.setSuffix(' °C nominal')
        form.addRow(self.temperature)
        for name, label in [('heat', 'Run heat input'), ('heat_overload', 'Run boiling')]:
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, name=name: self.run(name))
            self.buttons[name] = button
            form.addRow(button)
        controls.addWidget(heat)
        self.motor_enabled = QCheckBox('Enable motor response')
        self.motor_enabled.setChecked(True)
        self.motor_enabled.setVisible(coupled)
        self.motor_enabled.toggled.connect(lambda value: brain_panel.send('motor_effects_enabled', value))
        controls.addWidget(self.motor_enabled)
        self.resume = QCheckBox('Resume free behavior after experiment')
        self.resume.setChecked(coupled)
        self.resume.setVisible(coupled)
        self.resume.toggled.connect(lambda value: brain_panel.send('resume_after_protocol', value))
        controls.addWidget(self.resume)
        self.pause = QPushButton('Pause')
        self.pause.clicked.connect(lambda: brain_panel.send('running', False))
        controls.addWidget(self.pause)
        self.release = QPushButton('Release stimulation')
        self.release.clicked.connect(lambda: brain_panel.send('release'))
        controls.addWidget(self.release)
        reset = QPushButton('Reset body + brain' if coupled else 'Reset brain')
        reset.clicked.connect(lambda: brain_panel.send('reset'))
        controls.addWidget(reset)
        self.controls.setEnabled(False)
        layout.addWidget(self.controls)
        layout.addStretch()
        brain_panel.snapshot.connect(self.accept_snapshot)

    def run(self, name):
        if name == 'heat_overload':
            self.temperature.setValue(100)
        pre, stimulus, recovery = [spin.value() for spin in self.durations]
        self.brain_panel.send('protocol', scenario_protocol(name, baseline_ms=pre, stimulus_ms=stimulus,
                                                          recovery_ms=recovery, celsius=self.temperature.value(),
                                                          dataset=self.brain_panel.config.snapshot,
                                                          pain_circuit=self.brain_panel.config.pain_circuit))

    def accept_snapshot(self, packet):
        self.controls.setEnabled(not self.brain_panel.failed)
        effects = packet.get('motor_effects', {})
        self.motor_enabled.blockSignals(True)
        self.motor_enabled.setChecked(effects.get('enabled', False))
        self.motor_enabled.blockSignals(False)
        self.resume.blockSignals(True)
        self.resume.setChecked(packet.get('resume_after_protocol', False))
        self.resume.blockSignals(False)
        self.status.setText(f"{'Running' if packet['running'] else 'Paused'} · {packet['sim_time']:.3f} s")
