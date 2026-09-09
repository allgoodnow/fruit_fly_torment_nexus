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
        note = QLabel('EXPERIMENTS\nNeural responses are measurable.\nSubjective fear and pain are unverified.')
        note.setWordWrap(True)
        layout.addWidget(note)
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
        definitions = [('defensive', 'Fear / threat', 'Looming-pathway input; engineered escape response.'),
                       ('aversion', 'Pain pathway / candidate proxy',
                        ('Stimulates GNG121 (CB0059 counterpart) candidates.\nPeripheral nociception is not yet connected.' if brain_panel.config.experimental else 'Stimulates CB0059 central aversion candidates.\nPeripheral nociception is not yet connected.')),
                       ('seizure', 'Seizure-like experiment',
                        'Reduced inhibition with neural input.\nMotor disruption is an authored decoder.')]
        for name, title, detail in definitions:
            group = QGroupBox(title)
            form = QFormLayout(group)
            explanation = QLabel(detail)
            explanation.setWordWrap(True)
            form.addRow(explanation)
            button = QPushButton('Run '+('threat experiment' if name == 'defensive' else
                                        'aversion proxy' if name == 'aversion' else 'network / motor disruption'))
            button.clicked.connect(lambda checked=False, name=name: self.run(name))
            self.buttons[name] = button
            form.addRow(button)
            controls.addWidget(group)
        heat = QGroupBox('Heat scenario')
        form = QFormLayout(heat)
        self.temperature = QSpinBox()
        self.temperature.setRange(20, 100)
        self.temperature.setValue(40)
        self.temperature.setSuffix(' °C nominal')
        form.addRow(self.temperature)
        note = QLabel('Warmth-cell input saturates at 40°C.\n100°C is a scenario label; tissue damage and\nthermal short-circuits are not modeled.')
        note.setWordWrap(True)
        form.addRow(note)
        for name, label in [('heat', 'Run heat input'), ('heat_overload', 'Run 100°C + network overload')]:
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, name=name: self.run(name))
            self.buttons[name] = button
            form.addRow(button)
        controls.addWidget(heat)
        self.motor_enabled = QCheckBox('Enable escape / disruption motor proxy')
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
        self.release = QPushButton('Release inputs and network overlays')
        self.release.clicked.connect(lambda: brain_panel.send('release'))
        controls.addWidget(self.release)
        note = QLabel('Release preserves neural state; activity and\nmovement may continue. Reset reinitializes both.')
        note.setWordWrap(True)
        controls.addWidget(note)
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
                                                          dataset=self.brain_panel.config.snapshot))

    def accept_snapshot(self, packet):
        self.controls.setEnabled(not self.brain_panel.failed)
        effects = packet.get('motor_effects', {})
        self.motor_enabled.blockSignals(True)
        self.motor_enabled.setChecked(effects.get('enabled', False))
        self.motor_enabled.blockSignals(False)
        self.resume.blockSignals(True)
        self.resume.setChecked(packet.get('resume_after_protocol', False))
        self.resume.blockSignals(False)
        circuits = ', '.join(packet['circuit_inputs']) or 'none'
        temperature = packet['nominal_temperature_c']
        self.status.setText(f"{'Running' if packet['running'] else 'Paused'} · {packet['sim_time']:.3f} s\n"
                            f"Active circuit inputs: {circuits}\n"
                            f"Inhibition remaining: {packet['inhibition_gain']*100:g} %\n"
                            f"Heat: {str(temperature)+'°C nominal' if temperature is not None else 'off'}")
        if effects:
            self.status.setText(self.status.text()+f"\nEscape readout: {effects['escape_hz']:.1f} Hz"
                                f"\nMotor disruption: {effects['disruption']*100:.0f} %")
