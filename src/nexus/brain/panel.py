"""Native neural controls, independent of the 3D body's worker."""
from collections import deque
import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty, Full
import re
import sys

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSpinBox, QPlainTextEdit, QFileDialog,
    QGroupBox, QFormLayout, QCheckBox)

from .config import default_pack, model_config
from .worker import simulate_brain




class BrainPanel(QWidget):
    snapshot = Signal(object)
    session_event = Signal(str)

    def __init__(self, data_dir, command_sink=None, *, config=None):
        super().__init__()
        self.config = config or model_config(default_pack())
        self.command_sink = command_sink
        self.data_dir = data_dir
        self.process = None
        self.ready = False
        self.failed = False
        self.seq = 0
        self.telemetry = {}
        self.records = deque(maxlen=2000)
        self.intervention_seen = set()
        layout = QVBoxLayout(self)
        note = QLabel(self.config.title+"\n"+("Shared clock · experimental DNa02 steering" if command_sink else "Brain and body currently run independently."))
        note.setWordWrap(True)
        layout.addWidget(note)
        self.load_button = QPushButton("Load brain")
        self.load_button.clicked.connect(self.load)
        if command_sink:
            self.load_button.setText('Loading shared simulation…')
            self.load_button.setEnabled(False)
        layout.addWidget(self.load_button)
        self.status = QLabel("Brain not loaded")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.error = QLabel("")
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        self.controls = QWidget()
        controls = QVBoxLayout(self.controls)
        controls.setContentsMargins(0,0,0,0)
        row = QHBoxLayout()
        self.run_button = QPushButton("Run brain")
        self.run_button.setCheckable(True)
        self.run_button.clicked.connect(lambda value: self.send("running",value))
        step = QPushButton("Step 10 ms")
        step.clicked.connect(lambda: self.send("step"))
        row.addWidget(self.run_button)
        row.addWidget(step)
        controls.addLayout(row)
        self.bridge = None
        if command_sink:
            self.bridge = QCheckBox('Enable neural motor bridge')
            self.bridge.setChecked(True)
            self.bridge.toggled.connect(lambda value: self.send('bridge_enabled', value))
            controls.addWidget(self.bridge)
        selection = QGroupBox("Neural input")
        form = QFormLayout(selection)
        self.preset = QComboBox()
        self.preset.addItems([self.config.first_label, "MN9 readout cells (2)" if self.config.experimental else "MN9 readout cell", "Custom neuron IDs"])
        self.preset.addItems(['DNa02 left · steering', 'DNa02 right · steering', 'DNa02 both · steering'])
        if command_sink:
            self.preset.setCurrentIndex(3)
        self.preset.currentIndexChanged.connect(self.select_targets)
        form.addRow(self.preset)
        self.ids = QPlainTextEdit()
        self.ids.setPlaceholderText("Decimal neuron IDs for the selected dataset")
        self.ids.setMaximumHeight(65)
        self.ids.hide()
        form.addRow(self.ids)
        self.rate = QSpinBox()
        self.rate.setRange(1,1000)
        self.rate.setValue(200)
        self.rate.setSuffix(" Hz")
        form.addRow("Input rate",self.rate)
        stimulate = QPushButton("Apply stimulation")
        stimulate.clicked.connect(self.stimulate)
        form.addRow(stimulate)
        silence = QPushButton("Silence target outputs")
        silence.clicked.connect(self.silence)
        form.addRow(silence)
        controls.addWidget(selection)
        experiment = QGroupBox('Network perturbation · experimental')
        form = QFormLayout(experiment)
        explanation = QLabel('Reduce inhibitory connection strength.\nNeeds ongoing input to start activity.\nSeizure dynamics are not yet validated.')
        explanation.setWordWrap(True)
        form.addRow(explanation)
        self.inhibition = QSpinBox()
        self.inhibition.setRange(0, 100)
        self.inhibition.setValue(25)
        self.inhibition.setSuffix(' %')
        form.addRow('Inhibition remaining', self.inhibition)
        self.perturb_button = QPushButton('Apply reduced inhibition')
        self.perturb_button.clicked.connect(lambda: self.send('inhibition_gain', self.inhibition.value()/100))
        form.addRow(self.perturb_button)
        self.restore_button = QPushButton('Restore inhibition')
        self.restore_button.clicked.connect(lambda: self.send('inhibition_gain', 1.))
        form.addRow(self.restore_button)
        self.perturb_status = QLabel('Applied inhibition: 100 %')
        form.addRow(self.perturb_status)
        controls.addWidget(experiment)
        release = QPushButton("Release manual interventions" if command_sink else "Release all interventions")
        release.clicked.connect(lambda: self.send("release"))
        controls.addWidget(release)
        sequence = QGroupBox("Timed sequence")
        form = QFormLayout(sequence)
        self.sequence_mode = QComboBox()
        self.sequence_mode.addItems(['Selected neural input', 'Reduced inhibition + selected input'])
        form.addRow(self.sequence_mode)
        self.durations = []
        for name, value in (("Baseline",100),("Stimulation",500),("Recovery",500)):
            spin = QSpinBox()
            spin.setRange(1,60000)
            spin.setValue(value)
            spin.setSuffix(" ms")
            self.durations.append(spin)
            form.addRow(name,spin)
        run = QPushButton("Run sequence")
        run.clicked.connect(lambda: self.send("protocol",self.protocol()))
        form.addRow(run)
        row = QHBoxLayout()
        load = QPushButton("Load JSON…")
        load.clicked.connect(self.load_protocol)
        save = QPushButton("Save JSON…")
        save.clicked.connect(self.save_protocol)
        row.addWidget(load)
        row.addWidget(save)
        form.addRow(row)
        controls.addWidget(sequence)
        reset = QPushButton("Reset body + brain" if command_sink else "Reset brain")
        reset.clicked.connect(lambda: self.send("reset"))
        controls.addWidget(reset)
        self.controls.setEnabled(False)
        layout.addWidget(self.controls)
        self.metrics = QLabel("")
        self.metrics.setWordWrap(True)
        layout.addWidget(self.metrics)
        export = QPushButton("Export brain diagnostics…")
        export.clicked.connect(self.export)
        layout.addWidget(export)
        layout.addStretch()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(30)

    def select_targets(self):
        self.ids.setVisible(self.preset.currentIndex()==2)

    def targets(self):
        if self.preset.currentIndex()==0:
            return list(self.config.first_ids)
        if self.preset.currentIndex()==1:
            return list(self.config.mn9_ids)
        if self.preset.currentIndex() in (3, 4, 5):
            return list({3: self.config.left_ids, 4: self.config.right_ids,
                         5: self.config.left_ids+self.config.right_ids}[self.preset.currentIndex()])
        return [s for s in re.split(r"[\s,]+", self.ids.toPlainText().strip()) if s]

    def stimulate(self):
        self.send("stimulate",{"ids":self.targets(),"rate_hz":self.rate.value()})

    def silence(self):
        self.send("silence",{"ids":self.targets()})

    def protocol(self):
        pre, pulse, post = [spin.value() for spin in self.durations]
        description = {"format":"nexus-protocol-1","name":"Baseline / stimulation / recovery",
                "dataset": self.config.snapshot, "duration_ms":pre+pulse+post,"events":[
                    {"at_ms":0,"action":"release"},
                    {"at_ms":pre,"action":"stimulate","ids":self.targets(),"rate_hz":self.rate.value()},
                    {"at_ms":pre+pulse,"action":"release"}]}
        if self.sequence_mode.currentIndex() == 1:
            description['name'] = 'Reduced inhibition / selected input / recovery'
            description['events'].insert(1, {'at_ms': pre, 'action': 'inhibition_gain',
                                             'gain': self.inhibition.value()/100})
        return description

    def load(self):
        if self.command_sink is not None:
            return
        if self.process is not None:
            return
        directory = self.config.directory
        if not (directory / "manifest.json").is_file():
            self.fail(f"Prepared pack is missing: {directory}")
            return
        ctx = mp.get_context("spawn")
        self.commands,self.frames,self.events = ctx.Queue(maxsize=256),ctx.Queue(maxsize=2),ctx.Queue()
        self.process = ctx.Process(target=simulate_brain,args=(str(directory),self.commands,self.frames,self.events),
                                   kwargs={"allow_experimental": self.config.experimental}, name="Nexus brain")
        self.process.start()
        self.load_button.setEnabled(False)
        self.status.setText("Loading connectivity and preparing runtime…")
        self.session_event.emit('BRAIN · loading '+self.config.title)

    def send(self, kind, value=None):
        if not self.ready or self.failed:
            return
        if self.command_sink is not None:
            self.command_sink(kind, value)
            return
        self.seq += 1
        try:
            self.commands.put_nowait({"id":self.seq,"kind":kind,"value":value})
        except Full:
            self.status.setText("Command queue busy; try again.")

    def poll(self):
        if self.process is None:
            return
        while True:
            try:
                event = self.events.get_nowait()
            except Empty:
                break
            self.records.append(event)
            stamp = f"{event.get('sim_time', 0):.3f}s"
            self.session_event.emit(f"BRAIN {stamp} · {event.get('command', event['kind'])}" +
                                    (f" · {event['message']}" if 'message' in event else ''))
            if event["kind"]=="ready":
                self.ready = True
                self.controls.setEnabled(True)
            elif event["kind"]=="error":
                self.fail(event["message"])
            elif event["kind"]=="rejected":
                self.error.setText(event["message"])
            elif event["kind"]=="applied":
                self.error.setText("")
        packet = None
        while True:
            try:
                packet = self.frames.get_nowait()
            except Empty:
                break
        if packet is not None and not self.failed:
            self.receive_snapshot(packet)
        if not self.process.is_alive() and not self.failed:
            self.fail(f"Brain worker exited (code {self.process.exitcode})")

    def receive_snapshot(self, packet):
        if self.command_sink is not None and not self.failed:
            self.ready = True
            self.controls.setEnabled(True)
            self.load_button.setText("Brain connected")
        if self.failed:
            return
        self.telemetry = packet
        t = packet
        for intervention in t['interventions']:
            key = (t['generation'], json.dumps(intervention, sort_keys=True))
            if key not in self.intervention_seen:
                self.intervention_seen.add(key)
                detail = (f"inhibition remaining: {intervention['gain']*100:g} %"
                          if intervention['kind'] == 'inhibition_gain' else f"{intervention['kind']} intervention")
                if intervention['kind'] == 'circuit_input':
                    detail = f"{intervention['circuit']} input: {intervention['rate_hz']:g} Hz"
                elif intervention['kind'] == 'heat_scenario':
                    detail = f"heat scenario: {intervention['nominal_celsius']}°C nominal · {intervention['rate_hz']:g} Hz"
                self.session_event.emit(f"BRAIN {intervention['time']:.3f}s · {detail}")
        if len(self.intervention_seen) > 4000:
            self.intervention_seen = {(t['generation'], json.dumps(e, sort_keys=True)) for e in t['interventions']}
        active = len(t['stimulated_ids'])
        self.status.setText(f"{'Running' if t['running'] else 'Paused'} · {active} stimulated · {t['silenced_count']} silenced")
        self.perturb_status.setText(f"Applied inhibition: {t['inhibition_gain']*100:g} %")
        if t['protocol']:
            self.status.setText(self.status.text()+f"\nSequence {'complete' if t['protocol']['completed'] else 'active'}")
        if self.command_sink and t.get('environment'):
            food = t['environment']
            self.status.setText(self.status.text()+f"\nFood input: {'active' if food['active'] else 'off'} · {len(t['manual_ids'])} manual targets")
        self.run_button.blockSignals(True)
        self.run_button.setChecked(t['running'])
        self.run_button.setText(("Pause both" if t['running'] else "Run both") if self.command_sink else
                               ("Pause brain" if t['running'] else "Run brain"))
        self.run_button.blockSignals(False)
        mn9_label = "MN9 mean voltage" if len(self.config.mn9_ids) > 1 else "MN9 voltage"
        self.metrics.setText(f"Brain time: {t['sim_time']:.3f} s · {t['realtime_factor']:.2f}×\n"
            f"{t['neurons']:,} neurons · {t['edges']:,} edges\n"
            f"Spikes: {t['total_spikes']:,} · MN9: {t['mn9_spikes']}\n{mn9_label}: {t['mn9_mv']:.2f} mV\n"
            f"Population rate: {t['population_hz_per_neuron']:.3f} Hz/neuron")
        if self.bridge and 'motor_bridge' in t:
            m = t['motor_bridge']
            self.bridge.blockSignals(True)
            self.bridge.setChecked(m['enabled'])
            self.bridge.blockSignals(False)
            self.metrics.setText(self.metrics.text()+f"\nDNa02 output L/R: {m['left_hz']:.1f} / {m['right_hz']:.1f} Hz\n"
                                 f"Motor drive L/R: {m['left_drive']:.2f} / {m['right_drive']:.2f}")
        self.snapshot.emit(t)

    def fail(self, message):
        self.failed = True
        self.controls.setEnabled(False)
        self.status.setText("Brain stopped: "+message.splitlines()[-1])
        (self.data_dir / "brain-error.txt").write_text(message)

    def diagnostics(self):
        return {"version":"0.11.0","brain_drives_body":self.telemetry.get('brain_drives_body', False),
                "shared_clock":self.command_sink is not None,"telemetry":self.telemetry,"commands":list(self.records)}

    def export(self):
        path,_ = QFileDialog.getSaveFileName(self,"Brain diagnostics",str(self.data_dir / "brain.json"),"JSON (*.json)")
        if path:
            try:
                Path(path).write_text(json.dumps(self.diagnostics(),indent=2))
            except OSError as error:
                self.status.setText(str(error))

    def load_protocol(self):
        path,_ = QFileDialog.getOpenFileName(self,"Run saved sequence","","JSON (*.json)")
        if path:
            try:
                if Path(path).stat().st_size > 1000000:
                    raise ValueError("Sequence file is too large")
                self.send("protocol",json.loads(Path(path).read_text()))
            except (OSError,ValueError) as error:
                self.status.setText(str(error))

    def save_protocol(self):
        path,_ = QFileDialog.getSaveFileName(self,"Save sequence",str(self.data_dir / "sequence.json"),"JSON (*.json)")
        if path:
            try:
                Path(path).write_text(json.dumps(self.protocol(),indent=2))
            except OSError as error:
                self.status.setText(str(error))

    def shutdown(self):
        self.timer.stop()
        if self.process is None:
            return
        if self.process.is_alive():
            try:
                self.commands.put_nowait({"id":self.seq+1,"kind":"shutdown"})
            except Full:
                pass
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
        for channel in (self.commands,self.frames,self.events):
            channel.cancel_join_thread()
            channel.close()
