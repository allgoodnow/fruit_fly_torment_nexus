"""Native desktop front end. The first milestone uses the real body controller."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty, Full
import sys
import time


def main():
    mp.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["male-cns", "flywire-v630"], default="male-cns", help="Neural dataset; MaleCNS uses experimental LIF parameters")
    parser.add_argument("--male-cns-smoke-test", type=Path, help="Check MaleCNS controls, anatomy and shared-clock body")
    parser.add_argument("--smoke-test", type=Path, help="Run UI acceptance checks and write report/screenshot here")
    parser.add_argument("--brain-smoke-test", type=Path, help="Also exercise the real brain and timed sequence")
    parser.add_argument('--coupled-smoke-test', type=Path, help='Exercise shared-clock neural steering')
    parser.add_argument('--food-smoke-test', type=Path, help='Exercise contact-driven taste feedback')
    parser.add_argument('--perturbation-smoke-test', type=Path, help='Exercise reversible inhibition controls')
    parser.add_argument('--response-smoke-test', type=Path, help='Exercise threat, heat, and motor experiment controls')
    parser.add_argument('--behavior-smoke-test', type=Path, help='Exercise free ground behavior and neural interruption')
    parser.add_argument('--food-demo', action='store_true', help='Start a food crossing with no manual neural input')
    parser.add_argument('--independent', action='store_true', help='Use the original independent body and brain workers')
    parser.add_argument("--run-demo", action="store_true", help="Start both models with the selected sensory or steering input")
    args = parser.parse_args()
    if args.male_cns_smoke_test:
        args.dataset = 'male-cns'
        args.response_smoke_test = args.male_cns_smoke_test
    if args.dataset == 'male-cns' and (args.food_demo or args.food_smoke_test):
        parser.error('The food feedback demo currently requires --dataset flywire-v630')
    if args.brain_smoke_test:
        args.smoke_test = args.brain_smoke_test
    if args.coupled_smoke_test:
        args.smoke_test = args.coupled_smoke_test
    if args.food_smoke_test:
        args.smoke_test = args.food_smoke_test
    if args.perturbation_smoke_test:
        args.smoke_test = args.perturbation_smoke_test
    if args.response_smoke_test:
        args.smoke_test = args.response_smoke_test
    if args.behavior_smoke_test:
        args.smoke_test = args.behavior_smoke_test
    coupled = bool(args.food_smoke_test or args.food_demo or args.coupled_smoke_test or args.perturbation_smoke_test or args.response_smoke_test or args.behavior_smoke_test) or not (args.independent or args.smoke_test)

    from PySide6.QtCore import QStandardPaths, QTimer, Qt, Signal
    from PySide6.QtGui import QImage, QPainter, QPixmap, QPalette, QColor
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
        QLabel, QMainWindow, QPushButton, QSlider, QSplitter, QTextEdit,
        QVBoxLayout, QWidget, QTabWidget, QStackedWidget, QScrollArea,
    )
    import pyqtgraph as pg
    from nexus.worker import simulate
    from nexus.coupled import simulate_coupled
    from nexus.brain.panel import BrainPanel
    from nexus.brain.config import default_pack, model_config
    from nexus.brain.view import BrainView
    from nexus.food_panel import FoodPanel
    from nexus.intervention_panel import InterventionPanel

    try:
        config = model_config(default_pack(args.dataset))
    except (OSError, ValueError, KeyError) as error:
        parser.error(f'Cannot load {args.dataset}: {error}')
    app = QApplication(sys.argv[:1])
    app.setStyle('Fusion')
    palette = QPalette()
    for role, color in ((QPalette.ColorRole.Window, '#ffffff'), (QPalette.ColorRole.Base, '#ffffff'),
                        (QPalette.ColorRole.AlternateBase, '#f3f4f5'), (QPalette.ColorRole.Button, '#f5f5f5'),
                        (QPalette.ColorRole.WindowText, '#000000'), (QPalette.ColorRole.Text, '#000000'),
                        (QPalette.ColorRole.ButtonText, '#000000'), (QPalette.ColorRole.Highlight, '#d9dfe3'),
                        (QPalette.ColorRole.HighlightedText, '#000000'), (QPalette.ColorRole.ToolTipBase, '#ffffff'),
                        (QPalette.ColorRole.ToolTipText, '#000000')):
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    app.setApplicationName("Fruit Fly Torment Nexus")
    app.setOrganizationName("Nexus")
    data_dir = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation))
    if args.smoke_test:
        data_dir = args.smoke_test.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(data_dir / "matplotlib"))
    os.environ.setdefault("FLYGYM_ASSET_CACHE_DIR", str(data_dir / "assets"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(data_dir / "numba"))
    pg.setConfigOptions(antialias=False, background="#ffffff", foreground="#000000")

    class Viewport(QWidget):
        moved = Signal(float, float, float)

        def __init__(self):
            super().__init__()
            self.image = None
            self.last = None
            self.setMinimumSize(300, 280)
            self.setCursor(Qt.CursorShape.OpenHandCursor)

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.fillRect(self.rect(), Qt.GlobalColor.white)
            if self.image is not None:
                size = self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
                x, y = (self.width()-size.width())//2, (self.height()-size.height())//2
                painter.drawImage(x, y, self.image.scaled(size, Qt.AspectRatioMode.KeepAspectRatio,
                                                         Qt.TransformationMode.SmoothTransformation))
            else:
                painter.setPen(Qt.GlobalColor.black)
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Loading NeuroMechFly…")

        def mousePressEvent(self, event):
            if event.button() == Qt.MouseButton.LeftButton:
                self.last = event.position()

        def mouseMoveEvent(self, event):
            if self.last is not None:
                delta = event.position()-self.last
                self.last = event.position()
                self.moved.emit(-delta.x()*0.4, -delta.y()*0.3, 0)

        def mouseReleaseEvent(self, event):
            self.last = None

        def wheelEvent(self, event):
            self.moved.emit(0, 0, -event.angleDelta().y()/120*0.12)

    class Window(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Fruit Fly Torment Nexus — 3D fly and neural lab")
            self.resize(1500, 920)
            self.setMinimumSize(1080, 720)
            self.ctx = mp.get_context("spawn")
            self.commands = self.ctx.Queue(maxsize=256)
            self.frames = self.ctx.Queue(maxsize=2)
            self.events = self.ctx.Queue()
            target = simulate_coupled if coupled else simulate
            worker_args = (str(config.directory), self.commands, self.frames, self.events) if coupled else (self.commands, self.frames, self.events)
            self.process = self.ctx.Process(target=target, args=worker_args,
                                            kwargs={'autonomous': not args.smoke_test or bool(args.behavior_smoke_test), 'allow_experimental': config.experimental} if coupled else {},
                                            name="Nexus simulation")
            self.seq = 0
            self.records = deque(maxlen=2000)
            self.started_at = datetime.now(timezone.utc).isoformat()
            self.telemetry = {}
            self.ready = False
            self.failed = False
            self.closed = False
            self.camera_delta = [0.0, 0.0, 0.0]
            self.frame_count = 0
            self.world_log_seen = set()
            self.smoke_stage = 0
            self.smoke_clock = time.monotonic()
            self.smoke_checks = []
            self.plot_time = deque(maxlen=500)
            self.plot_values = [deque(maxlen=500) for _ in range(6)]
            root = QWidget()
            outer = QVBoxLayout(root)
            title = QLabel("FRUIT FLY TORMENT NEXUS")
            title.setStyleSheet("font-size: 22px; font-weight: 600; padding: 8px 0")
            outer.addWidget(title)
            badge = QLabel('RESEARCH PROTOTYPE  ·  Neural steering with a shared clock  ·  Experimental decoder + engineered gait' if coupled else
                           "RESEARCH PROTOTYPE  ·  3D fly + whole-brain neural lab  ·  Body and brain run independently")
            badge.setText(config.title+"  ·  "+badge.text())
            outer.addWidget(badge)
            split = QSplitter(Qt.Orientation.Horizontal)
            outer.addWidget(split, 1)
            scene_panel = QWidget()
            scene_layout = QVBoxLayout(scene_panel)
            scene_layout.setContentsMargins(0, 4, 8, 0)
            self.viewport = Viewport()
            self.viewport.moved.connect(self.camera_move)
            scenes = QSplitter(Qt.Orientation.Horizontal)
            body_scene = QWidget()
            body_layout = QVBoxLayout(body_scene)
            body_layout.setContentsMargins(0, 0, 0, 0)
            body_label = QLabel('BODY / NEUROMECHFLY')
            body_label.setMinimumHeight(35)
            body_layout.addWidget(body_label)
            body_layout.addWidget(self.viewport, 1)
            self.food_status = QLabel('Food feedback loading…' if coupled else 'Independent body mode')
            self.food_status.setWordWrap(True)
            body_layout.addWidget(self.food_status)
            self.behavior_status = QLabel('Ground behavior loading…' if coupled else '')
            self.behavior_status.setWordWrap(True)
            body_layout.addWidget(self.behavior_status)
            body_layout.addWidget(QLabel('Drag to orbit · Scroll to zoom\nCamera follows the fly'))
            scenes.addWidget(body_scene)
            self.brain_view = BrainView(config.directory)
            scenes.addWidget(self.brain_view)
            scenes.setSizes([520, 520])
            scene_layout.addWidget(scenes, 1)
            self.graph = pg.PlotWidget(title="Leg oscillator magnitude — controller state, not neural spikes")
            self.graph.setMaximumHeight(180)
            self.graph.setLabel("bottom", "Simulated time", units="s")
            self.curves = [self.graph.plot(pen=pg.mkPen(c, width=1.5)) for c in
                           ("#1b1b1b", "#4c6378", "#745e46", "#256653", "#63517f", "#7a525a")]
            self.graph_stack = QStackedWidget()
            self.graph_stack.setMaximumHeight(180)
            self.graph_stack.addWidget(self.graph)
            self.brain_graph = pg.PlotWidget(title="Recent neural spikes · rows are model indices")
            self.brain_graph.setLabel("bottom", "Brain time", units="s")
            self.brain_graph.setLabel("left", "Neuron index")
            self.brain_points = self.brain_graph.plot(pen=None,symbol='o',symbolSize=2,symbolBrush='#b91c1c',symbolPen=None)
            self.graph_stack.addWidget(self.brain_graph)
            scene_layout.addWidget(self.graph_stack)
            split.addWidget(scene_panel)
            side = QWidget()
            side.setMinimumWidth(280)
            side.setMaximumWidth(360)
            controls = QVBoxLayout(side)
            self.status = QLabel("Loading model…")
            self.status.setWordWrap(True)
            controls.addWidget(self.status)
            buttons = QHBoxLayout()
            self.run_button = QPushButton("Run")
            self.run_button.setCheckable(True)
            self.run_button.clicked.connect(lambda checked: self.send("running", checked))
            self.step_button = QPushButton("Step 10 ms")
            self.step_button.clicked.connect(lambda: self.send("step"))
            buttons.addWidget(self.run_button)
            buttons.addWidget(self.step_button)
            controls.addLayout(buttons)
            self.reset_button = QPushButton("Reset body + brain" if coupled else "Reset simulation")
            self.reset_button.clicked.connect(lambda: self.send("reset"))
            controls.addWidget(self.reset_button)
            motor = QGroupBox("Motor controller")
            form = QFormLayout(motor)
            self.wander = QCheckBox("Free ground behavior" if coupled else "Automatic steering")
            self.wander.setChecked(True)
            self.wander.toggled.connect(lambda checked: self.send('autonomous' if coupled else 'wander', checked))
            form.addRow(self.wander)
            if coupled:
                note = QLabel('Explore, turn, and rest. Authored behavior;\nneural responses take priority.')
                note.setWordWrap(True)
                form.addRow(note)
            self.drive = QSlider(Qt.Orientation.Horizontal)
            self.drive.setRange(0, 130)
            self.drive.setValue(100)
            self.drive.valueChanged.connect(lambda value: self.send("drive", value/100))
            form.addRow("Baseline walking drive" if coupled else "Stride drive", self.drive)
            self.turn = QSlider(Qt.Orientation.Horizontal)
            self.turn.setRange(-60, 60)
            self.turn.valueChanged.connect(lambda value: self.send("turn", value/100))
            form.addRow("L / R bias", self.turn)
            if coupled:
                self.turn.setEnabled(False)
            self.release = QPushButton("Restore baseline walking drive" if coupled else "Restore default motor input")
            self.release.clicked.connect(lambda: self.send("release"))
            form.addRow(self.release)
            controls.addWidget(motor)
            self.food_panel = FoodPanel(self.send, available=config.food_available) if coupled else None
            if self.food_panel:
                controls.addWidget(self.food_panel)
            camera_button = QPushButton("Reset camera")
            camera_button.clicked.connect(lambda: self.send("camera_reset"))
            controls.addWidget(camera_button)
            self.metrics = QLabel("Waiting for physics state")
            self.metrics.setWordWrap(True)
            controls.addWidget(self.metrics)
            brain = QGroupBox("Brain integration")
            brain_layout = QVBoxLayout(brain)
            pending = QLabel('DNa02 neural activity changes left/right walking drive. Open the Brain tab to stimulate either side.\n\n'
                             'Baseline walking and leg rhythms are engineered. Contact with food drives the reference taste cells through a pooled proxy.\n\n'
                             'Run, Pause, Step and Reset affect both models.' if coupled else
                             "Open the Brain tab for neural controls.\n\nThe walking body still uses its engineered controller. Neural output is not mapped to movement yet.")
            if coupled and not config.food_available:
                pending.setText('DNa02 activity changes left/right walking drive. Baseline walking and leg rhythms are engineered.\n\nMaleCNS includes the VNC, but its neurons are not yet mapped directly to leg muscles. Food feedback awaits a sugar-cell mapping.\n\nRun, Pause, Step and Reset affect both models.')
            pending.setWordWrap(True)
            brain_layout.addWidget(pending)
            controls.addWidget(brain)
            scene_layout.addWidget(QLabel("SESSION EVENTS / BODY + BRAIN"))
            self.log = QTextEdit()
            self.log.setObjectName('eventLog')
            self.log.setReadOnly(True)
            self.log.setAcceptRichText(False)
            self.log.setMinimumHeight(80)
            self.log.setMaximumHeight(115)
            self.log.document().setMaximumBlockCount(200)
            scene_layout.addWidget(self.log)
            controls.addStretch()
            export = QPushButton("Export session diagnostics…")
            export.clicked.connect(self.export)
            controls.addWidget(export)
            self.tabs = QTabWidget()
            self.tabs.setMinimumWidth(340)
            self.tabs.setMaximumWidth(400)
            body_scroll = QScrollArea()
            body_scroll.setWidgetResizable(True)
            body_scroll.setWidget(side)
            self.tabs.addTab(body_scroll,"Body")
            sink = (lambda kind, value: self.send('brain', {'kind': kind, 'value': value})) if coupled else None
            self.brain_panel = BrainPanel(data_dir, command_sink=sink, config=config)
            self.brain_panel.snapshot.connect(self.brain_snapshot)
            self.brain_panel.session_event.connect(self.log.append)
            brain_scroll = QScrollArea()
            brain_scroll.setWidgetResizable(True)
            brain_scroll.setWidget(self.brain_panel)
            self.tabs.addTab(brain_scroll,"Brain")
            self.intervention_panel = InterventionPanel(self.brain_panel, coupled=coupled)
            experiment_scroll = QScrollArea()
            experiment_scroll.setWidgetResizable(True)
            experiment_scroll.setWidget(self.intervention_panel)
            self.tabs.addTab(experiment_scroll, 'Experiments')
            self.tabs.currentChanged.connect(lambda index: self.graph_stack.setCurrentIndex(0 if index == 0 else 1))
            if coupled:
                self.tabs.setCurrentIndex(1)
            split.addWidget(self.tabs)
            split.setSizes([1100, 360])
            self.setCentralWidget(root)
            self.interactive = [self.run_button, self.step_button, self.reset_button, motor, camera_button]
            if self.food_panel:
                self.interactive.append(self.food_panel)
            for widget in self.interactive:
                widget.setEnabled(False)
            self.process.start()
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.poll)
            self.timer.start(30)
            if (args.run_demo or args.food_demo) and not args.smoke_test:
                self.tabs.setCurrentIndex(1)
                self.brain_panel.load()
                self.demo_body_started = self.demo_brain_started = False
                self.demo_timer = QTimer(self)
                self.demo_timer.timeout.connect(self.start_demo)
                self.demo_timer.start(100)
            if args.smoke_test:
                self.smoke_timer = QTimer(self)
                self.smoke_timer.timeout.connect(self.smoke)
                self.smoke_timer.start(100)

        def start_demo(self):
            if self.failed or self.brain_panel.failed:
                self.demo_timer.stop()
                print('Demo startup failed; see the application diagnostics.', flush=True)
                return
            if args.food_demo:
                if self.ready and self.brain_panel.ready and not self.demo_body_started:
                    self.send('food_demo')
                    self.demo_body_started = True
                brain = self.brain_panel.telemetry
                if brain.get('mn9_spikes', 0) > 0:
                    self.demo_timer.stop()
                    print(json.dumps({'food_demo_running': True, 'body_time': self.telemetry['sim_time'],
                                      'brain_time': brain['sim_time'], 'mn9_spikes': brain['mn9_spikes'],
                                      'manual_ids': brain['manual_ids'], 'food_active': brain['environment']['active']}), flush=True)
                return
            if self.ready and not self.demo_body_started:
                self.send('running', True)
                self.demo_body_started = True
            if self.brain_panel.ready and not self.demo_brain_started:
                self.brain_panel.stimulate()
                self.brain_panel.send('running', True)
                self.demo_brain_started = True
            brain = self.brain_panel.telemetry
            if self.telemetry.get('sim_time', 0) > 0 and brain.get('total_spikes', 0) > 0:
                self.demo_timer.stop()
                print(json.dumps({'demo_running': True, 'body_time': self.telemetry['sim_time'],
                                  'brain_time': brain['sim_time'], 'brain_spikes': brain['total_spikes'],
                                  'input': self.brain_panel.preset.currentText(), 'brain_drives_body': coupled}), flush=True)

        def camera_move(self, dx, dy, zoom):
            self.camera_delta = [a+b for a, b in zip(self.camera_delta, (dx, dy, zoom))]

        def brain_snapshot(self,t):
            self.brain_view.update_snapshot(t)
            self.brain_points.setData([s*.0001 for s in t['raster_steps']], t['raster_indices'])
            self.brain_graph.setXRange(max(0,t['sim_time']-.5),max(.1,t['sim_time']),padding=0)

        def send(self, kind, value=None):
            if not self.ready or self.failed:
                return
            self.seq += 1
            try:
                self.commands.put_nowait({"id": self.seq, "kind": kind, "value": value})
            except Full:
                self.status.setText("Command queue busy; please try again.")

        def poll(self):
            if any(self.camera_delta) and self.ready:
                self.send("camera", dict(zip(("dx", "dy", "zoom"), self.camera_delta)))
                self.camera_delta = [0.0, 0.0, 0.0]
            while True:
                try:
                    event = self.events.get_nowait()
                except Empty:
                    break
                self.records.append(event)
                if coupled:
                    self.brain_panel.records.append(event)
                    if event['kind'] == 'rejected':
                        self.brain_panel.error.setText(event['message'])
                    elif event['kind'] == 'applied':
                        self.brain_panel.error.setText('')
                if event["kind"] == "ready":
                    self.ready = True
                    for widget in self.interactive:
                        widget.setEnabled(True)
                    self.log.append('Body + brain loaded. Shared clock ready.' if coupled else "Body loaded. Ready to run.")
                    if coupled and (not args.smoke_test or args.behavior_smoke_test) and not (args.run_demo or args.food_demo):
                        self.send('running', True)
                elif event["kind"] == "error":
                    self.fail(event["message"])
                elif event['kind'] == 'protocol_complete':
                    self.log.append(f"BODY + BRAIN {event['sim_time']:.4f}s · sequence complete")
                elif event["kind"] == "applied":
                    if event["command"] not in ("camera", "drive", "turn"):
                        self.log.append(f"BODY {event['sim_time']:.3f}s · {event['command']}")
                elif event["kind"] == "rejected":
                    self.log.append(event["message"])
            packet = None
            while True:
                try:
                    packet = self.frames.get_nowait()
                except Empty:
                    break
            if packet is not None and not self.failed:
                if coupled:
                    self.brain_panel.receive_snapshot(packet['brain'])
                    food = packet['telemetry'].get('environment')
                    if food:
                        self.food_panel.receive(food)
                        self.food_status.setText(f"Food: {'present' if food['present'] else 'removed'} · Foot contact: {'yes' if food['contact'] else 'no'}\n"
                                                 f"Taste input: {'active' if food['active'] else 'off'} · Exposure: {food['active_seconds']:.3f} s")
                        for event in food['events']:
                            key = (packet['telemetry']['generation'], json.dumps(event, sort_keys=True))
                            if key not in self.world_log_seen:
                                self.world_log_seen.add(key)
                                detail = f" · contact={event['contact']} · input={event['active']}" if event['kind'] == 'food_contact' else ''
                                self.log.append(f"ENV {event['time']:.4f}s · {event['kind']}{detail}")
                        if len(self.world_log_seen) > 1000:
                            self.world_log_seen = {(packet['telemetry']['generation'], json.dumps(e, sort_keys=True)) for e in food['events']}
                previous = self.telemetry
                self.telemetry = t = packet["telemetry"]
                image = packet["pixels"]
                if image is not None:
                    h, w, _ = image.shape
                    self.viewport.image = QImage(image.data, w, h, 3*w, QImage.Format.Format_RGB888).copy()
                    self.viewport.update()
                    self.frame_count += 1
                self.status.setText("Running" if t["running"] else "Paused — model state preserved")
                behavior = t.get('ground_behavior')
                if behavior:
                    self.behavior_status.setText(f"Ground behavior: {behavior['state']} · authored controller")
                    old_behavior = previous.get('ground_behavior', {})
                    if behavior['state'] != old_behavior.get('state') or t['generation'] != previous.get('generation'):
                        self.log.append(f"BODY {t['sim_time']:.3f}s · {behavior['state']}")
                for widget, value in ((self.run_button, t["running"]), (self.wander, behavior['enabled'] if behavior else t["wander"])):
                    widget.blockSignals(True)
                    widget.setChecked(value)
                    widget.blockSignals(False)
                self.run_button.setText("Pause" if t["running"] else "Run")
                for slider, value in ((self.drive, t["drive"]), (self.turn, t["turn"])):
                    if not slider.isSliderDown():
                        slider.blockSignals(True)
                        slider.setValue(round(value*100))
                        slider.blockSignals(False)
                self.turn.setEnabled(not coupled and not t["wander"])
                x, y, z = t["position_mm"]
                self.metrics.setText(f"Simulation: {t['sim_time']:.3f} s\nRate: {t['realtime_factor']:.2f}× real time\n"
                                     f"Position: {x:.2f}, {y:.2f}, {z:.2f} mm\n"
                                     f"Displacement: {t['displacement_mm']:.2f} mm\n"
                                     f"{t['joints']} joints · {t['actuators']} actuators · {t['contacts']} contacts")
                if previous.get("generation") != t["generation"]:
                    self.plot_time.clear()
                    for values in self.plot_values:
                        values.clear()
                if not self.plot_time or t["sim_time"] != self.plot_time[-1]:
                    self.plot_time.append(t["sim_time"])
                    for values, value in zip(self.plot_values, t["magnitudes"]):
                        values.append(value)
                    for curve, values in zip(self.curves, self.plot_values):
                        curve.setData(list(self.plot_time), list(values))
            if not self.process.is_alive() and not self.failed and not self.closed:
                self.fail(f"Simulation worker exited (code {self.process.exitcode}).")

        def fail(self, message):
            self.failed = True
            self.status.setText("Simulation stopped — see diagnostic details")
            self.log.append(message)
            (data_dir / "last-error.txt").write_text(message)
            for widget in self.interactive:
                widget.setEnabled(False)
            if coupled:
                self.brain_panel.fail(message)

        def diagnostics(self):
            return {"version": "0.9.0", "started_at": self.started_at,
                    "model": "NeuroMechFly 2.1.0 / engineered hybrid locomotion",
                    "brain_connected": coupled, "sensory_feedback_connected": coupled, "telemetry": self.telemetry,
                    "events": list(self.records), "rendered_frames": self.frame_count,
                    "brain_lab":self.brain_panel.diagnostics(), "brain_view":self.brain_view.diagnostics()}

        def export(self):
            path, _ = QFileDialog.getSaveFileName(self, "Export diagnostics", str(data_dir / "session.json"), "JSON (*.json)")
            if path:
                try:
                    Path(path).write_text(json.dumps(self.diagnostics(), indent=2))
                    self.log.append(f"Exported {path}")
                except OSError as error:
                    self.log.append(f"Export failed: {error}")

        def smoke(self):
            """Exercise our app's public command interface and inspect rendered state."""
            if self.failed or self.brain_panel.failed or time.monotonic()-self.smoke_clock > (360 if config.experimental else 120):
                self.smoke_finish(False)
                return
            if args.coupled_smoke_test:
                self.coupled_smoke()
                return
            if args.food_smoke_test:
                self.food_smoke()
                return
            if args.perturbation_smoke_test:
                self.perturbation_smoke()
                return
            if args.response_smoke_test:
                self.response_smoke()
                return
            if args.behavior_smoke_test:
                self.behavior_smoke()
                return
            if self.smoke_stage >= 7:
                self.brain_smoke()
                return
            t = self.telemetry
            if not t:
                return
            if self.smoke_stage == 0:
                self.send("running", True)
                self.smoke_stage = 1
            elif self.smoke_stage == 1 and t["sim_time"] >= 0.08:
                self.smoke_checks.append("physics advances and renders")
                self.send("running", False)
                self.smoke_stage = 2
            elif self.smoke_stage == 2 and not t["running"]:
                self.pause_time = t["sim_time"]
                self.pause_wall = time.monotonic()
                self.smoke_stage = 3
            elif self.smoke_stage == 3 and time.monotonic()-self.pause_wall > 0.5:
                if abs(t["sim_time"]-self.pause_time) > 1e-9:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.append("pause preserves simulated time")
                self.send("step")
                self.smoke_stage = 4
            elif self.smoke_stage == 4 and t["sim_time"] > self.pause_time:
                if abs(t["sim_time"]-self.pause_time-0.01) > 1e-8:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.append("single-step advances exactly 10ms")
                self.send("camera", {"dx": 40, "dy": -10, "zoom": -0.1})
                self.camera_frames = self.frame_count
                self.smoke_stage = 5
            elif self.smoke_stage == 5 and self.frame_count > self.camera_frames:
                self.smoke_checks.append("camera redraws while paused")
                self.grab().save(str(data_dir / "app.png"))
                self.send("reset")
                self.smoke_stage = 6
            elif self.smoke_stage == 6 and t["sim_time"] == 0 and t["generation"] == 1:
                self.smoke_checks.append("reset reinitializes state")
                if args.brain_smoke_test:
                    self.tabs.setCurrentIndex(1)
                    self.brain_panel.load()
                    self.smoke_stage = 7
                else:
                    self.smoke_finish(True)

        def brain_smoke(self):
            panel = self.brain_panel
            t = panel.telemetry
            if self.smoke_stage == 7 and panel.ready:
                panel.send("protocol",panel.protocol())
                self.smoke_stage = 8
            elif self.smoke_stage == 8 and t.get('protocol',{}):
                if not t['protocol']['completed']:
                    return
                events = t['interventions']
                if t['sim_time'] != 1.1 or t['total_spikes'] == 0 or t['mn9_spikes'] == 0 or t['stimulated_ids']:
                    self.smoke_finish(False)
                    return
                if [(e['kind'],e['time']) for e in events] != [('release',0.0),('stimulate',.1),('release',.6)]:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.extend(["full brain loads without C++ compiler","sequence applies at exact neural times","real neural spikes and MN9 readout recorded"])
                panel.stimulate()
                self.smoke_stage = 9
            elif self.smoke_stage == 9 and t.get('stimulated_ids'):
                panel.send('release')
                self.smoke_stage = 10
            elif self.smoke_stage == 10 and not t.get('stimulated_ids'):
                if t['sim_time'] != 1.1 or t['running']:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.append("manual stimulus and release work while paused")
                self.grab().save(str(data_dir / "brain-app.png"))
                (data_dir / "brain-before-reset.json").write_text(json.dumps(panel.diagnostics(),indent=2))
                panel.send('reset')
                self.smoke_stage = 11
            elif self.smoke_stage == 11 and t.get('generation') == 1:
                if t['sim_time']!=0 or t['total_spikes']!=0 or self.brain_view.active_count:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.append("brain reset clears time and activity")
                self.send('running',True)
                panel.stimulate()
                panel.send('running',True)
                self.smoke_stage = 12
            elif self.smoke_stage == 12 and t.get('sim_time',0) >= .1 and self.telemetry.get('sim_time',0) >= .04:
                self.send('running',False)
                panel.send('running',False)
                self.smoke_stage = 13
            elif self.smoke_stage == 13 and not t.get('running') and not self.telemetry.get('running'):
                if not self.brain_view.active_count or not self.brain_view.gl.isValid():
                    self.smoke_finish(False)
                    return
                self.smoke_checks.append("3D body and brain advance concurrently and pause")
                self.smoke_checks.append("anatomical OpenGL brain shows real recent spikes and clears on reset")
                self.brain_view.gl.grabFramebuffer().save(str(data_dir / 'brain-3d.png'))
                self.grab().save(str(data_dir / "brain-app-concurrent.png"))
                self.brain_pause_tick = self.brain_view.tick
                self.brain_pause_count = self.brain_view.active_count
                self.pause_wall = time.monotonic()
                self.brain_view.gl.orbit(15, 5)
                self.smoke_stage = 14
            elif self.smoke_stage == 14 and time.monotonic()-self.pause_wall > .5:
                success = self.brain_view.tick == self.brain_pause_tick and self.brain_view.active_count == self.brain_pause_count
                if success:
                    self.smoke_checks.append('brain camera orbits while paused without advancing activity')
                self.smoke_finish(success)

        def coupled_smoke(self):
            t = self.brain_panel.telemetry
            if not t or not self.telemetry:
                return
            if abs(t['sim_time']-self.telemetry['sim_time']) > 1e-9:
                self.smoke_finish(False)
                return
            if self.smoke_stage == 0:
                self.smoke_checks.append('coupled brain/body load and anatomical OpenGL initializes')
                self.send('food_config', {'enabled': False})
                self.coupled_saw_turn = False
                sequence = {'format': 'nexus-protocol-1', 'duration_ms': 320.1,
                            'events': [{'at_ms': 0, 'action': 'release'},
                                       {'at_ms': 20.1, 'action': 'stimulate', 'ids': self.brain_panel.targets(), 'rate_hz': 200},
                                       {'at_ms': 170.1, 'action': 'release'}]}
                self.brain_panel.send('protocol', sequence)
                self.smoke_stage = 1
            elif self.smoke_stage == 1:
                m = t['motor_bridge']
                if t['stimulated_ids'] and t['sim_time'] >= .1 and m['left_drive'] < m['right_drive'] and self.brain_view.active_count:
                    if not self.coupled_saw_turn:
                        self.grab().save(str(data_dir / 'coupled-app.png'))
                        self.brain_view.gl.grabFramebuffer().save(str(data_dir / 'brain-3d.png'))
                    self.coupled_saw_turn = True
                if not t.get('protocol') or not t['protocol']['completed']:
                    return
                expected = [('release', 0.), ('stimulate', .0201), ('release', .1701)]
                if not self.coupled_saw_turn or t['sim_time'] != .3201 or t['running'] or t['stimulated_ids'] or not self.brain_view.gl.isValid():
                    self.smoke_finish(False)
                    return
                if [(e['kind'], e['time']) for e in t['interventions']] != expected:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.extend(['actual DNa02 spikes change left/right controller drive',
                                          '3D brain displays measured steering-neuron activity',
                                          'off-grid sequence events occur at exact neural times',
                                          'both clocks stop at the exact sequence end'])
                self.pause_wall = time.monotonic()
                self.camera_move(15, 0, 0)
                self.smoke_stage = 2
            elif self.smoke_stage == 2 and time.monotonic()-self.pause_wall > .5:
                if t['sim_time'] != .3201:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.append('camera movement leaves both paused clocks unchanged')
                self.brain_panel.send('step')
                self.smoke_stage = 3
            elif self.smoke_stage == 3 and t['sim_time'] > .3201:
                if t['sim_time'] != .3301:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.append('Brain single-step advances both models exactly 10ms')
                self.brain_panel.send('reset')
                self.smoke_stage = 4
            elif self.smoke_stage == 4 and t['generation'] == 1:
                if t['sim_time'] or t['total_spikes'] or self.brain_view.active_count:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.append('Brain reset clears both models and activity')
                self.brain_panel.send('bridge_enabled', False)
                self.brain_panel.stimulate()
                self.send('running', True)
                self.smoke_stage = 5
            elif self.smoke_stage == 5 and t['sim_time'] >= .15:
                m = t['motor_bridge']
                if not t['total_spikes'] or m['enabled'] or m['left_drive'] != m['right_drive']:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.append('decoder ablation preserves neural activity and removes motor bias')
                self.send('running', False)
                self.smoke_stage = 6
            elif self.smoke_stage == 6 and not t['running']:
                self.smoke_checks.append('Body pause also pauses the brain')
                self.smoke_finish(True)

        def food_smoke(self):
            t = self.brain_panel.telemetry
            if not t or not self.telemetry:
                return
            food = t['environment']
            if abs(t['sim_time']-self.telemetry['sim_time']) > 1e-9:
                self.smoke_finish(False)
                return
            if self.smoke_stage == 0:
                self.send('food_demo')
                self.smoke_stage = 1
            elif self.smoke_stage == 1 and food['active'] and t['mn9_spikes'] > 0:
                if t['manual_ids'] or len(t['sensory_ids']) != 21 or not self.brain_view.active_count:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.extend(['visible food patch and actual foot contact drive 21 taste inputs',
                                          'food input produces real downstream MN9 spikes without manual stimulation',
                                          'anatomical brain displays food-driven activity'])
                self.grab().save(str(data_dir / 'food-app.png'))
                self.send('running', False)
                self.smoke_stage = 2
            elif self.smoke_stage == 2 and not t['running']:
                self.food_pause_time = t['sim_time']
                self.food_pause_spikes = t['total_spikes']
                self.send('food_config', {'present': False})
                self.smoke_stage = 3
            elif self.smoke_stage == 3 and not food['present']:
                if t['sim_time'] != self.food_pause_time or t['sensory_ids'] or food['active'] or t['total_spikes'] != self.food_pause_spikes:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.append('removing food while paused clears sensory input without advancing or resetting')
                self.send('food_place', 'under')
                self.smoke_stage = 4
            elif self.smoke_stage == 4 and food['active']:
                self.smoke_checks.append('placing food under actual feet restores pending sensory input while paused')
                self.brain_panel.stimulate()
                self.smoke_stage = 5
            elif self.smoke_stage == 5 and t['manual_ids']:
                if len(t['sensory_ids']) != 21 or len(t['stimulated_ids']) != 22:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.append('manual steering and food input coexist in separate channels')
                self.brain_panel.send('release')
                self.smoke_stage = 6
            elif self.smoke_stage == 6 and not t['manual_ids']:
                if len(t['sensory_ids']) != 21:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.append('manual release preserves ongoing environmental input')
                self.send('food_config', {'enabled': False})
                self.smoke_stage = 7
            elif self.smoke_stage == 7 and not food['enabled']:
                if not food['present'] or not food['contact'] or food['active'] or t['sensory_ids']:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.append('feedback ablation preserves physical food contact and removes neural input')
                self.send('food_place', 'ahead')
                self.send('reset')
                self.smoke_stage = 8
            elif self.smoke_stage == 8 and t['sim_time'] == 0:
                if t['total_spikes'] or food['active_seconds'] or self.brain_view.active_count:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.append('shared reset clears neural activity, clocks, and exposure counters')
                self.tabs.setCurrentIndex(0)
                self.grab().save(str(data_dir / 'food-controls.png'))
                self.smoke_finish(True)

        def perturbation_smoke(self):
            panel, t = self.brain_panel, self.brain_panel.telemetry
            if not panel.ready or not t or not self.frame_count or self.brain_view.anatomy is None:
                return
            if abs(t['sim_time']-self.telemetry['sim_time']) > 1e-9:
                self.smoke_finish(False)
                return
            if self.smoke_stage == 0:
                self.send('food_config', {'enabled': False})
                panel.inhibition.setValue(25)
                panel.perturb_button.click()
                self.smoke_stage = 1
            elif self.smoke_stage == 1 and t['inhibition_gain'] == .25:
                if t['sim_time'] != 0 or t['running']:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.append('native inhibition button applies while paused without advancing time')
                panel.preset.setCurrentIndex(0)
                panel.sequence_mode.setCurrentIndex(1)
                for spin, value in zip(panel.durations, [100, 300, 150]):
                    spin.setValue(value)
                panel.send('protocol', panel.protocol())
                self.smoke_stage = 2
            elif self.smoke_stage == 2 and (t.get('protocol') or {}).get('completed'):
                times = [(e['kind'], e['time']) for e in t['interventions']]
                success = (t['sim_time'] == .55 and not t['running'] and t['inhibition_gain'] == 1
                           and not t['manual_ids'] and t['total_spikes'] > 0
                           and ('inhibition_gain', .1) in times and ('release', .4) in times)
                if not success:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.extend(['prepared inhibition/input/recovery sequence uses exact shared-clock boundaries',
                                          'full network generates spikes and release restores baseline gain',
                                          '3D body advances under the existing neural decoder'])
                self.pause_time, self.pause_spikes = t['sim_time'], t['total_spikes']
                panel.stimulate()
                panel.perturb_button.click()
                self.smoke_stage = 3
            elif self.smoke_stage == 3 and t['inhibition_gain'] == .25 and t['manual_ids']:
                panel.restore_button.click()
                self.smoke_stage = 4
            elif self.smoke_stage == 4 and t['inhibition_gain'] == 1:
                if not t['manual_ids'] or t['sim_time'] != self.pause_time or t['total_spikes'] != self.pause_spikes:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.append('restore inhibition retains manual input, time, and spike totals')
                panel.perturb_button.click()
                panel.send('release')
                self.smoke_stage = 5
            elif self.smoke_stage == 5 and t['inhibition_gain'] == 1 and not t['manual_ids']:
                if t['sim_time'] != self.pause_time or t['total_spikes'] != self.pause_spikes:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.append('manual release removes the overlay without resetting the session')
                self.grab().save(str(data_dir / 'perturbation-app.png'))
                panel.grab().save(str(data_dir / 'perturbation-controls.png'))
                panel.send('inhibition_gain', 0.)
                panel.send('reset')
                self.smoke_stage = 6
            elif self.smoke_stage == 6 and t['generation'] == 1:
                success = t['sim_time'] == 0 and t['total_spikes'] == 0 and t['inhibition_gain'] == 1
                if success:
                    self.smoke_checks.append('reset clears both clocks, spike counts, and the overlay')
                self.smoke_finish(success)

        def behavior_smoke(self):
            t = self.brain_panel.telemetry
            if not self.brain_panel.ready or not t or not self.frame_count:
                return
            b = t['ground_behavior']
            if abs(t['sim_time']-self.telemetry['sim_time']) > 1e-9:
                self.smoke_finish(False)
                return
            if self.smoke_stage == 0:
                self.tabs.setCurrentIndex(0)
                self.send('food_config', {'enabled': False})
                if not t['running']:
                    return
                if not b['enabled']:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.append('normal startup begins free ground behavior')
                self.behavior_seen = set()
                self.smoke_stage = 1
            elif self.smoke_stage == 1:
                self.behavior_seen.add(b['state'])
                if b['state'] == 'resting':
                    if not {'exploring', 'turning', 'resting'} <= self.behavior_seen:
                        self.smoke_finish(False)
                        return
                    self.smoke_checks.append('free behavior explores, turns, and rests while both clocks advance')
                    self.send('running', False)
                    self.smoke_stage = 2
            elif self.smoke_stage == 2 and not t['running']:
                self.behavior_pause = (t['sim_time'], b['behavior_time'])
                self.behavior_pause_wall = time.monotonic()
                self.grab().save(str(data_dir / 'resting-app.png'))
                self.smoke_stage = 3
            elif self.smoke_stage == 3 and time.monotonic()-self.behavior_pause_wall > .3:
                if (t['sim_time'], b['behavior_time']) != self.behavior_pause:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.append('pause preserves the brain, body, and behavior clocks')
                self.brain_panel.send('circuit', {'name': 'looming', 'rate_hz': 200})
                self.send('running', True)
                self.smoke_stage = 4
            elif self.smoke_stage == 4 and b['state'] == 'interrupted' and t['motor_effects']['escape'] > .1:
                self.smoke_checks.append('measured threat-pathway activity interrupts resting')
                self.grab().save(str(data_dir / 'interrupted-app.png'))
                self.brain_panel.send('release')
                self.smoke_stage = 5
            elif self.smoke_stage == 5 and not t['circuit_inputs'] and b['cycles'] >= 1 and b['state'] != 'interrupted':
                self.smoke_checks.append('normal behavior resumes after neural response subsides without reset')
                self.intervention_panel.resume.setChecked(True)
                for spin, value in zip(self.intervention_panel.durations, [100, 200, 200]):
                    spin.setValue(value)
                self.behavior_protocol_origin = t['sim_time']
                self.intervention_panel.buttons['defensive'].click()
                self.smoke_stage = 6
            elif self.smoke_stage == 6 and t['sim_time'] > self.behavior_protocol_origin+.7 and not t.get('protocol'):
                if not t['running'] or t['circuit_inputs'] or t['generation'] != 0:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.append('prepared experiment releases on schedule and continues free behavior')
                self.send('running', False)
                self.smoke_stage = 7
            elif self.smoke_stage == 7 and not t['running']:
                self.wander.setChecked(False)
                self.send('reset')
                self.smoke_stage = 8
            elif self.smoke_stage == 8 and t['generation'] == 1:
                success = t['sim_time'] == 0 and b['behavior_time'] == 0 and not b['enabled'] and not t['running']
                if success:
                    self.smoke_checks.append('reset restarts behavior timing and preserves the free-behavior toggle')
                self.smoke_finish(success)

        def response_smoke(self):
            panel, t = self.intervention_panel, self.brain_panel.telemetry
            if not self.brain_panel.ready or not t or not self.frame_count or self.brain_view.anatomy is None:
                return
            if abs(t['sim_time']-self.telemetry['sim_time']) > 1e-9:
                self.smoke_finish(False)
                return
            if self.smoke_stage == 0:
                self.send('food_config', {'enabled': False})
                if config.experimental:
                    if (t['dataset'] != 'male-cns:v1.0' or t['neurons'] != 166700
                            or self.brain_view.anatomy.manifest['snapshot'] != t['dataset']
                            or self.brain_view.anatomy.valid.sum() != 140638
                            or t['environment']['enabled'] or t['environment']['feedback_available']):
                        self.smoke_finish(False)
                        return
                    self.smoke_checks.append('MaleCNS loads 166700 neurons and matching real cell anchors; unmapped taste feedback stays disabled')
                self.tabs.setCurrentIndex(2)
                for spin, value in zip(panel.durations, [100, 500, 200]):
                    spin.setValue(value)
                self.response_cases = ['defensive', 'aversion', 'heat', 'seizure', 'heat_overload']
                self.response_index = 0
                self.smoke_stage = 1
            elif self.smoke_stage == 1:
                self.response_peak = self.response_escape = self.response_offset = 0.
                self.response_temperature = None
                self.response_max_active = 0
                self.response_active_capture = False
                name = self.response_cases[self.response_index]
                panel.buttons[name].click()
                self.smoke_stage = 2
            elif self.smoke_stage == 2:
                effects = t['motor_effects']
                self.response_max_active = max(self.response_max_active, self.brain_view.active_count)
                if config.experimental and self.brain_view.active_count and not self.response_active_capture:
                    self.grab().save(str(data_dir / (self.response_cases[self.response_index]+'-active.png')))
                    self.response_active_capture = True
                self.response_peak = max(self.response_peak, effects['disruption'])
                self.response_escape = max(self.response_escape, effects['escape'])
                self.response_offset = max(self.response_offset, self.telemetry['motor_offset_rms_rad'])
                if t['nominal_temperature_c'] is not None:
                    self.response_temperature = t['nominal_temperature_c']
                if not (t.get('protocol') or {}).get('completed'):
                    return
                name = self.response_cases[self.response_index]
                success = t['sim_time'] == .8 and not t['running'] and t['inhibition_gain'] == 1 and not t['circuit_inputs']
                if name == 'defensive':
                    success &= self.response_escape > .1 and self.response_offset > .02
                elif name == 'aversion':
                    success &= t['total_spikes'] > 0 and any(e.get('circuit') == 'aversion_proxy' for e in t['interventions'])
                elif name == 'heat':
                    success &= self.response_temperature == 40 and t['total_spikes'] > 1000
                else:
                    success &= self.response_peak > .1 and self.response_offset > .05
                    if name == 'heat_overload':
                        success &= self.response_temperature == 100
                if not success:
                    self.smoke_finish(False)
                    return
                if config.experimental and not self.response_max_active:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.append(name+' button runs, releases on schedule, and reports measured responses')
                self.grab().save(str(data_dir / (name+'-app.png')))
                panel.grab().save(str(data_dir / 'experiment-controls.png'))
                self.response_generation = t['generation']+1
                self.brain_panel.send('reset')
                self.smoke_stage = 3
            elif self.smoke_stage == 3 and t['generation'] == self.response_generation:
                if t['sim_time'] != 0 or t['total_spikes'] != 0 or t['motor_effects']['disruption'] != 0:
                    self.smoke_finish(False)
                    return
                self.response_index += 1
                if self.response_index < len(self.response_cases):
                    self.smoke_stage = 1
                else:
                    self.brain_panel.send('heat', 100)
                    self.brain_panel.send('circuit', {'name': 'looming', 'rate_hz': 200})
                    self.brain_panel.send('inhibition_gain', .25)
                    self.smoke_stage = 4
            elif self.smoke_stage == 4 and t['inhibition_gain'] == .25 and len(t['circuit_inputs']) == 2:
                if t['sim_time'] != 0:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.append('threat and heat inputs coexist while paused')
                panel.release.click()
                self.smoke_stage = 5
            elif self.smoke_stage == 5 and not t['circuit_inputs'] and t['inhibition_gain'] == 1:
                if t['sim_time'] != 0 or t['nominal_temperature_c'] is not None:
                    self.smoke_finish(False)
                    return
                self.smoke_checks.append('release clears scenario inputs and overlays without advancing time')
                panel.motor_enabled.setChecked(False)
                self.smoke_stage = 6
            elif self.smoke_stage == 6 and not t['motor_effects']['enabled']:
                self.smoke_checks.append('motor proxy ablation can be controlled from the native experiments tab')
                self.smoke_finish(True)

        def smoke_finish(self, success):
            self.smoke_timer.stop()
            report = self.diagnostics()
            report.update(success=success, checks=self.smoke_checks)
            (data_dir / "acceptance.json").write_text(json.dumps(report, indent=2))
            print(json.dumps({"success": success, "checks": self.smoke_checks}), flush=True)
            self.close()
            app.exit(0 if success else 1)

        def closeEvent(self, event):
            self.closed = True
            self.timer.stop()
            self.brain_panel.shutdown()
            if self.process.is_alive():
                try:
                    self.commands.put_nowait({"id": self.seq+1, "kind": "shutdown"})
                except Full:
                    pass
                self.process.join(timeout=2)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=2)
            for channel in (self.commands, self.frames, self.events):
                channel.cancel_join_thread()
                channel.close()
            event.accept()

    app.setStyleSheet("""
        QWidget { background: #ffffff; color: #000000; font-size: 13px; }
        QPushButton { background: #f5f6f7; padding: 8px; border: 1px solid #bcc3c8; border-radius: 3px; }
        QPushButton:hover { background: #e9edef; } QPushButton:checked { background: #dce2e5; border-color: #454e54; }
        QPushButton:disabled { color: #555555; background: #fafafa; border-color: #dedede; }
        QGroupBox { border: 1px solid #cbd0d4; margin-top: 12px; padding-top: 12px; }
        QGroupBox::title { subcontrol-origin: margin; padding: 0 4px; }
        QTextEdit, QPlainTextEdit { background: #ffffff; border: 1px solid #cbd0d4; }
        QTextEdit#eventLog { color: #b91c1c; font-family: monospace; font-size: 12px; }
        QTabBar::tab { padding: 9px 20px; background: #f4f5f6; border: 1px solid #cbd0d4; }
        QTabBar::tab:selected { background: #ffffff; border-bottom-color: #ffffff; }
        QTabWidget::pane { border: 1px solid #cbd0d4; }
        QSplitter::handle { background: #e4e7e9; }
    """)
    window = Window()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
