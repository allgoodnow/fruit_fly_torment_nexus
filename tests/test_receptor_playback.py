import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QFileDialog

from nexus.brain.panel import BrainPanel
from nexus.coupled import CoupledSession
from nexus.receptor_playback import file_hash
from nexus.stimulation import active_stimulation
from nexus.vision_panel import VisionPanel
from test_receptor_replay import replay_model
from test_vision import EyeBody, movie

CURVE = Path(__file__).resolve().parents[1]/'experiments/receptor-release-assay-curve.json'


class PlaybackBody(EyeBody):
    def telemetry(self):
        return {'sim_time': self.time}


@pytest.fixture
def prepared_session(tmp_path, movie):
    brain = replay_model()
    brain.graph.registry_sha256 = 'synthetic-playback-fixture'
    brain.graph.circuits['readouts'] = {'steering_left': ['15'], 'steering_right': ['3'], 'mn9': ['15']}
    brain.graph.circuits['circuits']['giant_fiber'] = {'ids': ['3', '4']}
    voltage = np.full((2000, 2), -70.)
    voltage[1000:] = -60.
    np.savez_compressed(tmp_path/'response.npz', ids=np.asarray(['13', '14']), voltage_mv=voltage)
    report = {'format': 'nexus-video-phototransduction-1', 'success': True,
              'membrane_dt_ms': .1, 'video_input_fps': 20, 'exposure_ms': 200,
              'registry_sha256': brain.graph.registry_sha256, 'source_sha256': file_hash(movie),
              'receptors': [{'id': '13'}, {'id': '14'}],
              'white_rate_hz': 30000., 'black_rate_hz': 0., 'transfer': 'srgb', 'root_seed': 88300}
    path = tmp_path/'report.json'
    path.write_text(json.dumps(report))
    session = CoupledSession(brain, PlaybackBody())
    session.command('vision_video', str(movie))
    yield session, {'report_path': str(path), 'curve_path': str(CURVE)}, voltage
    session.eyes.close()


def test_prepared_input_follows_pause_release_protocol_and_recording_endpoint(prepared_session):
    session, files, voltage = prepared_session
    session.command('vision_receptors', files)
    assert session.brain.step == 0 and not session.running and not session.eyes.enabled
    session.command('eye_feedback', True)
    session.command('running', True)
    session.advance(333)
    saved = session.brain.v.copy(), session.brain.counts.copy(), copy.deepcopy(session.eyes.snapshot())
    session.command('running', False)
    assert session.eyes.snapshot() == saved[2]
    session.command('neural_release')
    session.advance(100)
    assert session.brain.step == 333 and session.body.time == pytest.approx(.0333)
    np.testing.assert_array_equal(session.brain.v, saved[0])
    np.testing.assert_array_equal(session.brain.counts, saved[1])
    assert 'VISION' not in active_stimulation(session.snapshot()['brain'])
    session.command('eye_feedback', True)
    session.command('protocol', {'format': 'nexus-protocol-1', 'duration_ms': 70., 'events': [
        {'at_ms': 0., 'action': 'release'},
        {'at_ms': 12., 'action': 'stimulate', 'ids': ['15'], 'rate_hz': 300.},
        {'at_ms': 32., 'action': 'release'}]})
    session.advance(1000)
    assert session.brain.step == 1033 and session.protocol.completed and not session.running
    assert session.eyes.enabled and not len(session.brain.inputs)
    assert 'VISION' in active_stimulation(session.snapshot()['brain'])
    assert any(e['kind'] == 'stimulate' and e['time'] == pytest.approx(.0453) for e in session.brain.events)
    session.command('running', True)
    session.advance(2000)
    assert session.brain.step == 2000 and session.body.time == pytest.approx(.2)
    assert session.eyes.receptors.ended and not session.running and not session.eyes.enabled
    assert session.eyes.video.index == 3  # Recording ends before the longer clip.
    np.testing.assert_array_equal(session.brain.v[session.brain.receptor_indices], voltage[-1])
    assert not session.brain.counts[session.brain.receptor_indices].any()
    assert not session.brain.counts[session.brain.graded_indices].any()
    session.command('reset')
    assert session.brain.time == session.body.time == 0 and session.eyes.receptors is None
    assert not session.brain.receptor_mask.any()


def test_native_loader_and_vision_controls_follow_prepared_state(prepared_session, monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    session, files, _ = prepared_session
    config = SimpleNamespace(experimental=True, first_label='Fixture', mn9_ids=('15',),
                             first_ids=('13',), left_ids=('15',), right_ids=('3',))
    sent = []
    brain_panel = BrainPanel(tmp_path, command_sink=lambda k, v: sent.append((k, v)), config=config)
    vision_panel = VisionPanel()
    packet = session.snapshot()
    brain_panel.receive_snapshot(packet['brain'])
    assert brain_panel.receptor_load.isEnabled()
    choices = iter([files['report_path'], files['curve_path']])
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (next(choices), ''))
    brain_panel.receptor_load.click()
    assert sent == [('vision_receptors', files)]
    session.command(*sent.pop())
    packet = session.snapshot()
    brain_panel.receive_snapshot(packet['brain'])
    vision_panel.update_snapshot(packet['brain']['eye_feedback'], packet['vision_pixels'], False)
    assert not brain_panel.receptor_load.isEnabled() and not brain_panel.graded.isEnabled()
    assert vision_panel.mapping.currentData() == 'receptor_replay'
    assert not vision_panel.mapping.isEnabled() and not vision_panel.adaptation.isEnabled()
    assert not vision_panel.restart.isEnabled() and not vision_panel.eyes.isEnabled()
    assert vision_panel.feed.isEnabled() and vision_panel.preview.image is not None
    session.command('eye_feedback', True)
    session.advance(2000)
    packet = session.snapshot()
    vision_panel.update_snapshot(packet['brain']['eye_feedback'], packet['vision_pixels'], False)
    assert vision_panel.status.text().startswith('Ended') and not vision_panel.feed.isEnabled()
    session.command('reset')
    packet = session.snapshot()
    brain_panel.receive_snapshot(packet['brain'])
    vision_panel.update_snapshot(packet['brain']['eye_feedback'], None, False)
    assert vision_panel.load.isEnabled() and vision_panel.eyes.isEnabled() and vision_panel.adaptation.isEnabled()
    assert vision_panel.mapping.currentData() == 'pooled' and brain_panel.graded.isEnabled()
    brain_panel.close()
    vision_panel.close()
