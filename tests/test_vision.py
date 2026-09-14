"""Visual input routing, sampling clock, and real local video decoding."""
import copy

import numpy as np
import pandas as pd
import pytest

from nexus.brain.runtime import Brain
from nexus.coupled import CoupledSession
from nexus.datasets.eyes import attach_eye_inputs
from nexus.vision import pooled_brightness
from nexus.video import VideoSource
from test_descending import brain as base_brain
from test_coupled import ClockBody


def eye_brain():
    brain = base_brain()
    brain.graph.circuits['circuits']['eye_left'] = {'ids': ['10763']}
    brain.graph.circuits['circuits']['eye_right'] = {'ids': ['11288']}
    return brain


class EyeBody(ClockBody):
    def eye_brightness(self):
        return np.array([.2 + self.time, .6])


@pytest.fixture
def movie(tmp_path):
    import imageio_ffmpeg
    path = tmp_path / 'black white.mp4'
    writer = imageio_ffmpeg.write_frames(str(path), (64, 48), fps=20, codec='libx264', quality=10,
                                        macro_block_size=1, ffmpeg_log_level='error')
    writer.send(None)
    for brightness in [0, 0, 255, 255, 0, 0]:
        writer.send(np.full((48, 64, 3), brightness, dtype=np.uint8))
    writer.close()
    return path


def test_eye_mask_and_left_right_are_not_spectator_brightness():
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:4, 2:4] = True
    frames[0, mask] = 255
    np.testing.assert_array_equal(pooled_brightness(frames, mask), [1, 0])
    with pytest.raises(ValueError):
        pooled_brightness(frames, np.zeros_like(mask))


def test_eye_input_is_independent_max_combined_and_released():
    brain = eye_brain()
    brain.stimulate(['10763'], 80)
    brain.set_eye_input([.4, .7])
    assert brain.rates.tolist() == [80, 70]
    brain.set_eye_input([0, 0])
    assert brain.rates.tolist() == [80]
    brain.set_eye_input([.4, .7])
    before = brain.inputs.copy(), brain.rates.copy()
    for invalid in [[float('nan'), .3], [1.1, 0], [1]]:
        with pytest.raises(ValueError):
            brain.set_eye_input(invalid)
        np.testing.assert_array_equal(brain.inputs, before[0])
        np.testing.assert_array_equal(brain.rates, before[1])
    brain.release()
    assert not brain.eye_inputs and not len(brain.inputs)


def test_sampling_is_chunk_independent_and_pause_does_not_resample():
    sessions = [CoupledSession(eye_brain(), EyeBody()) for _ in range(2)]
    for s in sessions:
        s.command('eye_feedback', True)
    sessions[0].advance(1200)
    for _ in range(12):
        sessions[1].advance(100)
    for name in ['v', 'g', 'counts', 'rates']:
        np.testing.assert_array_equal(getattr(sessions[0].brain, name), getattr(sessions[1].brain, name))
    s = sessions[0]
    assert s.eyes.samples == 3 and s.eyes.sample_step == 1000
    saved = s.eyes.snapshot()
    s.command('running', False)
    assert s.eyes.snapshot() == saved
    s.command('neural_release')
    assert not s.eyes.enabled and not s.brain.eye_inputs
    s.advance(100)
    assert s.eyes.samples == 3


def test_registry_preserves_scan_side_counts_and_rejects_ambiguous_inputs():
    registry = {'snapshot': 'male-cns:v1.0', 'circuits': {}}
    table = pd.DataFrame({'bodyId': [1, 2, 3], 'type': ['R1-R6'] * 3,
                          'rootSide': ['L', 'R', 'R'], 'superclass': ['ol_sensory'] * 3})
    nt = pd.DataFrame({'body': [3, 1, 2], 'consensus_nt': ['histamine'] * 3})
    result = attach_eye_inputs(registry, table, nt, [1, 2, 3])
    assert result['circuits']['eye_left']['ids'] == ['1']
    assert result['circuits']['eye_right']['ids'] == ['2', '3']
    assert not registry['circuits']
    table.loc[0, 'rootSide'] = None
    with pytest.raises(ValueError, match='side'):
        attach_eye_inputs(registry, table, nt, [1, 2, 3])


def test_real_mp4_decodes_bounded_frames_and_ends(movie):
    video = VideoSource(movie)
    try:
        assert video.frame.shape[0] <= 180 and video.frame.shape[1] <= 320
        assert video.frame.mean() < 1
        assert video.frame_at(2).mean() > 250  # H.264/YUV round-trip is not lossless RGB.
        assert video.frame_at(5).mean() < 1
        assert video.frame_at(6) is None and video.ended and video.stream is None
    finally:
        video.close()


def test_video_pause_disable_resume_restart_and_end(movie):
    s = CoupledSession(eye_brain(), EyeBody())
    s.command('vision_video', str(movie))
    assert not s.running and not s.eyes.enabled and s.eyes.video.index == 0
    s.command('eye_feedback', True)
    s.advance(1000)
    assert s.eyes.video.index == 1 and not s.brain.eye_inputs
    s.command('eye_feedback', False)
    s.advance(1000)
    assert s.eyes.video.index == 1
    s.command('eye_feedback', True)
    s.advance(500)
    assert s.eyes.video.index == 2 and s.brain.eye_inputs
    s.command('running', False)
    saved = copy.deepcopy(s.eyes.snapshot())
    assert s.eyes.snapshot() == saved
    s.advance(2000)
    assert s.eyes.video.ended and not s.eyes.enabled and not s.brain.eye_inputs
    s.command('vision_restart')
    assert not s.running and s.eyes.video.index == 0 and not s.eyes.enabled
    s.command('eye_feedback', True)
    s.command('reset')
    assert s.brain.step == 0 and s.eyes.video is None and s.eyes.preview is None
    assert not s.eyes.enabled and not s.brain.eye_inputs


def test_bad_video_preserves_current_source_and_prepared_trials_disable_feedback(movie, tmp_path):
    s = CoupledSession(eye_brain(), EyeBody())
    s.command('vision_video', str(movie))
    s.command('eye_feedback', True)
    source = s.eyes.video
    bad = tmp_path / 'bad.mp4'
    bad.write_bytes(b'not a movie')
    with pytest.raises(ValueError, match='Cannot decode'):
        s.command('vision_video', str(bad))
    assert s.eyes.video is source and s.eyes.enabled
    with pytest.raises(ValueError):
        s.command('protocol', {'format': 'invalid'})
    assert s.eyes.enabled
    s.command('protocol', {'format': 'nexus-protocol-1', 'duration_ms': 100, 'events': []})
    assert not s.eyes.enabled
    with pytest.raises(ValueError, match='prepared sequence'):
        s.command('eye_feedback', True)
    s.eyes.close()


def test_playback_error_releases_input_and_changing_source_clears_error(movie, monkeypatch):
    s = CoupledSession(eye_brain(), EyeBody())
    s.command('vision_video', str(movie))
    s.command('eye_feedback', True)
    s.advance(1500)
    assert s.brain.eye_inputs
    def broken_frame(index):
        raise ValueError('Video decoding failed during playback')
    monkeypatch.setattr(s.eyes.video, 'frame_at', broken_frame)
    s.advance(500)
    assert s.eyes.error and not s.eyes.enabled and not s.brain.eye_inputs
    s.command('vision_video', str(movie))
    assert s.eyes.error is None
    s.command('eye_feedback', True)
    s.advance(500)
    s.command('vision_eyes')
    assert s.eyes.video is None and s.eyes.samples == 0 and s.eyes.error is None


def test_vision_panel_load_controls_and_snapshot_do_not_send_extra_commands(movie, monkeypatch):
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication, QFileDialog
    from nexus.vision_panel import VisionPanel
    app = QApplication.instance() or QApplication([])
    panel = VisionPanel()
    sent = []
    panel.command.connect(lambda *args: sent.append(args))
    state = {'available': True, 'enabled': True, 'source': 'video',
             'video_frame': 2, 'video_name': movie.name}
    pixels = np.full((48, 64, 3), 127, dtype=np.uint8)
    panel.update_snapshot(state, pixels, False)
    assert not sent and panel.feed.isChecked()
    assert 'Paused' in panel.status.text() and not panel.preview.image.isNull()
    pixels[:] = 0
    assert panel.preview.image.pixelColor(0, 0).red() == 127
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(movie), ''))
    panel.load.click()
    assert sent.pop() == ('vision_video', str(movie))
    panel.restart.click()
    assert sent.pop() == ('vision_restart', None)
    panel.feed.click()
    assert sent.pop() == ('eye_feedback', False)
    panel.eyes.click()
    assert sent.pop() == ('vision_eyes', None)
    panel.update_snapshot({'available': True, 'source': 'eyes'}, None, False)
    assert not panel.restart.isEnabled() and panel.preview.image is None
    panel.close()
