import copy
import math
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
import pytest

from nexus.light_adaptation import LightAdaptation, TAU_MS, STRENGTH
from nexus.coupled import CoupledSession
from test_vision import eye_brain, EyeBody, movie
from test_visual_columns import mapped_brain


def test_sustained_light_adapts_and_darkness_recovers_sensitivity_per_channel():
    a=LightAdaptation()
    first=a.sample({'L':np.array([1.,0.]), 'R':np.array([.5,1.])})
    np.testing.assert_array_equal(first['L'],[1,0])
    a.advance(10000)
    settled=a.sample({'L':np.array([1.,0.]),'R':np.array([.5,1.])})
    np.testing.assert_allclose(settled['L'][0], 1/(1+STRENGTH*(1-math.exp(-1000/TAU_MS))))
    assert .25<settled['L'][0]<.26 and settled['L'][1]==0
    # A previously dark cell responds strongly; its bright neighbour retains adaptation.
    switched=a.sample({'L':np.array([1.,1.]), 'R':np.array([0.,0.])})
    assert switched['L'][1]==1 and switched['L'][0]<.26
    a.sample({'L':np.array([0.,0.]),'R':np.array([0.,0.])})
    a.advance(10000)
    recovered=a.sample({'L':np.array([1.,0.]),'R':np.array([.5,1.])})
    assert recovered['L'][0]>.94


def test_exposure_clock_is_chunk_independent_and_invalid_changes_are_atomic():
    a,b=LightAdaptation(),LightAdaptation()
    for item in [a,b]:item.sample({'L':[.3,1.]})
    a.advance(500)
    for _ in range(5):b.advance(100)
    assert a.snapshot()==b.snapshot()
    np.testing.assert_array_equal(a.sample({'L':[.8,.2]})['L'],b.sample({'L':[.8,.2]})['L'])
    before=copy.deepcopy(a.snapshot())
    for value in [{'L':[float('nan'),0]}, {'L':[1.2,0]}, {'R':[.2]}, {'L':[.3]}]:
        with pytest.raises(ValueError):a.sample(value)
        assert a.snapshot()==before
    with pytest.raises(ValueError):a.advance(.5)
    a.reset()
    assert a.snapshot()['exposure_ms']==0 and not a.snapshot()['mean_background']


def test_coupled_adaptation_pause_disable_resume_release_and_reset():
    sessions=[CoupledSession(eye_brain(),EyeBody()) for _ in range(2)]
    for s in sessions:
        s.command('vision_adaptation',True)
        s.command('eye_feedback',True)
    sessions[0].advance(1200)
    for _ in range(12):sessions[1].advance(100)
    a,b=sessions
    np.testing.assert_array_equal(a.brain.v,b.brain.v)
    np.testing.assert_array_equal(a.brain.counts,b.brain.counts)
    assert a.eyes.snapshot()==b.eyes.snapshot()
    assert a.eyes.snapshot()['rates_hz']['R']<60
    a.command('running',False)
    state=a.eyes.adaptation.snapshot()
    assert state==a.eyes.adaptation.snapshot()
    a.command('neural_release')
    a.advance(500)
    assert not a.brain.eye_inputs and a.eyes.adaptation.snapshot()==state
    a.command('eye_feedback',True)
    a.advance(100)
    assert a.eyes.adaptation.snapshot()['exposure_ms']==130
    a.command('reset')
    assert not a.eyes.adaptive and a.eyes.adaptation.snapshot()['exposure_ms']==0


def test_video_mapping_changes_restart_and_source_reset_only_adaptation(movie):
    s=CoupledSession(mapped_brain(),EyeBody())
    s.command('vision_video',str(movie))
    s.command('vision_mapping','spatial')
    s.command('vision_adaptation',True)
    s.command('eye_feedback',True)
    s.advance(2000)
    assert s.eyes.adaptation.snapshot()['exposure_ms']==200
    assert s.eyes.snapshot()['rates_hz']['L']<99
    clock=s.brain.step
    s.command('vision_restart')
    assert s.eyes.adaptive and s.eyes.mapping_mode=='spatial'
    assert s.brain.step==clock and s.eyes.adaptation.snapshot()['exposure_ms']==0
    s.command('eye_feedback',True)
    s.advance(1500)
    s.command('vision_mapping','pooled')
    assert s.eyes.adaptive and not s.brain.eye_inputs and not s.eyes.adaptation.background
    s.command('eye_feedback',True)
    s.advance(100)
    s.command('vision_eyes')
    assert s.eyes.adaptive and not s.eyes.adaptation.background
    with pytest.raises(ValueError):s.command('vision_adaptation','yes')
    assert s.eyes.adaptive
    s.eyes.close()


def test_adaptation_checkbox_uses_acknowledged_state_without_reissuing_commands():
    from PySide6.QtWidgets import QApplication
    from nexus.vision_panel import VisionPanel
    app=QApplication.instance() or QApplication([])
    panel=VisionPanel();sent=[]
    panel.command.connect(lambda *args:sent.append(args))
    panel.update_snapshot({'available':True,'adaptation':{'selected':True}},None,False)
    assert panel.adaptation.isChecked() and not sent
    panel.adaptation.click()
    assert sent==[('vision_adaptation',False)]
    panel.close()
