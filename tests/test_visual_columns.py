import copy
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pandas as pd
import pytest

from nexus.datasets.visual_columns import attach_visual_columns
from nexus.retina import VisualColumns
from nexus.coupled import CoupledSession
from test_vision import eye_brain, EyeBody, movie


def mapped_brain():
    b = eye_brain()
    b.graph.circuits['snapshot'] = 'male-cns:v1.0'
    b.graph.circuits['visual_columns'] = {'format': 'nexus-visual-columns-1', 'eyes': {
        'L': {'cells': [{'id': '10763', 'uv': [.1, .5]}]},
        'R': {'cells': [{'id': '11288', 'uv': [.9, .5]}]}}}
    return b


def test_column_inference_requires_independent_agreement_and_rejects_weak_tied_support():
    ids = np.arange(1, 11)
    rows = pd.DataFrame({'bodyId': ids, 'type': ['R1-R6']*2 + ['L1', 'L2', 'L1', 'L2']*2,
                         'rootSide': ['L', 'R'] + [None]*8,
                         'instance': ['R1-R6_L','R1-R6_R'] + ['L1_L','L2_L']*2 + ['L1_R','L2_R']*2,
                         'assignedOlHex1': [np.nan]*2 + [1,1,2,2]*2,
                         'assignedOlHex2': [np.nan]*2 + [1,1,2,2]*2})
    registry = {'snapshot': 'male-cns:v1.0', 'circuits': {
        'eye_left': {'ids': ['1']}, 'eye_right': {'ids': ['2']}}}
    offsets = np.array([0,4,6]+[6]*8)
    posts = np.array([2,3,4,5,6,7])
    contacts = np.array([20,20,1,1,20,20])
    result = attach_visual_columns(registry, rows, ids, offsets, posts, contacts)
    assert 'visual_columns' not in registry
    assert result['visual_columns']['eyes']['L']['cells'][0]['hex'] == [1,1]
    assert result['visual_columns']['eyes']['L']['cells'][0]['L1_L2_contacts'] == [20,20]
    # L1 prefers one column while L2 prefers the other; pooled dominance cannot override it.
    for bad in [[100,1,0,20,20,20], [20,20,20,20,20,20], [4,4,0,0,20,20], [20,20,5,5,20,20]]:
        with pytest.raises(ValueError, match='No reliable'):
            attach_visual_columns(registry, rows, ids, offsets, posts, np.array(bad))


def test_same_mean_different_image_layout_produces_different_inputs():
    mapping = VisualColumns(mapped_brain().graph.circuits)
    left = np.zeros((16,32,3), dtype=np.uint8)
    left[:,:16] = 255
    right = left[:,::-1].copy()
    assert left.mean() == right.mean()
    np.testing.assert_array_equal(mapping.sample(left)['L'], [1])
    np.testing.assert_array_equal(mapping.sample(left)['R'], [0])
    np.testing.assert_array_equal(mapping.sample(right)['L'], [0])
    np.testing.assert_array_equal(mapping.sample(right)['R'], [1])
    with pytest.raises(ValueError):
        mapping.sample(np.zeros((1,1)))


def test_mapping_rejects_foreign_duplicate_and_nonfinite_cells():
    for edit in ['foreign','duplicate','nan']:
        registry = copy.deepcopy(mapped_brain().graph.circuits)
        cells = registry['visual_columns']['eyes']['L']['cells']
        if edit == 'foreign': cells[0]['id'] = '11288'
        elif edit == 'duplicate': cells.append(cells[0])
        else: cells[0]['uv'][0] = float('nan')
        with pytest.raises(ValueError):
            VisualColumns(registry)


def test_spatial_inputs_validate_atomically_combine_max_and_release():
    b = mapped_brain()
    b.stimulate(['10763'], 80)
    b.set_spatial_eye_input({'L':[.4], 'R':[.7]})
    assert b.rates.tolist() == [80,70]
    for invalid in [{'L':[.5]}, {'L':[2], 'R':[.1]}, {'L':[.1,.2], 'R':[.5]}]:
        with pytest.raises(ValueError): b.set_spatial_eye_input(invalid)
        assert b.rates.tolist() == [80,70]
    b.clear_eye_input()
    assert b.rates.tolist() == [80]
    b.release()
    assert not len(b.inputs)


def test_spatial_video_mode_lifecycle_and_clock(movie):
    a,b = [CoupledSession(mapped_brain(), EyeBody()) for _ in range(2)]
    for s in [a,b]:
        with pytest.raises(ValueError): s.command('vision_mapping', 'spatial')
        s.command('vision_video', str(movie))
        s.command('vision_mapping', 'spatial')
        assert not s.running and not s.eyes.enabled
        s.command('eye_feedback', True)
    a.advance(1600)
    for _ in range(16): b.advance(100)
    np.testing.assert_array_equal(a.brain.counts,b.brain.counts)
    np.testing.assert_array_equal(a.brain.v,b.brain.v)
    assert a.eyes.snapshot()['rates_hz']['L'] > 90
    a.command('running', False)
    state=a.eyes.snapshot()
    assert state==a.eyes.snapshot()
    a.command('vision_restart')
    assert a.eyes.mapping_mode=='spatial' and not a.eyes.enabled and a.brain.step==1600
    a.command('eye_feedback', True)
    a.advance(3500)
    assert a.eyes.video.ended and not a.brain.eye_inputs and not a.eyes.enabled
    a.command('vision_eyes')
    assert a.eyes.mapping_mode=='pooled'
    b.command('vision_mapping', 'pooled')
    assert not b.brain.eye_inputs and not b.eyes.enabled
    a.eyes.close(); b.eyes.close()


def test_native_spatial_selector_requires_video_and_mapping_and_emits_change():
    from PySide6.QtWidgets import QApplication
    from nexus.vision_panel import VisionPanel
    app=QApplication.instance() or QApplication([])
    panel=VisionPanel()
    sent=[]
    panel.command.connect(lambda *args: sent.append(args))
    panel.update_snapshot({'available':True,'source':'eyes','spatial_available':True},None,False)
    assert not panel.mapping.isEnabled()
    panel.update_snapshot({'available':True,'source':'video','spatial_available':True},None,False)
    assert panel.mapping.isEnabled() and not sent
    panel.mapping.setCurrentIndex(1)
    assert sent==[('vision_mapping','spatial')]
    panel.update_snapshot({'available':True,'source':'video','spatial_available':True,'mapping_mode':'spatial'},None,False)
    assert len(sent)==1
    panel.close()
