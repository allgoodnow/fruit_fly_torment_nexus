import copy
import numpy as np
import pandas as pd
import pytest
from nexus.brain.runtime import Brain,Connectome
from nexus.datasets.graded_relays import attach_graded_relays,relay_ids


def graded_brain():
    g=Connectome.from_edges(list(range(1,11)),[0,1,2,3,4,5],[2,3,6,7,8,9],[-600,-600,80,-80,80,-80],snapshot='male-cns:v1.0')
    g.circuits={'snapshot':'male-cns:v1.0','circuits':{'eye_left':{'ids':['1']},'eye_right':{'ids':['2']}},
                'graded_relays':{'format':'nexus-graded-relays-1','groups':{'L1_L':['3'],'L1_R':['4'],'L2_L':['5'],'L2_R':['6']}}}
    return Brain(g)


def test_exact_type_cohort_and_validation_do_not_expand_from_prefixes():
    a=pd.DataFrame({'bodyId':[1,2,3,4,5,6], 'type':['L1','L1','L2','L2','L1x','L1'],
                    'instance':['L1_L','L1_R','L2_L','L2_R','L1_L','L1_L'],
                    'superclass':['ol_intrinsic']*5+['cb_intrinsic']})
    r={'snapshot':'male-cns:v1.0'}
    result=attach_graded_relays(r,a,np.arange(1,7))
    assert set(relay_ids(result))=={'1','2','3','4'} and 'graded_relays' not in r
    with pytest.raises(ValueError):attach_graded_relays(r,a.iloc[:-3],np.arange(1,7))
    with pytest.raises(ValueError):attach_graded_relays(r,pd.concat([a,a]),np.arange(1,7))
    result['graded_relays']['groups']['L2_L']=['1']
    with pytest.raises(ValueError):relay_ids(result)


def test_analog_release_has_delay_and_propagates_without_relay_spikes():
    b=graded_brain();b.set_graded_relays(True)
    b.advance(.0018)
    assert not b.g.any() and not b.counts.any()
    b.advance(.0001)
    np.testing.assert_allclose(b.g[6:],[1.6,-1.6,1.6,-1.6])
    assert np.all(b.v==-52)
    b.advance(.05)
    assert not b.counts[b.graded_indices].any()
    assert b.v[7]<-52 and b.v[9]<-52 and b.counts[6]>0


def test_photoreceptor_conductance_is_bounded_and_changes_analog_output():
    runs={}
    for condition in ['baseline','light','eye_block','relay_block']:
        b=graded_brain();b.set_graded_relays(True)
        if condition in ('light','eye_block'):b.set_eye_input([1.,1.])
        if condition=='eye_block':b.silence(['1','2'])
        if condition=='relay_block':b.silence(['3','4','5','6'])
        result=b.advance(.2,trace_ids=[3,4]);runs[condition]=b
        assert result['voltage_mv'].min()>=-70
        assert not b.counts[b.graded_indices].any()
    assert runs['light'].v[2]<runs['baseline'].v[2]
    assert runs['light'].counts[6]<runs['baseline'].counts[6]
    np.testing.assert_array_equal(runs['baseline'].counts[2:],runs['eye_block'].counts[2:])
    np.testing.assert_array_equal(runs['baseline'].v[2:],runs['eye_block'].v[2:])
    assert not runs['relay_block'].counts.any() and np.all(runs['relay_block'].v==-52)


def test_chunking_release_and_reset_preserve_the_right_states():
    a,b=graded_brain(),graded_brain()
    for item in (a,b):item.set_graded_relays(True);item.set_eye_input([.3,.6])
    a.advance(.1234)
    for step in [100,17,500,617]:b.advance(step*.0001)
    for name in ['v','g','histamine_g','counts','queue','graded_queue']:
        np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
    assert a.rng.bit_generator.state==b.rng.bit_generator.state
    before={name:getattr(a,name).copy() for name in ['v','g','histamine_g','counts','graded_queue']}
    a.release();assert a.graded_enabled and not a.eye_inputs
    for name,value in before.items():np.testing.assert_array_equal(getattr(a,name),value)
    a.reset();assert not a.graded_enabled and not a.histamine_g.any() and not a.graded_queue.any()
    a.advance(.02);assert not a.counts.any() and np.all(a.v==-52)


def test_model_switches_are_atomic_and_require_zero_time():
    b=graded_brain();b.set_relay_background(True)
    with pytest.raises(ValueError):b.set_graded_relays(True)
    assert not b.graded_enabled and b.background_enabled
    b.set_relay_background(False);b.set_graded_relays(True)
    with pytest.raises(ValueError):b.set_relay_background(True)
    b.advance(.01);state=b.v.copy();rng=copy.deepcopy(b.rng.bit_generator.state)
    for value in [False,True,'yes',1]:
        with pytest.raises(ValueError):b.set_graded_relays(value)
    assert b.graded_enabled and b.rng.bit_generator.state==rng
    np.testing.assert_array_equal(b.v,state)


def test_silencing_applies_to_already_queued_analog_release():
    b=graded_brain();b.set_graded_relays(True);b.advance(.0001)
    assert b.graded_queue.any()
    b.silence(['3','4','5','6']);b.advance(.01)
    assert not b.g.any() and np.all(b.v==-52)
