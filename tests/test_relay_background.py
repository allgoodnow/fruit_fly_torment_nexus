import copy
import numpy as np
import pytest
from nexus.brain.runtime import Brain, Connectome
from nexus.stimulation import active_stimulation


def relay_brain(seed=73100):
    graph=Connectome.from_edges([1,2,3,4,5,6], [0,1,2,3], [2,3,4,5], [-60,-60,80,80], snapshot='male-cns:v1.0')
    graph.circuits={'circuits':{'eye_left':{'ids':['1']},'eye_right':{'ids':['2']}}}
    return Brain(graph,seed=seed)


def test_background_propagates_and_light_modulates_via_connections():
    runs={}
    for mode in ['baseline','light','eye_block','relay_block']:
        b=relay_brain();b.set_relay_background(True)
        if mode in ['light','eye_block']:b.set_eye_input([1.,1.])
        if mode=='eye_block':b.silence(['1','2'])
        if mode=='relay_block':b.silence(['3','4'])
        b.advance(.3);runs[mode]=b
    assert runs['baseline'].counts[4:].sum()>0
    assert runs['light'].counts[4:].sum()<runs['baseline'].counts[4:].sum()
    np.testing.assert_array_equal(runs['baseline'].counts[2:],runs['eye_block'].counts[2:])
    np.testing.assert_array_equal(runs['baseline'].v[2:],runs['eye_block'].v[2:])
    assert runs['relay_block'].counts[4:].sum()==0
    assert runs['relay_block'].counts[2:4].sum()>0


def test_bias_is_chunk_independent_and_keeps_poisson_stream_separate():
    a,b=relay_brain(),relay_brain()
    initial=copy.deepcopy(a.rng.bit_generator.state)
    for item in (a,b):
        item.set_relay_background(True);item.set_eye_input([.3,.7])
    assert initial==a.rng.bit_generator.state
    a.advance(.12)
    for _ in range(12):b.advance(.01)
    for name in ('v','g','counts','last_spike','queue','queue_size'):
        np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
    assert a.rng.bit_generator.state==b.rng.bit_generator.state
    np.testing.assert_array_equal(a.background,b.background)
    assert np.count_nonzero(a.background)==2
    assert np.all(a.refractory[a.background_targets]==22)


def test_invalid_changes_are_atomic_release_preserves_state_reset_reproduces():
    b=relay_brain();b.set_relay_background(True);b.advance(.1)
    saved={name:getattr(b,name).copy() for name in ('v','g','counts','background','queue')}
    for bad in ('yes',1,None):
        with pytest.raises(ValueError):b.set_relay_background(bad)
        assert b.background_enabled
    b.release()
    assert not b.background_enabled and b.time==.1
    for name,value in saved.items():np.testing.assert_array_equal(getattr(b,name),value)
    b.reset();assert not b.background_enabled
    b.set_relay_background(True);b.advance(.1)
    np.testing.assert_array_equal(b.counts,saved['counts'])
    np.testing.assert_array_equal(b.v,saved['v'])
    quiet=relay_brain();quiet.set_relay_background(True);quiet.set_relay_background(False);quiet.advance(.1)
    assert not quiet.counts.any() and np.all(quiet.v==-52)
    quiet.graph.snapshot='630';quiet.background_targets=None
    with pytest.raises(ValueError):quiet.set_relay_background(True)


def test_coupled_commands_pause_mark_input_active_and_release():
    from test_vision import eye_brain,EyeBody
    from nexus.coupled import CoupledSession
    b=eye_brain()
    graph=Connectome.from_edges(b.graph.ids,[0,4],[4,8],[-60,80],snapshot=b.graph.snapshot)
    graph.circuits=b.graph.circuits
    s=CoupledSession(Brain(graph),EyeBody())
    s.command('running',True);s.command('relay_background',True)
    assert not s.running and s.brain.background_enabled
    s.advance(1000)
    assert s.brain.counts.sum()>0 and s.brain.time==pytest.approx(s.body.time)
    held=s.brain.v.copy()
    s.command('brain',{'kind':'release'})
    assert not s.brain.background_enabled
    np.testing.assert_array_equal(s.brain.v,held)
    assert active_stimulation({'relay_background':{'enabled':True}})==('BASELINE',)
    assert active_stimulation({'relay_background':{'enabled':False}})==()
