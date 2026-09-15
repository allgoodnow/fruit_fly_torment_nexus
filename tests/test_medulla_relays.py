import copy
import numpy as np
import pandas as pd
import pytest
from nexus.brain.runtime import Brain,Connectome
from nexus.datasets.graded_relays import attach_graded_relays,relay_ids,LAMINA_TYPES,MEDULLA_TYPES


def cohort():
    rows=[{'bodyId':i+1,'type':typ,'instance':f'{typ}_{side}','superclass':'ol_intrinsic'}
          for i,(typ,side) in enumerate((t,s) for t in LAMINA_TYPES+MEDULLA_TYPES for s in ('L','R'))]
    return pd.DataFrame(rows)


def model():
    a=cohort();ids=list(range(1,16))
    # Inputs 13/14 inhibit lamina 1/2; graded lamina -> medulla -> output 15.
    g=Connectome.from_edges(ids,[12,13,0,1,4,5],[0,1,4,5,14,14],[-600,-600,80,80,200,-50],snapshot='male-cns:v1.0')
    registry={'snapshot':g.snapshot,'circuits':{'eye_left':{'ids':['13']},'eye_right':{'ids':['14']}}}
    g.circuits=attach_graded_relays(registry,a,ids,include_medulla=True)
    b=Brain(g);b.set_graded_relays(True);return b


def test_extended_cohort_exact_and_requires_bounded_synapses():
    a=cohort();r={'snapshot':'male-cns:v1.0'}
    a=pd.concat([a,pd.DataFrame([{'bodyId':99,'type':'Tm3x','instance':'Tm3_L','superclass':'ol_intrinsic'}])],ignore_index=True)
    result=attach_graded_relays(r,a,a.bodyId,include_medulla=True)
    assert len(relay_ids(result))==12 and '99' not in relay_ids(result)
    assert result['graded_relays']['synapse_model']=='conductance-v1'
    del result['graded_relays']['synapse_model']
    with pytest.raises(ValueError):relay_ids(result)
    a.loc[a['type']=='Tm2','instance']='ambiguous'
    with pytest.raises(ValueError):attach_graded_relays(r,a,a.bodyId,include_medulla=True)


def test_conductance_solution_obeys_reversals_even_with_huge_input():
    b=model();b.silence([str(i) for i in range(1,13)])
    # Exercise both graded and spiking recipients without direct pulse input.
    b.histamine_g[[0,14]]=1e8
    b.excitation_g[[1,13]]=1e8
    b.advance(.0001)
    assert np.all(b.v>=-70) and np.all(b.v<=0)
    assert b.v[0]==pytest.approx(-70,abs=1e-5)
    assert b.v[1]==pytest.approx(0,abs=1e-5)
    b.advance(.02)
    assert np.isfinite(b.v).all() and np.all(b.v>=-70) and np.all(b.v<=0)
    assert not b.counts[b.graded_indices].any()


def test_voltage_changes_cross_two_nonspiking_stages_and_blockade_removes_them():
    runs={}
    for mode in ('dark','light','eye_block','medulla_block'):
        b=model()
        if mode in ('light','eye_block'):b.set_eye_input([1.,1.])
        if mode=='eye_block':b.silence(['13','14'])
        if mode=='medulla_block':b.silence(['5','6'])
        b.advance(.15);runs[mode]=b
        assert not b.counts[b.graded_indices].any()
        assert b.v.min()>=-70
    assert abs(runs['dark'].v[4]-runs['light'].v[4])>.5
    assert abs(runs['dark'].v[14]-runs['light'].v[14])>.1 or runs['dark'].counts[14]!=runs['light'].counts[14]
    np.testing.assert_array_equal(runs['dark'].v[:12],runs['eye_block'].v[:12])
    assert runs['dark'].v[14]==runs['eye_block'].v[14]
    assert runs['medulla_block'].v[14]==-52 and runs['medulla_block'].counts[14]==0


def test_chunking_reset_and_release_keep_conductance_state_consistent():
    a,b=model(),model()
    for item in (a,b):item.set_eye_input([.4,.7])
    a.advance(.1234)
    for ticks in [1,99,134,1000]:b.advance(ticks*.0001)
    for name in ('v','g','histamine_g','excitation_g','counts','graded_queue','queue','queue_size'):
        np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
    assert a.rng.bit_generator.state==b.rng.bit_generator.state
    held=a.excitation_g.copy();a.release()
    assert a.bounded_synapses and a.graded_enabled
    np.testing.assert_array_equal(a.excitation_g,held)
    a.reset();assert not a.bounded_synapses and not a.histamine_g.any() and not a.excitation_g.any()
    a.advance(.01);assert np.all(a.v==-52)
