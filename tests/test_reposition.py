import copy
import numpy as np
import mujoco
from nexus.body import FlyBody
from nexus.coupled import CoupledSession
from nexus.brain.motor import DNA02
from test_coupled import small_brain
from test_recovery import released_monitor, observe


def test_manual_reposition_preserves_neural_state_clocks_location_and_pending_sequence():
    b = small_brain()
    body = FlyBody(render=False)
    try:
        s = CoupledSession(b, body, autonomous=True)
        s.command('protocol', {'format':'nexus-protocol-1','duration_ms':500,
                              'events':[{'at_ms':0,'action':'stimulate','ids':[DNA02[0]],'rate_hz':200},
                                        {'at_ms':400,'action':'release'}]})
        s.advance(1000)
        free = np.flatnonzero(body.sim.mj_model.jnt_type==mujoco.mjtJoint.mjJNT_FREE)[0]
        adr = body.sim.mj_model.jnt_qposadr[free]
        body.sim.mj_data.qpos[adr+3:adr+7] = [0,1,0,0]
        mujoco.mj_forward(body.sim.mj_model,body.sim.mj_data)
        assert body.upright() < -.9
        xy, origin, path = body.position()[:2].copy(),body.origin.copy(),list(body.path)
        fields=['v','g','counts','queue','queue_size','inputs','rates','output_gain']
        arrays={k:getattr(b,k).copy() for k in fields}
        rng=copy.deepcopy(b.rng.bit_generator.state)
        protocol=s.protocol; cursor=protocol.cursor
        s.command('reposition_body')
        assert b.time==body.time==.1 and not s.running and s.generation==0
        assert s.protocol is protocol and protocol.cursor==cursor
        assert s.reposition_count==1 and body.upright()>.95
        np.testing.assert_allclose(body.position()[:2],xy,atol=1e-9)
        np.testing.assert_array_equal(body.origin,origin)
        assert list(body.path)==path
        for k in fields:np.testing.assert_array_equal(getattr(b,k),arrays[k])
        assert b.rng.bit_generator.state==rng
        assert s.recovery.status()['assisted_repositions']==1
        s.command('running',True)
        s.advance(100)
        assert abs(b.time-body.time)<1e-9
        s.command('reset')
        assert s.reposition_count==0 and b.time==body.time==0
    finally:
        body.close()


def test_reposition_never_counts_as_immediate_settling():
    m=released_monitor()
    observe(m,5100)
    assert m.status()['phase']=='settled'
    m.body_repositioned(5100)
    assert m.status()['phase']=='observing' and m.status()['settled_at_ms'] is None
    assert m.events[-1]['kind']=='body_repositioned'
    observe(m,10099)
    assert m.status()['phase']=='observing'
    observe(m,10100)
    assert m.status()['phase']=='settled'
