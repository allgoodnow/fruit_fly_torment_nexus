import numpy as np
import pytest
from nexus.brain.runtime import Brain, Connectome


def graph():
    return Connectome.from_edges(np.arange(100, 106),
        [0,0,1,1,2,2,3,4,5], [1,2,2,3,3,4,4,5,0],
        [20,10,25,-15,30,40,10,30,-20])


def test_matches_brian_voltage_spikes_delays_and_refractory():
    import brian2 as b
    b.start_scope()
    b.prefs.codegen.target = "numpy"
    b.defaultclock.dt = 0.1*b.ms
    g = graph()
    engine = Brain(g)
    engine.stimulate([100, 101], 500)
    events = np.random.default_rng(140).random((700, 2)) < .05
    neu = b.NeuronGroup(6,
        'dv/dt=(-52*mV-v+g)/(20*ms):volt (unless refractory)\n'
        'dg/dt=-g/(5*ms):volt (unless refractory)\nrfc:second',
        threshold='v > -45*mV', reset='v=-52*mV; g=0*mV',
        refractory='rfc', method='linear')
    neu.v = -52*b.mV
    neu.rfc = 2.2*b.ms
    neu.rfc[:2] = 0*b.ms
    syn = b.Synapses(neu, neu, 'w:volt', on_pre='g+=w', delay=1.8*b.ms)
    pre = np.repeat(np.arange(6), np.diff(g.offsets))
    syn.connect(i=pre, j=g.posts)
    syn.w = g.weights*b.mV
    @b.network_operation(when='synapses', order=0)
    def replay():
        tick = round(float(b.defaultclock.t/b.ms)*10)
        targets = np.flatnonzero(events[tick])
        targets = targets[np.asarray(neu.not_refractory[:])[targets]]
        neu.v[targets] += 68.75*b.mV
    spikes = b.SpikeMonitor(neu)
    states = b.StateMonitor(neu, ['v','g'], record=True, when='end')
    net = b.Network(neu, syn, replay, spikes, states)
    net.run(70*b.ms)
    actual = engine.advance(.07, input_events=events, trace_ids=g.ids)
    np.testing.assert_array_equal(actual['indices'], spikes.i[:])
    np.testing.assert_array_equal(actual['steps'], np.rint(spikes.t[:]/b.ms*10).astype(int))
    np.testing.assert_allclose(actual['voltage_mv'], states.v[:].T/b.mV, atol=2e-10, rtol=0)
    np.testing.assert_allclose(engine.g, neu.g[:]/b.mV, atol=2e-10, rtol=0)
    assert len(actual['indices']) > 30


def test_chunking_reset_and_release_preserve_state():
    a, c = Brain(graph()), Brain(graph())
    for brain in (a, c):
        brain.stimulate([100, 101], 500)
    a.advance(.05)
    for _ in range(5):
        c.advance(.01)
    for name in ('v','g','last_spike','queue','queue_size','counts'):
        np.testing.assert_array_equal(getattr(a,name),getattr(c,name))
    c.release()
    for name in ('v','g','last_spike','queue','queue_size','counts'):
        np.testing.assert_array_equal(getattr(a,name),getattr(c,name))
    assert c.time == .05 and len(c.inputs) == 0
    assert len(c.history) > 0
    c.reset()
    c.stimulate([100, 101], 500)
    c.advance(.05)
    np.testing.assert_array_equal(a.v,c.v)
    np.testing.assert_array_equal(a.counts,c.counts)
    assert c.time == a.time


def test_invalid_commands_do_not_remove_existing_stimulation():
    brain = Brain(graph(), history_limit=10)
    brain.stimulate([100], 500)
    for ids, rate in (([999],500),([100,100],500),([100],float('nan'))):
        with pytest.raises(ValueError):
            brain.stimulate(ids,rate)
        assert brain.inputs.tolist() == [0]
    brain.advance(.1)
    assert len(brain.history) == 10
    with pytest.raises(ValueError):
        brain.advance(.00015)


def test_output_silencing_blocks_downstream_activity_and_release_restores_it():
    brain = Brain(Connectome.from_edges([100,101], [0], [1], [50]))
    brain.stimulate([100],1000)
    brain.advance(.05)
    assert brain.counts[1] > 0
    brain.silence([100])
    brain.advance(.03)  # Allow existing postsynaptic state to decay.
    before = brain.counts.copy()
    brain.advance(.05)
    assert brain.counts[0] > before[0]
    assert brain.counts[1] == before[1]
    brain.release()
    brain.stimulate([100],1000)
    brain.advance(.05)
    assert brain.counts[1] > before[1]


def test_dense_activity_keeps_recording_bounded_and_counts_exact():
    brain = Brain(Connectome.from_edges(np.arange(100),[],[],[]))
    brain.stimulate(list(range(100)),1000)
    result = brain.advance(1)
    assert len(result['indices']) == 20000
    assert result['unrecorded_spikes'] > 0
    assert len(result['indices'])+result['unrecorded_spikes'] == brain.counts.sum()
    assert (np.diff(result['steps']) >= 0).all()
    assert len(brain.history) == 20000
