import copy
import numpy as np
import pytest
from nexus.brain.runtime import Brain, Connectome
from nexus.brain.protocol import Protocol


def brain():
    return Brain(Connectome.from_edges([100,101], [0],[1],[50]))


def description():
    return {"format":"nexus-protocol-1","duration_ms":35,"events":[
        {"at_ms":3.7,"action":"stimulate","ids":["100"],"rate_hz":1000},
        {"at_ms":22.4,"action":"release"}]}


def test_sequence_boundaries_match_manual_calls_and_finish_exactly():
    a,c = brain(),brain()
    protocol = Protocol(description(),a)
    for _ in range(4):
        protocol.advance(a,100)
    c.advance(.0037)
    c.stimulate([100],1000)
    c.advance(.0187)
    c.release()
    c.advance(.0126)
    assert protocol.completed and a.time == .035
    assert [(e['kind'],e['time']) for e in a.events] == [('stimulate',.0037),('release',.0224)]
    for name in ('v','g','last_spike','queue','queue_size','counts'):
        np.testing.assert_array_equal(getattr(a,name),getattr(c,name))


def test_validation_is_atomic_and_rejects_malformed_sequences():
    b = brain()
    b.stimulate([100],100)
    invalid = description()
    invalid['events'].append({'at_ms':30,'action':'stimulate','ids':['999'],'rate_hz':100})
    for value in (invalid, [], {'format':'nexus-protocol-1','duration_ms':10,'events':[None]}):
        with pytest.raises(ValueError):
            Protocol(value,b)
    assert b.inputs.tolist()==[0] and len(b.events)==1
