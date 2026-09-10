import pytest

from nexus.stimulation import active_stimulation
from nexus.brain.protocol import Protocol
from nexus.brain.runtime import Brain, Connectome
from nexus.brain.telemetry import NeuralTelemetry


def packet(*names, temperature=None, gain=1, **extra):
    return dict(circuit_inputs={n: {'ids': ['1'], 'rate_hz': 100} for n in names},
                nominal_temperature_c=temperature, inhibition_gain=gain, **extra)


@pytest.mark.parametrize('state,expected', [
    (packet('looming'), ('FEAR',)),
    (packet('nociception_proxy'), ('PAIN',)),
    (packet('aversion_proxy'), ('PAIN',)),
    (packet('warmth', gain=.25), ('SEIZURE',)),
    (packet('warmth', temperature=100, gain=.25), ('BOILING',)),
    (packet('warmth', temperature=40), ('HEAT',)),
    (packet('looming', 'nociception_proxy', 'warmth', temperature=100, gain=.25), ('FEAR', 'PAIN', 'BOILING')),
    (packet('nociception_proxy', 'aversion_proxy'), ('PAIN',)),
    (packet(manual_ids=['1']), ('CUSTOM',)),
    (packet(manual_ids=['1'], gain=.25), ('SEIZURE',)),
    (packet(sensory_ids=['1'], sensory_rate_hz=200, silenced_count=1), ('SILENCING',)),
    (packet(total_spikes=100000, motor_effects={'disruption': 1}, protocol={'name': 'Pain'}), ()),
])
def test_labels_follow_inputs_not_buttons_residual_spikes_or_body_motion(state, expected):
    assert active_stimulation(state) == expected
    state['running'] = False
    assert active_stimulation(state) == expected  # Pausing preserves applied input.


def test_no_label_for_zero_rate_input_or_temperature_without_warmth_input():
    state = packet('looming', temperature=100)
    state['circuit_inputs']['looming']['rate_hz'] = 0
    assert active_stimulation(state) == ()


def test_live_protocol_baseline_onset_overlap_release_and_reset():
    brain = Brain(Connectome.from_edges([1, 2, 3], [], [], [], snapshot='male-cns:v1.0'))
    brain.graph.circuits = {'circuits': {name: {'ids': [str(i)]} for name, i in
                                        [('looming', 1), ('nociception_proxy', 2), ('warmth', 3)]}}
    brain.graph.circuits['readouts'] = {'mn9': ['1']}
    telemetry = NeuralTelemetry(brain)
    plan = Protocol({'format': 'nexus-protocol-1', 'duration_ms': 100,
                     'events': [{'at_ms': 10, 'action': 'circuit', 'name': 'looming', 'rate_hz': 100},
                                {'at_ms': 20, 'action': 'circuit', 'name': 'nociception_proxy', 'rate_hz': 100},
                                {'at_ms': 30, 'action': 'release'}]}, brain)
    def labels():
        return active_stimulation(telemetry.snapshot(brain, False, 0, plan))
    assert labels() == ()
    plan.advance(brain, 100)
    assert labels() == ('FEAR',)
    plan.advance(brain, 100)
    assert labels() == ('FEAR', 'PAIN')
    before = brain.step
    assert labels() == ('FEAR', 'PAIN') and brain.step == before
    plan.advance(brain, 100)
    assert labels() == ()
    brain.set_circuit_input('warmth', 100)
    brain.reset()
    assert labels() == ()
