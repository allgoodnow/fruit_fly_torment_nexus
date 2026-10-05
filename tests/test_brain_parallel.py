import numpy as np
from numba import get_num_threads

from nexus.brain.runtime import Brain
from nexus.brain.telemetry import NeuralTelemetry
from test_receptor_replay import replay_model


def test_parallel_receptor_replay_preserves_states_events_and_thread_policy():
    serial = replay_model()
    serial.graph.circuits['readouts'] = {'mn9': ['15']}
    parallel = Brain(serial.graph, integration_threads=2)
    parallel.set_graded_relays(True)
    voltage = np.full((1200, 2), -70.)
    voltage[300:] = -60.
    release = np.where(voltage == -70., 5., 20.)
    previous_threads = get_num_threads()
    traces = [[], []]
    for brain in (serial, parallel):
        brain.configure_receptor_replay(['13', '14'], reference_mv=-70.)
    cursor = 0
    # Normal control changes between chunks keep both queues and the RNG intact.
    for ticks, command in [(303, 'input'), (497, 'release'), (400, None)]:
        for j, brain in enumerate((serial, parallel)):
            if command == 'input':
                brain.stimulate(['15'], 250.)
            elif command == 'release':
                brain.release()
            result = brain.advance(ticks*.0001, trace_ids=['13', '1', '5', '15'],
                                   receptor_voltage_mv=voltage[cursor:cursor+ticks],
                                   receptor_release_hz=release[cursor:cursor+ticks])
            traces[j].append(result['voltage_mv'])
        assert get_num_threads() == previous_threads
        for name in ('v', 'g', 'enabled', 'last_spike', 'histamine_g', 'excitation_g',
                     'counts', 'queue', 'queue_size', 'graded_queue', 'receptor_queue'):
            np.testing.assert_array_equal(getattr(serial, name), getattr(parallel, name))
        assert list(serial.history) == list(parallel.history)
        assert list(serial.activity) == list(parallel.activity)
        assert serial.rng.bit_generator.state == parallel.rng.bit_generator.state
        cursor += ticks
    np.testing.assert_array_equal(np.concatenate(traces[0]), np.concatenate(traces[1]))
    assert serial.counts.sum() > 0
    assert not parallel.counts[parallel.receptor_indices].any()
    snapshot = NeuralTelemetry(parallel).snapshot(parallel, False, 0)
    assert snapshot['runtime'] == {'backend': 'cpu', 'integration_threads': 2,
                                  'synaptic_delivery': 'serial', 'dt_ms': .1}
    parallel.reset()
    assert parallel.integration_threads == 2
    parallel.advance(.01)
    np.testing.assert_array_equal(parallel.v, -52.)
