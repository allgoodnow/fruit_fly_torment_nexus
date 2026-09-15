from pathlib import Path

import numpy as np
import pytest

from nexus.brain.photoreceptor import (
    PhotoreceptorMembrane, SOURCE_COMMIT, derivative, initial_state,
)

FIXTURE = Path(__file__).parent / 'fixtures/photoreceptor-membrane-v1.npz'


def test_derivatives_match_expressions_extracted_from_pinned_matlab():
    with np.load(FIXTURE, allow_pickle=False) as reference:
        assert str(reference['source_commit']) == SOURCE_COMMIT
        actual = np.array([derivative(y, current) for y, current in
                           zip(reference['states'], reference['currents_na'])])
        np.testing.assert_allclose(actual, reference['derivatives'], rtol=1e-11, atol=1e-11)


def test_current_pulses_match_adaptive_reference_without_threshold_resets():
    with np.load(FIXTURE, allow_pickle=False) as reference:
        currents = np.zeros((3000, 5))
        currents[1000:2000] = reference['pulse_amplitudes_na']
        model = PhotoreceptorMembrane(5)
        samples, voltage = [], []
        start = 0
        for end in reference['pulse_sample_indices'] + 1:
            voltage.append(model.advance(currents[start:end]))
            samples.append(model.state.copy())
            start = end
        actual = np.stack(samples, axis=1)
        np.testing.assert_allclose(actual, reference['pulse_reference_states'], rtol=1e-5, atol=1e-4)
        # A strong current can hold voltage above the old LIF threshold. It must
        # not trigger threshold/reset behavior in a graded photoreceptor.
        v = np.concatenate(voltage)
        assert np.all(v[1500:2000, -1] > -45.)
        assert v[-1, -1] < -69.


def test_replay_chunking_batch_independence_and_reset():
    currents = np.zeros((1234, 2))
    currents[111:809] = [.2, 1.]
    a, b = PhotoreceptorMembrane(2), PhotoreceptorMembrane(2)
    whole = a.advance(currents)
    pieces = np.concatenate([b.advance(currents[:1]), b.advance(currents[1:317]),
                             b.advance(currents[317:])])
    np.testing.assert_array_equal(whole, pieces)
    np.testing.assert_array_equal(a.state, b.state)
    assert a.step == b.step == len(currents)
    for cell in range(2):
        isolated = PhotoreceptorMembrane()
        np.testing.assert_array_equal(isolated.advance(currents[:, cell:cell+1])[:, 0], whole[:, cell])
    a.reset()
    np.testing.assert_array_equal(a.state, initial_state(2))
    assert a.step == 0


def test_removable_rate_singularities_are_finite_and_continuous():
    for voltage in (-23.8032, -59.639, 13.4859):
        y = initial_state()[0]
        y[0] = voltage
        center = derivative(y, .2)
        assert np.isfinite(center).all()
        for offset in (-1e-8, 1e-8):
            y[0] = voltage + offset
            np.testing.assert_allclose(derivative(y, .2), center, rtol=1e-7, atol=1e-7)


def test_invalid_inputs_and_failed_integration_preserve_state_and_clock():
    model = PhotoreceptorMembrane()
    model.advance(np.full((100, 1), .2))
    state, tick = model.state.copy(), model.step
    for invalid in ([], [1], [[np.nan]], [[np.inf]], [[1, 2]]):
        with pytest.raises(ValueError):
            model.advance(invalid)
        np.testing.assert_array_equal(model.state, state)
        assert model.step == tick
    with pytest.raises(RuntimeError):
        model.advance(np.full((10, 1), 1e8))
    np.testing.assert_array_equal(model.state, state)
    assert model.step == tick
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            PhotoreceptorMembrane(invalid)
