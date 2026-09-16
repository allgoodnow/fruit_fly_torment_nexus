from pathlib import Path

import numpy as np
import pytest

from nexus.brain.photoreceptor import (
    PhotoreceptorMembrane, SOURCE_COMMIT, channel_derivative, initial_state,
    trp_current_na,
)

FIXTURE = Path(__file__).parent / 'fixtures/photoreceptor-channels-v1.npz'


def test_channel_current_and_derivatives_match_pinned_source_expressions():
    with np.load(FIXTURE, allow_pickle=False) as reference:
        assert str(reference['source_commit']) == SOURCE_COMMIT
        currents = [trp_current_na(n, y[0]) for n, y in
                    zip(reference['open_channels'], reference['states'])]
        derivatives = [channel_derivative(y, n) for n, y in
                       zip(reference['open_channels'], reference['states'])]
        np.testing.assert_allclose(currents, reference['currents_na'], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(derivatives, reference['derivatives'], rtol=1e-11, atol=1e-11)
    # Explicit unit anchor: 1 channel * 8 pS * 90 mV = .72 pA = .00072 nA.
    assert trp_current_na(1, -70) == pytest.approx(.00072)
    assert trp_current_na(27, 20) == 0
    assert trp_current_na(27, 25) == 0  # Source rectification, not a voltage clamp.


def test_channel_pulses_match_independent_adaptive_reference():
    with np.load(FIXTURE, allow_pickle=False) as reference:
        channels = np.zeros((3000, 5), dtype=np.int64)
        channels[1000:2000] = reference['pulse_amplitudes']
        model = PhotoreceptorMembrane(5)
        samples, start = [], 0
        for end in reference['sample_indices'] + 1:
            model.advance_channels(channels[start:end])
            samples.append(model.state.copy())
            start = end
        np.testing.assert_allclose(np.stack(samples, axis=1), reference['pulse_reference_states'],
                                   rtol=1e-5, atol=3e-4)


def test_feedback_reduces_drive_and_zero_openings_preserve_membrane_recovery():
    a, b = PhotoreceptorMembrane(), PhotoreceptorMembrane()
    channels = np.full((1000, 1), 5400)
    live = a.advance_channels(channels)
    frozen = b.advance(channels * trp_current_na(1, -70))
    assert live[-1, 0] < frozen[-1, 0]
    assert trp_current_na(5400, live[-1, 0]) < trp_current_na(5400, -70)
    assert np.all(live[500:] > -45)  # Graded depolarization above the old threshold.
    held = a.state.copy()
    recovery = a.advance_channels(np.zeros((1000, 1), dtype=int))
    assert recovery[0, 0] != initial_state()[0, 0]
    assert recovery[-1, 0] < -69
    assert not np.array_equal(a.state, held)
    # Switching input representation does not reset or retain imposed current.
    c = PhotoreceptorMembrane()
    c.state[:] = held
    np.testing.assert_array_equal(c.advance(np.zeros((1000, 1))), recovery)
    np.testing.assert_array_equal(c.state, a.state)


def test_channel_batches_chunking_reset_and_zero_current_equivalence():
    channels = np.zeros((1234, 2), dtype=int)
    channels[113:877] = [270, 2700]
    a, b = PhotoreceptorMembrane(2), PhotoreceptorMembrane(2)
    whole = a.advance_channels(channels)
    chunks = np.concatenate([b.advance_channels(channels[:113]),
                             b.advance_channels(channels[113:651]),
                             b.advance_channels(channels[651:])])
    np.testing.assert_array_equal(whole, chunks)
    np.testing.assert_array_equal(a.state, b.state)
    assert a.step == b.step == 1234
    for cell in range(2):
        isolated = PhotoreceptorMembrane()
        np.testing.assert_array_equal(isolated.advance_channels(channels[:, cell:cell+1])[:, 0], whole[:, cell])
    a.reset()
    b.reset()
    np.testing.assert_array_equal(a.state, initial_state(2))
    assert a.step == 0
    np.testing.assert_array_equal(a.advance_channels(np.zeros((1000, 2), dtype=int)),
                                  b.advance(np.zeros((1000, 2))))


def test_channel_validation_and_numerical_failures_are_atomic():
    model = PhotoreceptorMembrane()
    model.advance_channels(np.full((100, 1), 270))
    held, tick = model.state.copy(), model.step
    for invalid in ([], [1], [[1, 2]], [[True]], [['1']], [[1+0j]],
                    [[-1]], [[.5]], [[np.nan]], [[np.inf]], [[2**53]]):
        with pytest.raises(ValueError):
            model.advance_channels(invalid)
        np.testing.assert_array_equal(model.state, held)
        assert model.step == tick
    with pytest.raises(RuntimeError):
        model.advance_channels(np.full((10, 1), 10**10))
    np.testing.assert_array_equal(model.state, held)
    assert model.step == tick


def test_long_channel_train_recovers_without_ion_or_gate_instability():
    model = PhotoreceptorMembrane(3)
    for _ in range(20):
        model.advance_channels(np.tile([0, 270, 2700], (1000, 1)))
    assert model.step == 20000
    assert np.isfinite(model.state).all()
    assert (model.state[:, 6:] > 0).all()
    model.advance_channels(np.zeros((10000, 3), dtype=int))
    assert np.all(model.state[:, 0] < -69)
