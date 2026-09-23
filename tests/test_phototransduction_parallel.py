import copy

import numpy as np
import pytest
from numba import cuda

from nexus.brain.phototransduction_parallel import ParallelPhototransduction

backends = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not cuda.is_available() or cuda.config.ENABLE_CUDASIM, reason='Requires physical CUDA access'))]


def identical(a, b):
    for key in ('states', 'due', 'reactions', 'reversals', 'streams'):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    np.testing.assert_array_equal(a.membrane.state, b.membrane.state)
    assert a.rng.bit_generator.state == b.rng.bit_generator.state
    assert (a.time_ms, a.events, a.photons, a.initialized, a.membrane.step) == (
        b.time_ms, b.events, b.photons, b.initialized, b.membrane.step)


@pytest.mark.parametrize('backend', backends)
def test_parallel_chunks_reset_and_darkness_preserve_state(backend):
    photons = np.zeros(130, dtype=int)
    photons[10], photons[53:58], photons[90] = 100, 3, 50
    a = ParallelPhototransduction(microvilli=129, seed=83101, backend=backend)
    b = ParallelPhototransduction(microvilli=129, seed=83101, backend=backend)
    whole = a.advance(photons)
    chunks = [b.advance(photons[:11]), b.advance(photons[11:54]), b.advance(photons[54:])]
    for key in whole:
        np.testing.assert_array_equal(whole[key], np.concatenate([c[key] for c in chunks]))
    identical(a, b)
    assert not whole['open_channels'][:100].any()
    assert whole['open_channels'].max() > 0
    assert whole['open_channels'].max() <= 129*27
    assert np.all(a.states[:, 2]+a.states[:, 4]+a.states[:, 7] <= 50)
    a.reset()
    for key, value in a.advance(photons).items():
        np.testing.assert_array_equal(value, whole[key])
    a.reset()
    assert not a.advance(np.zeros(130, dtype=int))['open_channels'].any()


@pytest.mark.parametrize('backend', backends)
def test_parallel_failed_membrane_or_cascade_commits_nothing(backend, monkeypatch):
    model = ParallelPhototransduction(microvilli=129, seed=83102, backend=backend)
    model.advance([10, 0, 0])
    before = copy.deepcopy(model)
    with monkeypatch.context() as m:
        def fail(_):
            raise RuntimeError('Injected membrane failure')
        m.setattr(model.membrane, 'advance_channels', fail)
        with pytest.raises(RuntimeError, match='Injected'):
            model.advance([0, 50, 0])
    identical(model, before)
    # Compare recovery after failure with an untouched copy, including RNG.
    expected, actual = before.advance([0, 50, 0]), model.advance([0, 50, 0])
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    identical(model, before)
    model.reset()
    model.states[-1, 7] = 51  # Invalid propensity in the final partial block.
    before = copy.deepcopy(model)
    with pytest.raises(RuntimeError, match='molecular update'):
        model.advance([1, 0])
    identical(model, before)


def test_parallel_rejects_invalid_input_and_backend():
    model = ParallelPhototransduction(microvilli=1)
    before = copy.deepcopy(model)
    for value in ([], [True], ['1'], [-1], [.5], [np.inf], [np.nan], [[1]], [1000001]):
        with pytest.raises(ValueError):
            model.advance(value)
        identical(model, before)
    for kwargs in ({'seed': -1}, {'seed': True}, {'seed': 2**64}, {'microvilli': 0},
                   {'microvilli': True}, {'backend': 'automatic'}):
        with pytest.raises(ValueError):
            ParallelPhototransduction(**kwargs)


@pytest.mark.skipif(not cuda.is_available() or cuda.config.ENABLE_CUDASIM, reason='Requires physical CUDA access')
def test_matching_gpu_cpu_streams_and_future_continuation():
    a = ParallelPhototransduction(microvilli=257, seed=83103)
    b = ParallelPhototransduction(microvilli=257, seed=83103, backend='cuda')
    for photons in (np.r_[np.zeros(20, dtype=int), 257, np.zeros(79, dtype=int)],
                    np.full(35, 3), np.zeros(80, dtype=int)):
        left, right = a.advance(photons), b.advance(photons)
        np.testing.assert_array_equal(left['open_channels'], right['open_channels'])
        np.testing.assert_allclose(left['voltage_mv'], right['voltage_mv'], atol=1e-7, rtol=0)
        np.testing.assert_array_equal(a.streams, b.streams)
        np.testing.assert_array_equal(a.reactions, b.reactions)
        np.testing.assert_allclose(a.states, b.states, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(a.due, b.due, rtol=1e-10, atol=1e-8)
        assert a.events == b.events


def test_streams_start_distinct_and_reproduce_after_reset():
    model = ParallelPhototransduction(microvilli=257, seed=83104)
    initial = model.streams.copy()
    assert len(np.unique(initial)) == 257
    model.advance([0])
    assert not np.array_equal(model.streams, initial)
    model.reset()
    np.testing.assert_array_equal(model.streams, initial)
