from pathlib import Path

import numpy as np
import pytest
from numba import cuda

from nexus.brain.photoreceptor import PhotoreceptorMembrane, initial_state
from nexus.brain.photoreceptor_cuda import CudaPhotoreceptorMembrane

hardware = pytest.mark.skipif(
    not cuda.is_available() or cuda.config.ENABLE_CUDASIM,
    reason='Requires an accessible physical CUDA GPU; CPU tests remain available',
)


def test_missing_cuda_is_reported_without_silent_backend_switch(monkeypatch):
    monkeypatch.setattr(cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA is unavailable'):
        CudaPhotoreceptorMembrane()


@hardware
@pytest.mark.parametrize('cells', [1, 129, 3377])
def test_gpu_channels_match_cpu_for_small_partial_and_full_eye_batches(cells):
    channels = np.zeros((100, cells), dtype=np.int64)
    channels[20:60] = np.resize([27, 270, 2700, 5400, 0], cells)
    cpu, gpu = PhotoreceptorMembrane(cells), CudaPhotoreceptorMembrane(cells)
    expected, actual = cpu.advance_channels(channels), gpu.advance_channels(channels)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
    np.testing.assert_allclose(gpu.state, cpu.state, rtol=1e-9, atol=1e-10)
    assert gpu.step == cpu.step == len(channels)


@hardware
def test_gpu_chunking_reset_and_cell_isolation():
    channels = np.zeros((1000, 3), dtype=int)
    channels[113:651] = [0, 270, 2700]
    a, b = CudaPhotoreceptorMembrane(3), CudaPhotoreceptorMembrane(3)
    whole = a.advance_channels(channels)
    chunks = np.concatenate([b.advance_channels(channels[:113]),
                             b.advance_channels(channels[113:651]),
                             b.advance_channels(channels[651:])])
    np.testing.assert_array_equal(whole, chunks)
    np.testing.assert_array_equal(a.state, b.state)
    isolated = CudaPhotoreceptorMembrane()
    np.testing.assert_array_equal(isolated.advance_channels(channels[:, 2:3])[:, 0], whole[:, 2])
    a.reset()
    np.testing.assert_array_equal(a.state, initial_state(3))
    assert a.step == 0
    np.testing.assert_array_equal(a.advance_channels(channels), whole)


@hardware
def test_gpu_failure_does_not_commit_other_cells_or_time():
    gpu = CudaPhotoreceptorMembrane(129)
    gpu.advance_channels(np.full((10, 129), 270))
    held, step = gpu.state.copy(), gpu.step
    invalid = np.zeros((10, 129), dtype=np.int64)
    invalid[:, -1] = 10**10  # Only the partial-block cell leaves the physical range.
    with pytest.raises(RuntimeError, match='valid state range'):
        gpu.advance_channels(invalid)
    np.testing.assert_array_equal(gpu.state, held)
    assert gpu.step == step
    for bad in (np.full((1, 129), -1), np.full((1, 129), .5), np.full((1, 129), np.nan)):
        with pytest.raises(ValueError):
            gpu.advance_channels(bad)
        np.testing.assert_array_equal(gpu.state, held)
        assert gpu.step == step
    # A device-reported numerical failure must leave the model usable.
    cpu = PhotoreceptorMembrane(129)
    cpu.state[:] = held
    np.testing.assert_allclose(gpu.advance_channels(np.zeros((10, 129), dtype=int)),
                               cpu.advance_channels(np.zeros((10, 129), dtype=int)), rtol=0, atol=1e-7)


@hardware
def test_gpu_pulses_match_independent_source_equation_fixture():
    fixture = Path(__file__).parent/'fixtures/photoreceptor-channels-v1.npz'
    with np.load(fixture, allow_pickle=False) as reference:
        channels = np.zeros((3000, 5), dtype=np.int64)
        channels[1000:2000] = reference['pulse_amplitudes']
        gpu = CudaPhotoreceptorMembrane(5)
        samples, start = [], 0
        for end in reference['sample_indices']+1:
            gpu.advance_channels(channels[start:end])
            samples.append(gpu.state.copy())
            start = end
        np.testing.assert_allclose(np.stack(samples, axis=1), reference['pulse_reference_states'],
                                   rtol=1e-5, atol=3e-4)
