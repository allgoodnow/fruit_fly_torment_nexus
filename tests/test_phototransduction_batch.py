import numpy as np
import pytest
from numba import cuda

from nexus.brain.phototransduction_batch import BatchedPhototransduction
from nexus.brain.phototransduction_parallel import ParallelPhototransduction


@pytest.mark.skipif(not cuda.is_available() or cuda.config.ENABLE_CUDASIM,
                    reason='Requires physical CUDA access')
@pytest.mark.parametrize('membrane_backend', ['cpu', 'cuda'])
def test_batch_preserves_independent_receptor_responses_and_continuation(membrane_backend):
    seeds = [87100, 87101, 87102]
    photons = np.zeros((120, 3), dtype=np.int64)
    photons[20, 1] = 80
    photons[20:70, 2] = 2
    batch = BatchedPhototransduction(seeds, microvilli=129, batch_cells=2,
                                     membrane_backend=membrane_backend)
    whole = BatchedPhototransduction(seeds, microvilli=129, batch_cells=3,
                                     membrane_backend=membrane_backend)
    references = [ParallelPhototransduction(microvilli=129, seed=s) for s in seeds]
    pieces = []
    for counts in (photons[:41], photons[41:]):
        result = batch.advance(counts)
        pieces.append(result)
        for receptor, reference in enumerate(references):
            expected = reference.advance(counts[:, receptor])
            np.testing.assert_array_equal(result['open_channels'][:, receptor], expected['open_channels'])
            np.testing.assert_allclose(result['voltage_mv'][:, receptor], expected['voltage_mv'],
                                       rtol=0, atol=1e-7)
            region = slice(receptor*129, (receptor+1)*129)
            np.testing.assert_array_equal(batch.streams[region], reference.streams)
            np.testing.assert_allclose(batch.states[region], reference.states, atol=1e-9, rtol=1e-9)
            assert batch.events[receptor] == reference.events
            assert batch.rngs[receptor].bit_generator.state == reference.rng.bit_generator.state
    expected = whole.advance(photons)
    for key in expected:
        np.testing.assert_array_equal(np.concatenate([p[key] for p in pieces]), expected[key])
    assert not expected['open_channels'][:, 0].any()
    assert expected['open_channels'][:, 1:].max() > 0
