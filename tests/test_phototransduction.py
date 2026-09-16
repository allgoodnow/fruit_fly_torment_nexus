import copy
from pathlib import Path

import numpy as np
import pytest

from nexus.brain.photoreceptor import PhotoreceptorMembrane, SOURCE_COMMIT
from nexus.brain.phototransduction import (
    Phototransduction, _reaction, calcium_update, ghk_currents, reaction_rates,
)

FIXTURE = Path(__file__).parent / 'fixtures/phototransduction-v1.npz'


def test_reaction_and_calcium_kernels_match_pinned_matlab_expressions():
    with np.load(FIXTURE, allow_pickle=False) as reference:
        assert str(reference['source_commit']) == SOURCE_COMMIT
        for i, (y, previous) in enumerate(zip(reference['states'], reference['previous_reversal_v'])):
            np.testing.assert_allclose(reaction_rates(y), reference['rates'][i], rtol=1e-12, atol=1e-11)
            np.testing.assert_allclose(calcium_update(y, previous), reference['calcium_updates'][i], rtol=1e-12, atol=1e-11)
            np.testing.assert_allclose(ghk_currents(y[0], y[1]/6.022/3/100, previous),
                                       reference['ghk_currents'][i], rtol=1e-12, atol=1e-11)


def test_darkness_and_photon_response_have_causal_channel_to_membrane_timing():
    dark = Phototransduction(microvilli=64)
    baseline = dark.advance(np.zeros(180, dtype=int))
    assert not baseline['open_channels'].any()
    membrane = PhotoreceptorMembrane()
    np.testing.assert_array_equal(baseline['voltage_mv'], membrane.advance(np.zeros((1800, 1)))[:, 0])
    light = Phototransduction(microvilli=64)
    photons = np.zeros(180, dtype=int)
    photons[20] = 64
    response = light.advance(photons)
    assert not response['open_channels'][:200].any()
    np.testing.assert_array_equal(response['voltage_mv'][:200], baseline['voltage_mv'][:200])
    assert response['open_channels'].max() > 0
    assert response['voltage_mv'].max() > baseline['voltage_mv'].max()
    assert light.photons == 64 and light.time_ms == 180 and light.membrane.step == 1800
    # Trace handoff uses actual summed channels, without a population multiplier.
    membrane.reset()
    np.testing.assert_array_equal(membrane.advance_channels(response['open_channels'][:, None])[:, 0],
                                  response['voltage_mv'])


def test_chunked_replay_preserves_pending_reactions_rng_membrane_and_reset():
    photons = np.zeros(180, dtype=int)
    photons[10] = 30
    photons[61:65] = 4
    a, b = Phototransduction(microvilli=64), Phototransduction(microvilli=64)
    whole = a.advance(photons)
    chunks = [b.advance(photons[:11]), b.advance(photons[11:73]), b.advance(photons[73:])]
    for key in whole:
        np.testing.assert_array_equal(whole[key], np.concatenate([chunk[key] for chunk in chunks]))
    for key in ('states', 'reversals', 'due', 'reactions'):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    np.testing.assert_array_equal(a.membrane.state, b.membrane.state)
    assert a.rng.bit_generator.state == b.rng.bit_generator.state
    assert (a.photons, a.events, a.time_ms) == (b.photons, b.events, b.time_ms)
    a.reset()
    assert a.photons == a.events == a.time_ms == 0
    assert not a.initialized and a.membrane.step == 0
    for key, value in a.advance(photons).items():
        np.testing.assert_array_equal(value, whole[key])


def test_reactions_conserve_g_pool_and_do_not_open_more_than_27_channels():
    y = np.array([26., 1., 3., 1., 2., 40., 0., 20.])
    old_total = y[2]+y[4]+y[7]
    _reaction(y, 2)  # G activation transfers one available G to active G.
    assert y[2]+y[4]+y[7] == old_total and y[2] == 4 and y[7] == 19
    _reaction(y, 4)  # PLC activation transfers one active G to bound G.
    assert y[2]+y[4]+y[7] == old_total and y[4] == 3 and y[2] == 3
    _reaction(y, 0)
    assert y[0] == 27 and reaction_rates(y)[0, 0] == 0
    receptor = Phototransduction(microvilli=64)
    response = receptor.advance(np.full(300, 4))
    assert (response['open_channels'] <= 64*27).all()
    assert (receptor.states[:, 2]+receptor.states[:, 4]+receptor.states[:, 7] <= 50).all()
    assert (receptor.states >= 0).all()
    assert (receptor.states[:, 0] <= 27).all()


def test_calcium_feedback_increases_source_shutoff_propensities():
    y = np.array([10., 100., 3., 1., 2., 40., 0., 20.])
    before = reaction_rates(y)
    y[6] = 300
    after = reaction_rates(y)
    for index in (0, 3, 4, 5):
        assert after[index, 1] > before[index, 1]


def test_invalid_inputs_and_failed_membrane_handoff_leave_no_partial_advance(monkeypatch):
    receptor = Phototransduction(microvilli=8)
    receptor.advance([1, 0, 0])
    before = copy.deepcopy(receptor)

    def unchanged():
        for key in ('states', 'reversals', 'due', 'reactions'):
            np.testing.assert_array_equal(getattr(receptor, key), getattr(before, key))
        np.testing.assert_array_equal(receptor.membrane.state, before.membrane.state)
        assert receptor.rng.bit_generator.state == before.rng.bit_generator.state
        assert (receptor.time_ms, receptor.photons, receptor.events, receptor.initialized) == (
            before.time_ms, before.photons, before.events, before.initialized)
    for invalid in ([], [[1]], [-1], [True], ['2'], [.5], [np.nan], [np.inf], [1000001]):
        with pytest.raises(ValueError):
            receptor.advance(invalid)
        unchanged()

    def fail(_):
        raise RuntimeError('Injected membrane failure')
    monkeypatch.setattr(receptor.membrane, 'advance_channels', fail)
    with pytest.raises(RuntimeError, match='Injected'):
        receptor.advance([1, 0, 0])
    unchanged()
    for n in (0, -1, 30001, True, 1.5):
        with pytest.raises(ValueError):
            Phototransduction(microvilli=n)
