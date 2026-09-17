import copy
import math

import numpy as np
import pytest

from nexus.brain.phototransduction import (
    Phototransduction, SCHEDULE_BLOCK_SIZE, _prepare, _reaction, _until,
)


def scan_until(states, reversals, due, reactions, boundary, rng):
    """Independent exhaustive traversal, with no scheduling index or cached sum."""
    events = 0
    for cell in range(len(states)):
        while due[cell] <= boundary:
            when = due[cell]
            _reaction(states[cell], reactions[cell])
            due[cell], reactions[cell], reversals[cell] = _prepare(states[cell], reversals[cell], when, rng)
            events += 1
    return events


@pytest.mark.parametrize('size', [1, 127, 128, 129, 257])
def test_block_scheduler_matches_full_scan_at_boundaries_and_partial_blocks(size):
    model = Phototransduction(microvilli=size, seed=82300)
    # Force due events on first/last cells and both sides of block boundaries.
    targets = sorted({0, size-1, *[x for x in (127, 128, 255, 256) if x < size]})
    for i, cell in enumerate(targets):
        model.due[cell] = .1*(1+i%3)
        model.reactions[cell] = 0  # Open one TRP channel; pools stay valid.
    states, reversals = model.states.copy(), model.reversals.copy()
    due, reactions = model.due.copy(), model.reactions.copy()
    rng = copy.deepcopy(model.rng)
    blocks = np.array([model.due[i:i+SCHEDULE_BLOCK_SIZE].min()
                       for i in range(0, size, SCHEDULE_BLOCK_SIZE)])
    # A stale earlier bound is allowed (e.g. after photon rescheduling).
    blocks[0] = 0
    for boundary in (0., .1, .2, .3, .5, 1., 10.):
        before = int(model.states[:, 0].sum())
        events, change = _until(model.states, model.reversals, model.due,
                                model.reactions, blocks, boundary, model.rng)
        expected_events = scan_until(states, reversals, due, reactions, boundary, rng)
        assert events == expected_events
        assert change == int(states[:, 0].sum())-before
        for actual, expected in [(model.states, states), (model.reversals, reversals),
                                 (model.due, due), (model.reactions, reactions)]:
            np.testing.assert_array_equal(actual, expected)
        assert model.rng.bit_generator.state == rng.bit_generator.state
        for block, earliest in enumerate(blocks):
            start = block*SCHEDULE_BLOCK_SIZE
            assert earliest <= model.due[start:start+SCHEDULE_BLOCK_SIZE].min()


def test_empty_schedule_does_not_change_state_or_consume_randomness():
    model = Phototransduction(microvilli=129)
    before = copy.deepcopy(model)
    events, change = _until(model.states, model.reversals, model.due, model.reactions,
                            np.array([math.inf, math.inf]), 1000., model.rng)
    assert events == change == 0
    np.testing.assert_array_equal(model.states, before.states)
    assert model.rng.bit_generator.state == before.rng.bit_generator.state


@pytest.mark.parametrize('invalid', [-1., np.nan, np.inf])
def test_scalar_event_checks_still_reject_invalid_molecular_values(invalid):
    model = Phototransduction(microvilli=1)
    state = model.states[0].copy()
    state[5] = invalid
    with pytest.raises(RuntimeError, match='Invalid phototransduction molecular state'):
        _reaction(state, 0)


def test_negative_propensity_still_aborts_an_atomic_advance():
    model = Phototransduction(microvilli=129, seed=82301)
    # Invalid G-pool occupancy yields a negative refractory-pool recovery rate.
    model.states[-1, 7] = 51
    before = copy.deepcopy(model)
    with pytest.raises(RuntimeError, match='Invalid phototransduction reaction propensity'):
        model.advance([0, 1])
    for key in ('states', 'reversals', 'due', 'reactions'):
        np.testing.assert_array_equal(getattr(model, key), getattr(before, key))
    np.testing.assert_array_equal(model.membrane.state, before.membrane.state)
    assert model.rng.bit_generator.state == before.rng.bit_generator.state
    assert model.time_ms == model.events == model.photons == model.membrane.step == 0
