"""Research phototransduction reaction network, adapted from Song/Juusola 2017.

SPDX-License-Identifier: GPL-3.0-only
Source and numerical departures: docs/phototransduction.md and
licenses/photoreceptor/NOTICE.md. This is not the application's vision pathway.

The reaction propensities, GHK currents and fast calcium update are taken from
the pinned MATLAB code. Event timing is an explicit new variant: unrounded
exponential waits, no empirical latency offset or protected Rh* removal. It
must not be described as an exact reproduction of the published simulator.
The direct Octave audit found larger channel responses in this variant. A
follow-up isolated substantial suppression from upstream time rounding and
stale input rates; see docs/phototransduction-timing-ablation.md. These checks
do not establish physiological accuracy or complete simulator equivalence.
"""
import copy
import math
import sys

import numpy as np
from numba import njit

from .photoreceptor import PhotoreceptorMembrane

# Initial_PR_Goodaa1new.m, aa=1; per-ms reaction units used by the source.
PARAMETERS = np.array([
    .3*6.022*3*100, .18*6.022*3*100, 2., 3., .0037, 40.,
    .0047*.3*5, .0035, .048*3, 11.1, .0039*4, 100., 2., 1.3,
    .004, 37.8, 0., .15, 100., 10., 11.5, .025, .03, .0055,
    1., 27., .003,
])
PARAMETERS.setflags(write=False)
STATE_NAMES = ('open_trp', 'calcium_scaled', 'active_g', 'active_rhodopsin',
               'active_plc', 'messenger_a', 'bound_calmodulin', 'available_g')
# Scheduling index only; this does not group or scale biological states.
SCHEDULE_BLOCK_SIZE = 128


@njit(cache=not getattr(sys, 'frozen', False))
def ghk_currents(open_trp, calcium_mm, previous_reversal_v):
    """Pinned TRP_Rev_GHK_online: total/Ca/Na/Mg/K in pA, reversal in V."""
    beta, vm = 96485./8.314/293., -.07
    area = math.pi*1.5*.06*1e-8
    concentrations_in = (calcium_mm, 8., 3., 140.)
    concentrations_out = (1.5, 120., 4., 5.)
    valences = (2., 1., 2., 1.)
    weights = (.85, .02, .11, .02)
    flux = np.empty(4)
    epsilon = 0.
    for i in range(4):
        z = valences[i]
        exp_term = math.exp(-z*vm*beta)
        flux[i] = z*beta*vm*(concentrations_in[i]-concentrations_out[i]*exp_term)/(1.-exp_term)
        epsilon += z*weights[i]*flux[i]
    epsilon *= area*96485.*1e12
    permeability = (open_trp+.0001)*8.*(previous_reversal_v+.07)/epsilon
    currents = np.empty(4)
    for i in range(4):
        currents[i] = weights[i]*permeability*flux[i]*96485.*valences[i]*area*1e12
    reversal = 8.314*293./96485.*math.log(
        (.02*120.+.85*1.5+.11*4.+.02*5.) / (.02*8.+.85*calcium_mm+.11*3.+.02*140.))
    return currents.sum(), currents[0], currents[1], currents[2], currents[3], reversal


@njit(cache=not getattr(sys, 'frozen', False))
def reaction_rates(y):
    """Source up/down propensities at a state; no state mutation or clipping."""
    p = PARAMETERS
    fbp = (y[1]/p[0])**p[2] / (1.+(y[1]/p[0])**p[2])
    fbn = 50.*(y[6]/p[1])**p[3] / (1.+(y[6]/p[1])**p[3])
    z = np.zeros((8, 2))
    z[0, 0] = p[17]*(y[5]/p[18])**p[12]*(1.+p[20]*fbp)*(p[25]-y[0])
    z[0, 1] = y[0]*p[21]*(1.+p[19]*fbn)
    z[2, 0] = p[6]*y[7]*y[3]
    z[2, 1] = p[26]*y[2]*y[4]
    z[7, 0] = p[7]*(50.-y[4]-y[2]-y[7])
    z[3, 1] = y[3]*(1.+p[5]*fbn)*p[4]
    z[4, 0] = p[10]*(p[11]-y[4])*y[2]
    z[4, 1] = p[8]*y[4]*(1.+p[9]*fbn)
    z[5, 0] = p[13]*y[4]*(1.+p[16]*fbp)
    z[5, 1] = p[14]*y[5]*(1.+p[15]*fbn)
    occupancy = y[6]/.5/6.022/3./100.
    z[6, 0] = p[22]*y[1]*(1.-occupancy)
    z[6, 1] = p[23]*y[6]
    return z


@njit(cache=not getattr(sys, 'frozen', False))
def calcium_update(y, previous_reversal_v):
    """Source fast-calcium closure at fixed -70 mV, once per event preparation.

    This retains the source's voltage-clamped cascade. Whole-cell voltage
    feedback is applied later to the summed channel count, not fed into GHK.
    """
    currents = ghk_currents(y[0], y[1]/6.022/3./100., previous_reversal_v)
    occupancy = y[6]/.5/6.022/3./100.
    calcium = 6.022*3.*100.*(
        currents[1]*1e9/1.002/2./96485./3./1000. + 2.*.0055*occupancy + 2e-4
    ) / (1. + 2.*.03*(1.-occupancy) + 7.24)
    return calcium, currents[5]


@njit(cache=not getattr(sys, 'frozen', False))
def _prepare(y, previous_reversal_v, time_ms, rng):
    # Source ordering: compute rates, then update the fast calcium variable.
    rates = reaction_rates(y)
    total = rates.sum()
    # Validate scalar entries without allocating two temporary boolean arrays
    # for every molecular event. Keep the rate calculation and summation order.
    for index in range(8):
        for direction in range(2):
            if not math.isfinite(rates[index, direction]) or rates[index, direction] < 0:
                raise RuntimeError('Invalid phototransduction reaction propensity')
    calcium, reversal = calcium_update(y, previous_reversal_v)
    y[1] = calcium
    if not math.isfinite(calcium) or calcium <= 0 or not math.isfinite(reversal):
        raise RuntimeError('Invalid phototransduction calcium state')
    if total == 0:
        return math.inf, -1, reversal
    # Continuous event timing; unlike source, no .001 uniform floor, .2/ms
    # offset, rounded waiting times or protected first Rh* deactivation.
    wait = rng.exponential(1./total)
    event_time = time_ms + wait
    if event_time <= time_ms:
        raise RuntimeError('Phototransduction event time lost numerical precision')
    target, cumulative = rng.random()*total, 0.
    for direction in range(2):
        for index in range(8):
            cumulative += rates[index, direction]
            if target < cumulative:
                return event_time, direction*8+index, reversal
    raise RuntimeError('Could not select phototransduction reaction')


@njit(cache=not getattr(sys, 'frozen', False))
def _reaction(y, reaction):
    direction, index = reaction//8, reaction % 8
    y[index] += 1 if direction == 0 else -1
    if direction == 0 and index == 4:
        y[2] -= 1  # Activating PLC consumes active G.
    if direction == 0 and index == 2:
        y[7] -= 1  # Activating G consumes available G.
    for index in range(8):
        if y[index] < 0 or not math.isfinite(y[index]):
            raise RuntimeError('Invalid phototransduction molecular state')
    if y[0] > 27 or y[4]+y[2]+y[7] > 50:
        raise RuntimeError('Invalid phototransduction molecular state')


@njit(cache=not getattr(sys, 'frozen', False))
def _until(states, reversals, due, reactions, block_due, boundary, rng):
    events = 0
    channel_delta = 0
    for block in range(len(block_due)):
        if block_due[block] > boundary:
            continue
        earliest = math.inf
        # Preserve the original ascending cell order and within-cell event
        # order. A global time-sorted queue would change seeded RNG assignment.
        start = block*SCHEDULE_BLOCK_SIZE
        for cell in range(start, min(start+SCHEDULE_BLOCK_SIZE, len(states))):
            while due[cell] <= boundary:
                when = due[cell]
                before = int(states[cell, 0])
                _reaction(states[cell], reactions[cell])
                channel_delta += int(states[cell, 0])-before
                due[cell], reactions[cell], reversals[cell] = _prepare(states[cell], reversals[cell], when, rng)
                events += 1
                if events > 1000000:
                    raise RuntimeError('Phototransduction work limit exceeded in a sample')
            earliest = min(earliest, due[cell])
        block_due[block] = earliest
    return events, channel_delta


@njit(cache=not getattr(sys, 'frozen', False))
def _run(states, reversals, due, reactions, photons, start_ms, rng, initialized):
    counts = np.empty((len(photons)*10, 1), dtype=np.int64)
    events = 0
    if not initialized:
        for cell in range(len(states)):
            due[cell], reactions[cell], reversals[cell] = _prepare(states[cell], reversals[cell], start_ms, rng)
    # Rebuild this transient index for every atomic advance; no extra mutable
    # checkpoint state is needed for rollback, reset or chunked input.
    block_due = np.full((len(states)+SCHEDULE_BLOCK_SIZE-1)//SCHEDULE_BLOCK_SIZE, math.inf)
    for cell in range(len(states)):
        block = cell//SCHEDULE_BLOCK_SIZE
        block_due[block] = min(block_due[block], due[cell])
    open_channels = int(states[:, 0].sum())
    for frame in range(len(photons)):
        boundary = start_ms+frame
        completed, change = _until(states, reversals, due, reactions, block_due, boundary, rng)
        events += completed
        open_channels += change
        # Counts are absorbed photons at each 1 ms boundary, as in the source.
        # Uniform allocation is equivalent to its multinomial distribution.
        if photons[frame]:
            arrivals = np.zeros(len(states), dtype=np.int64)
            for _ in range(photons[frame]):
                arrivals[rng.integers(0, len(states))] += 1
            for cell in range(len(states)):
                if arrivals[cell]:
                    states[cell, 3] += arrivals[cell]
                    # Input changes rates. Resample pending reaction and wait;
                    # exponential waiting times are memoryless.
                    due[cell], reactions[cell], reversals[cell] = _prepare(states[cell], reversals[cell], boundary, rng)
                    block = cell//SCHEDULE_BLOCK_SIZE
                    # A rescheduled event can become earlier OR later. Keeping
                    # a stale lower bound is safe; the next scan refreshes it.
                    block_due[block] = min(block_due[block], due[cell])
        for sample in range(10):
            now = (boundary*10+sample)/10.
            completed, change = _until(states, reversals, due, reactions, block_due, now, rng)
            events += completed
            open_channels += change
            # Left-edge channel samples avoid applying future openings early.
            counts[frame*10+sample, 0] = open_channels
        completed, change = _until(states, reversals, due, reactions, block_due, boundary+1, rng)
        events += completed
        open_channels += change
    return counts, events


class Phototransduction:
    """Offline receptor with explicit microvilli, seeded events and BG1 membrane.

    advance() takes a 1-D integer trace of absorbed photons per 1 ms sample.
    These are counts, not rates or pixel intensities. Default population is the
    source's 30,000 microvilli; smaller populations are not scaled up.
    Failed advances preserve molecular, membrane, pending-event and RNG state.
    """
    def __init__(self, *, microvilli=30000, seed=73100):
        if (isinstance(microvilli, bool) or not isinstance(microvilli, (int, np.integer))
                or not 1 <= microvilli <= 30000):
            raise ValueError('Choose 1 to 30000 explicit microvilli')
        self.microvilli, self.seed = int(microvilli), seed
        self.membrane = PhotoreceptorMembrane()
        self.reset()

    def reset(self):
        self.states = np.tile([0., 1., 0., 0., 0., 0., 0., 50.], (self.microvilli, 1))
        self.reversals = np.full(self.microvilli, .009)
        self.due = np.full(self.microvilli, math.inf)
        self.reactions = np.full(self.microvilli, -1, dtype=np.int64)
        self.rng = np.random.default_rng(self.seed)
        self.membrane.reset()
        self.time_ms = self.photons = self.events = 0
        self.initialized = False

    def advance(self, absorbed_photons):
        raw = np.asarray(absorbed_photons)
        if (raw.ndim != 1 or raw.dtype.kind not in 'iuf' or len(raw) == 0
                or not np.isfinite(raw).all() or (raw < 0).any()
                or (raw > 1000000).any() or (raw != np.floor(raw)).any()):
            raise ValueError('Expected integer absorbed photon counts per ms, from 0 to 1000000')
        photons = raw.astype(np.int64)
        states, reversals = self.states.copy(), self.reversals.copy()
        due, reactions = self.due.copy(), self.reactions.copy()
        rng = copy.deepcopy(self.rng)
        channels, events = _run(states, reversals, due, reactions, photons,
                                self.time_ms, rng, self.initialized)
        voltage = self.membrane.advance_channels(channels)
        self.states, self.reversals = states, reversals
        self.due, self.reactions, self.rng = due, reactions, rng
        self.time_ms += len(photons)
        self.photons += int(photons.sum())
        self.events += events
        self.initialized = True
        return {'voltage_mv': voltage[:, 0], 'open_channels': channels[:, 0]}
