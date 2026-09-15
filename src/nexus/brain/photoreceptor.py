"""Research-only R1–R6 membrane model; not connected to the application's eyes.

Python adaptation of Zhuoyi Song's June 2017 ``wt_cc_model_pump.m``,
``NaCaPump_Body.m`` and ``NaKPump.m`` for Juusola et al., eLife 6:e26117.
Upstream: JuusolaLab/Microsaccadic_Sampling_Paper, a4453f7e47abf2ea2a924c376d4c1c157c6011e2.
SPDX-License-Identifier: GPL-3.0-only
See licenses/photoreceptor/ and docs/photoreceptor-membrane.md.

The published BG1 parameter set and equations are retained. A fixed-step RK4
solver replaces ode45; removable rate singularities use their analytic limit.
Inputs are currents in
nA, NOT RGB brightness, photons, or spikes. There is no threshold/reset or
invented conversion to histamine release. This is the membrane subsystem only.
"""
import math
import sys

import numpy as np
from numba import njit

SOURCE_COMMIT = 'a4453f7e47abf2ea2a924c376d4c1c157c6011e2'
STATE_NAMES = ('voltage_mv', 'shab_inactivation', 'shab_activation',
               'shaker_activation', 'shaker_inactivation', 'novel_k_activation',
               'sodium_mmol_l', 'potassium_mmol_l', 'calcium_mmol_l')
DT_MS = .1


@njit(cache=not getattr(sys, 'frozen', False))
def _ratio(x, scale):
    """x / (exp(x/scale)-1), including its removable singularity."""
    z = x / scale
    if abs(z) < 1e-6:
        return scale * (1. - z / 2. + z*z / 12.)
    return x / math.expm1(z)


@njit(cache=not getattr(sys, 'frozen', False))
def derivative(y, current_na):
    """Published BG1 equations; derivatives per ms, currents inward-positive."""
    v, h, n, m, ha, nk, nai, ki, cai = y
    # Voltage-dependent steady states and kinetics from the source HH model.
    h_inf = 1. / (1. + math.exp((-25.7-v) / -6.4))
    n_inf = (1. / (1. + math.exp((-1.-v) / 9.1))) ** .5
    m_probability = 1. / (1. + math.exp((-23.7-v) / 12.8))
    m_inf = m_probability ** (1./3.)
    ha_first = .8 / (1. + math.exp((-55.3-v) / -3.9))
    ha_second = .2 / (1. + math.exp((-74.8-v) / -10.7))
    ha_inf = ha_first + ha_second
    nk_inf = 1. / (1. + math.exp((-14.-v) / 10.6))
    h_rate = 1.35 / 1200.
    n_rate = 1.35 * (.116258 * math.exp((-v-25.6551)/32.1933)
                     + .00659219 * _ratio(-v-23.8032, 1.34548))
    m_rate = 1.35 * (.008174 * math.exp((-v+1.61882)/24.6538)
                     + .058139 * _ratio(-v-59.639, 4.50122))
    ha_rate = 1.35 * (.230299 * math.exp((-v-192.973)/31.31961)
                      + .0437316 * _ratio(-v+13.4859, 11.11))
    nk_rate = 1.35 / (13. + 6232. / (30. * math.sqrt(math.pi/2.))
                     * math.exp(-2. * ((v+19.4)/30.)**2))
    # The published pump routines receive mV, despite their header saying volts.
    vo = v * .001
    q = vo * 96485. / (2. * 8.314472 * 293.)
    numerator = math.exp(.65*q) * nai**3 * 1.5 - math.exp(-.35*q) * 120.**3 * cai
    denominator = (87.5**3 + 120.**3) * (1.38+1.5) * (1. + .001*math.exp(-.35*q))
    inaca = -1200. * numerator / denominator
    ica = (2000. * 4. * 1.57e-5) * (1.5-cai) / (.5 + 1.5-cai)
    inak = (-3.7 / 3. / 5.1) * nai / (nai+33.)
    # BG1, as selected in Vol_FeedbackCluster.m (not the stale inline defaults).
    cm, vl, gl, vk, gl2 = 4., -57.1, 6.*.585e-3, -85., .4*.85e-2
    gks = n*n*h*3e-3
    gka = m**3 * (ha*.8e-3 + .087e-3)
    gnew = nk*.11e-3
    gcl = (5.*.585e-4) / (1. + math.exp((vo+.0986)/.0045))
    potassium_current = (gnew+gka+gks+gl2) * (v-vk) * 1.57e-5 * 1e6
    total = gka+gks+gl+gl2+gnew+gcl
    reversal_sum = (gnew+gka+gks+gl2)*vk + (gl+gcl)*vl
    dy = np.empty(9)
    dy[0] = (-v*total + reversal_sum)/(.001*cm) + (current_na+inaca+inak+ica)/(1000.*cm*1.57e-5)
    dy[1] = (h_inf-h)*h_rate
    dy[2] = (n_inf-n)*n_rate
    dy[3] = (m_inf-m)*m_rate
    dy[4] = (ha_inf-ha)*ha_rate
    dy[5] = (nk_inf-nk)*nk_rate
    scale = 1000. / 2.92 / 96485.
    dy[6] = (current_na*.2054 + 3.*inaca + 3.*inak)*scale
    dy[7] = (current_na*.2401 - 2.*inak - potassium_current)*scale
    dy[8] = (current_na*.41 - 2.*inaca + ica)*scale/2.
    return dy


def initial_state(cells=1):
    if isinstance(cells, bool) or not isinstance(cells, (int, np.integer)) or cells < 1:
        raise ValueError('Cell count must be a positive integer')
    v = -70.
    row = [v, 1./(1.+math.exp((-25.7-v)/-6.4)),
           (1./(1.+math.exp((-1.-v)/9.1)))**.5,
           (1./(1.+math.exp((-23.7-v)/12.8)))**(1./3.),
           .8/(1.+math.exp((-55.3-v)/-3.9))+.2/(1.+math.exp((-74.8-v)/-10.7)),
           1./(1.+math.exp((-14.-v)/10.6)), 8., 140., .00016]
    return np.tile(row, (cells, 1))


@njit(cache=not getattr(sys, 'frozen', False))
def _integrate(state, currents, dt_ms):
    voltage = np.empty(currents.shape)
    for tick in range(len(currents)):
        for cell in range(len(state)):
            y = state[cell]
            current = currents[tick, cell]
            k1 = derivative(y, current)
            k2 = derivative(y + .5*dt_ms*k1, current)
            k3 = derivative(y + .5*dt_ms*k2, current)
            k4 = derivative(y + dt_ms*k3, current)
            state[cell] = y + dt_ms/6. * (k1+2.*k2+2.*k3+k4)
            for j in range(9):
                value = state[cell, j]
                if (not math.isfinite(value) or (1 <= j <= 5 and not 0 <= value <= 1)
                        or (j >= 6 and value <= 0)):
                    raise RuntimeError('Photoreceptor integration left the valid state range')
            voltage[tick, cell] = state[cell, 0]
    return voltage


class PhotoreceptorMembrane:
    """Persistent, independent BG1 cells driven by physical current traces.

    Current is held over each 0.1 ms interval; returned voltage is at interval
    end. reset() restores the published initial condition, not an equilibrated
    dark state. advance() commits only finite states with physical gate and ion
    ranges; failure leaves the preceding state and clock intact.
    """
    def __init__(self, cells=1):
        self.state = initial_state(cells)
        self.step = 0

    def reset(self):
        self.state[:] = initial_state(len(self.state))
        self.step = 0

    def advance(self, current_na):
        currents = np.asarray(current_na, dtype=np.float64)
        if (currents.ndim != 2 or currents.shape[1] != len(self.state) or len(currents) == 0
                or not np.isfinite(currents).all()):
            raise ValueError('Expected finite current samples with shape (ticks, cells), in nA')
        candidate = self.state.copy()
        try:
            voltage = _integrate(candidate, np.ascontiguousarray(currents), DT_MS)
        except (OverflowError, ZeroDivisionError) as error:
            raise RuntimeError('Photoreceptor integration left the valid state range') from error
        self.state[:] = candidate
        self.step += len(currents)
        return voltage
