"""Versioned, source-matched circuit inputs and explicit temperature scenario."""
from functools import lru_cache
import json
import math
from pathlib import Path


@lru_cache(maxsize=1)
def registry():
    return json.loads(Path(__file__).with_name('intervention-circuits.json').read_text())


def circuit_ids(name):
    if name not in ('looming', 'warmth', 'aversion_proxy', 'giant_fiber'):
        raise ValueError(f'Unknown circuit: {name}')
    return list(registry()['circuits'][name]['ids'])


def heat_rate(temperature):
    if isinstance(temperature, (bool, str)):
        raise ValueError('Scenario temperature must be a number from 20 to 100 °C')
    temperature = float(temperature)
    if not math.isfinite(temperature) or not 20 <= temperature <= 100:
        raise ValueError('Scenario temperature must be a number from 20 to 100 °C')
    # An explicit scenario mapping, not a fitted dose-response. Saturates at
    # 40 degrees; 100 does not invent protein damage, membrane failure, or pain.
    return temperature, min(1., max(0., (temperature-25)/15))*300.
