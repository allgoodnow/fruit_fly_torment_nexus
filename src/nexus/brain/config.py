"""Small, specimen-specific UI configuration; graph validation stays in the worker."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

from .targets import SUGAR, MN9
from .motor import DNA02_LEFT, DNA02_RIGHT


@dataclass(frozen=True)
class ModelConfig:
    directory: Path
    snapshot: str
    title: str
    experimental: bool
    first_label: str
    first_ids: tuple[str, ...]
    mn9_ids: tuple[str, ...]
    left_ids: tuple[str, ...]
    right_ids: tuple[str, ...]
    food_available: bool


def default_pack(dataset='male-cns'):
    base = Path(sys._MEIPASS) if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[3]
    paths = {'male-cns': 'brain-male-cns-v1.0-lif', 'flywire-v630': 'brain-v630'}
    return base/'data'/paths[dataset]


def model_config(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest.get('format') != 'nexus-connectome-1':
        raise ValueError('A prepared neural runtime pack is required')
    if manifest['snapshot'] == '630':
        return ModelConfig(directory, '630', 'FlyWire v630', False, 'Sugar sensory cells (21)',
                           tuple(SUGAR), (MN9,), (DNA02_LEFT,), (DNA02_RIGHT,), True)
    if manifest['snapshot'] != 'male-cns:v1.0' or not manifest.get('experimental'):
        raise ValueError('Unsupported neural dataset')
    contents = (directory/'circuits.json').read_bytes()
    if hashlib.sha256(contents).hexdigest() != manifest['files']['circuits.json']:
        raise ValueError('Circuit registry checksum mismatch')
    registry = json.loads(contents)
    if registry['snapshot'] != manifest['snapshot']:
        raise ValueError('Circuit registry belongs to another dataset')
    readouts = registry['readouts']
    return ModelConfig(directory, manifest['snapshot'], 'MaleCNS v1.0 · experimental', True,
                       f"Warmth sensory cells ({len(registry['circuits']['warmth']['ids'])})", tuple(registry['circuits']['warmth']['ids']),
                       tuple(readouts['mn9']), tuple(readouts['steering_left']),
                       tuple(readouts['steering_right']), False)
