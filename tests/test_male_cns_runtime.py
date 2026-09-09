"""Model policy, specimen-bound circuits, and bilateral neural readout checks."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from nexus.brain.runtime import Brain, Connectome
from nexus.brain.targets import readout_ids, SUGAR
from nexus.brain.motor import SteeringDecoder, DNA02
from nexus.brain.motor_effects import MotorEffects
from nexus.brain.telemetry import NeuralTelemetry
from nexus.datasets.male_cns import DATASET, FILES, digest
from nexus.datasets.male_cns_runtime import prepare_runtime, transmitter_signs


@pytest.fixture
def pack(tmp_path):
    raw, structural, output = [tmp_path/p for p in ['raw', 'structure', 'runtime']]
    raw.mkdir()
    structural.mkdir()
    ids = np.arange(101, 111, dtype=np.int64)
    types = ['LPLC2', 'TRN_VP2', 'GNG121', 'GNG121', 'DNp01', 'DNp01', 'DNa02', 'DNa02', 'MN9', 'MN9']
    rows = []
    for index, (body, cell_type) in enumerate(zip(ids, types)):
        rows.append({'bodyId': body, 'type': cell_type, 'superclass': 'cb_intrinsic',
                     'flywireType': 'CB0059' if cell_type == 'GNG121' else cell_type,
                     'somaSide': 'L' if index % 2 == 0 else 'R',
                     'rootSide': None, 'mancBodyid': None, 'mancType': None})
    pd.DataFrame(rows).to_feather(raw/FILES['annotations'])
    labels = ['acetylcholine', 'acetylcholine', 'gaba', 'gaba', 'acetylcholine', 'unclear',
              'acetylcholine', 'acetylcholine', 'glutamate', 'histamine']
    # Deliberately shuffled source rows verify alignment by body ID.
    pd.DataFrame({'body': ids, 'consensus_nt': labels}).iloc[::-1].to_feather(raw/FILES['neurotransmitters'])
    arrays = {'ids': ids, 'offsets': np.arange(11, dtype=np.int64),
              'posts': np.roll(np.arange(10, dtype=np.uint32), -1),
              'contacts': np.arange(1, 11, dtype=np.uint32)}
    for name, value in arrays.items():
        np.save(structural/f'{name}.npy', value, allow_pickle=False)
    source = {'format': 'nexus-structural-connectome-1', 'dataset': DATASET,
              'sources': {'files': [{'file': FILES[name], 'sha256': digest(raw/FILES[name])}
                                    for name in ['annotations', 'neurotransmitters']]},
              'files': {f'{name}.npy': {'sha256': digest(structural/f'{name}.npy')} for name in arrays}}
    (structural/'manifest.json').write_text(json.dumps(source))
    prepare_runtime(structural, raw, output)
    return output


def test_weight_conversion_alignment_omissions_and_explicit_opt_in(pack):
    with pytest.raises(ValueError, match='explicit opt-in'):
        Connectome.load(pack)
    graph = Connectome.load(pack, allow_experimental=True)
    np.testing.assert_allclose(graph.weights, np.arange(1, 11)*[1, 1, -1, -1, 1, 0, 1, 1, -1, 0]*.275)
    assert graph.model['transmitter_coverage']['unclear']['outgoing_edges'] == 1
    assert graph.model['transmitter_coverage']['histamine']['assigned_sign'] == 0
    assert graph.snapshot == DATASET


def test_unknown_and_missing_transmitters_are_distinct():
    table = pd.DataFrame({'body': [3, 1], 'consensus_nt': ['gaba', 'acetylcholine']})
    signs, labels = transmitter_signs(np.array([1, 2, 3]), table)
    assert signs.tolist() == [1, 0, -1] and labels.tolist() == ['acetylcholine', 'missing', 'gaba']
    with pytest.raises(ValueError, match='Unrecognized'):
        transmitter_signs(np.array([1]), pd.DataFrame({'body': [1], 'consensus_nt': ['new_label']}))
    with pytest.raises(ValueError, match='Duplicate'):
        transmitter_signs(np.array([1, 3]), pd.concat([table, table]))


def test_circuit_release_and_rejected_old_ids_preserve_state(pack):
    brain = Brain(Connectome.load(pack, allow_experimental=True))
    brain.set_circuit_input('aversion_proxy', 200)
    assert brain.circuit_ids('aversion_proxy') == ['103', '104']
    brain.set_heat(100)
    assert set(brain.graph.ids[brain.inputs]) == {102, 103, 104}
    for stale in [SUGAR, DNA02]:
        with pytest.raises(ValueError, match='Unknown or invalid'):
            brain.stimulate(stale, 200)
    with pytest.raises(ValueError, match='sugar-cell mapping'):
        readout_ids('sugar', brain.graph)
    brain.advance(.01)
    saved = brain.v.copy(), brain.g.copy(), brain.counts.copy(), copy.deepcopy(brain.rng.bit_generator.state)
    brain.release()
    for a, b in zip(saved[:3], [brain.v, brain.g, brain.counts]):
        np.testing.assert_array_equal(a, b)
    assert saved[3] == brain.rng.bit_generator.state and brain.time == .01


def test_bilateral_mn9_and_motor_channels_use_male_ids(pack):
    brain = Brain(Connectome.load(pack, allow_experimental=True))
    decoder, effects = SteeringDecoder(brain), MotorEffects(brain)
    assert brain.graph.ids[decoder.indices].tolist() == [107, 108]
    assert brain.graph.ids[effects.gf].tolist() == [105, 106]
    brain.counts[-2:] = [4, 7]
    brain.v[-2:] = [-51, -49]
    packet = NeuralTelemetry(brain).snapshot(brain, False, 0)
    assert packet['dataset'] == DATASET
    assert packet['mn9_spikes'] == 11 and packet['mn9_mv'] == -50
    assert [r['id'] for r in packet['mn9_cells']] == ['109', '110']
    assert [r['spikes'] for r in packet['mn9_cells']] == [4, 7]
    decoder.observe([10, 0], .01, [1, 1])
    assert decoder.output(1)['left_drive'] < decoder.output(1)['right_drive']


@pytest.mark.parametrize('mutation', ['hash', 'snapshot', 'foreign_id'])
def test_registry_cannot_silently_cross_specimens(pack, mutation):
    path = pack/'circuits.json'
    registry = json.loads(path.read_text())
    if mutation == 'snapshot':
        registry['snapshot'] = '630'
    else:
        registry['circuits']['aversion_proxy']['ids'] = [SUGAR[0]]
    path.write_text(json.dumps(registry))
    if mutation != 'hash':
        manifest = json.loads((pack/'manifest.json').read_text())
        manifest['files']['circuits.json'] = digest(path)
        (pack/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match={'hash': 'checksum', 'snapshot': 'dataset mismatch',
                                        'foreign_id': 'missing or duplicated'}[mutation]):
        Connectome.load(pack, allow_experimental=True)
