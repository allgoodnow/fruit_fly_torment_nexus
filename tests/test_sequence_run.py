import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nexus.brain.protocol import Protocol
from nexus.brain.scenarios import sequence_protocol
from nexus.coupled import CoupledSession
from nexus.sequence_run import record_sequence
from test_coupled import ClockBody, small_brain
from nexus.brain.motor import DNA02


class RecordingBody(ClockBody):
    def __init__(self):
        super().__init__()
        self.sim = SimpleNamespace(mj_data=SimpleNamespace(qpos=np.zeros(3), qvel=np.zeros(3),
                                                          xmat=np.eye(3).reshape(1, 9)))
        self.thorax_id = 0
        self.motor_offset_rms = 0.

    def position(self):
        return np.array([self.time, 0., 1.])


def test_joined_stages_keep_fractional_boundaries_and_explicit_dataset():
    plan = sequence_protocol([{'name': 'aversion', 'baseline_ms': 1.2, 'stimulus_ms': 3.7, 'recovery_ms': 2.3},
                              {'name': 'heat', 'baseline_ms': 2.1, 'stimulus_ms': 4.2, 'recovery_ms': 1.3}],
                             dataset='male-cns:v1.0', pain_circuit='nociception_proxy')
    assert plan['duration_ms'] == 14.8
    assert [e['at_ms'] for e in plan['events']] == [0., 1.2, 4.9, 7.2, 9.3, 13.5]
    assert plan['events'][1]['name'] == 'nociception_proxy'
    for stages in [[], [{}]*51, [{'name': 'heat', 'dataset': '630'}], [{'name': 'heat', 'baseline_ms': 60000,
                    'stimulus_ms': 60000, 'recovery_ms': 60000}]*4]:
        with pytest.raises(ValueError):
            sequence_protocol(stages, dataset='630')


def test_recorder_preserves_existing_state_and_exact_non_round_endpoint(tmp_path):
    a, b = [CoupledSession(small_brain(), RecordingBody(), autonomous=True) for _ in range(2)]
    for session in (a, b):
        session.command('stimulate', {'ids': [DNA02[0]], 'rate_hz': 200})
        session.advance(1000)
    starting = a.brain.counts.copy()
    protocol = {'format': 'nexus-protocol-1', 'dataset': '630', 'duration_ms': 23.1,
                'events': [{'at_ms': 1.2, 'action': 'release'},
                           {'at_ms': 11.3, 'action': 'stimulate', 'ids': [DNA02[1]], 'rate_hz': 200},
                           {'at_ms': 19.7, 'action': 'release'}]}
    result = record_sequence(a, protocol, tmp_path/'run')
    b.command('resume_after_protocol', False)
    b.command('protocol', protocol)
    while b.running:
        b.advance(100)
    np.testing.assert_array_equal(a.brain.counts, b.brain.counts)
    np.testing.assert_array_equal(a.brain.v, b.brain.v)
    assert a.brain.rng.bit_generator.state == b.brain.rng.bit_generator.state
    assert a.generation == 0 and a.brain.time == pytest.approx(.1231)
    assert result['success'] and result['elapsed_ms'] == 23.1 and result['samples'] == 3
    trace = [json.loads(line) for line in (tmp_path/'run/trace.jsonl').read_text().splitlines()]
    assert [(r['from_ms'], r['to_ms']) for r in trace] == [(100., 110.), (110., 120.), (120., 123.1)]
    assert sum(r['longitudinal_delta_mm'] for r in trace) == pytest.approx(.0231)
    counts = np.load(tmp_path/'run/counts.npz')
    np.testing.assert_array_equal(counts['start_counts'], starting)
    assert result['total_spikes'] == int((counts['end_counts']-counts['start_counts']).sum())
    assert sum(r['spikes'] for r in trace) == result['total_spikes']
    with pytest.raises(FileExistsError):
        record_sequence(a, protocol, tmp_path/'run')


def test_invalid_plan_leaves_no_output_and_failed_run_is_not_marked_successful(tmp_path):
    session = CoupledSession(small_brain(), RecordingBody())
    with pytest.raises(ValueError):
        record_sequence(session, {'format': 'wrong'}, tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists() and session.brain.step == 0
    session.body.sim.mj_data.qpos[0] = float('nan')
    with pytest.raises(RuntimeError, match='Non-finite'):
        record_sequence(session, {'format': 'nexus-protocol-1', 'duration_ms': 20, 'events': []}, tmp_path/'bad')
    report = json.loads((tmp_path/'bad/report.json').read_text())
    assert not report['success'] and report['status'] == 'failed'


def test_all_shipped_presets_are_bound_to_their_dataset():
    from nexus.brain.config import default_pack
    from nexus.brain.runtime import Brain, Connectome
    root = Path(__file__).resolve().parents[1]
    for dataset, folder in [('male-cns', root/'experiments'), ('flywire-v630', root/'experiments/flywire-v630')]:
        brain = Brain(Connectome.load(default_pack(dataset), allow_experimental=dataset == 'male-cns'))
        for path in folder.glob('*-preset.json'):
            protocol = json.loads(path.read_text())
            assert protocol['dataset'] == brain.graph.snapshot
            Protocol(protocol, brain)
        if dataset == 'male-cns':
            plan = json.loads((folder/'male-cns-continuous-sequence.json').read_text())
            Protocol(plan, brain)
            assert len(plan['stages']) == 5
