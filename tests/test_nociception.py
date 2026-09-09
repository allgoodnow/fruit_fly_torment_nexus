"""Guard the published ID crosswalk and directed, thresholded pathway analysis."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nexus.datasets.nociception import literal_assignment, map_cohort, trace_ascending_paths
from nexus.brain.runtime import Brain, Connectome
from nexus.brain.protocol import Protocol


def test_source_literal_is_read_without_executing_notebook_code(tmp_path):
    destination = tmp_path/'must-not-exist'
    notebook = {'cells': [{'source': [f"open({str(destination)!r}, 'w').write('bad')\nmd = [3, 1, 2]\n"]}]}
    assert literal_assignment(notebook, 0, 'md') == [3, 1, 2]
    assert not destination.exists()
    for source in ['md = [1, 1]', 'md = [True]', 'md = [1.5]', 'md = [-1]', 'md = []', 'md = make_ids()', 'md = [1]\nmd = [2]']:
        with pytest.raises((ValueError, TypeError)):
            literal_assignment({'cells': [{'source': [source]}]}, 0, 'md')


def test_crosswalk_never_picks_an_arbitrary_split_or_nonruntime_match():
    frame = pd.DataFrame({'bodyId': [101, 102, 103, 104, 105, 106],
                          'mancBodyid': [1, 2, 2, 3, 4, 5],
                          'superclass': ['vnc_sensory']*4+['ascending_neuron', 'vnc_sensory'],
                          'subclass': ['abdomen']*5+['leg']})
    for name in ['type', 'mancType', 'class', 'status', 'entryNerve']:
        frame[name] = None
    selected, rows = map_cohort([1, 2, 3, 4, 5, 6], frame, [101, 102, 105, 106])
    assert selected == [101]
    assert [r['status'] for r in rows] == ['eligible_candidate', 'ambiguous_match',
                                          'outside_classified_runtime', 'incompatible_annotation',
                                          'incompatible_annotation', 'missing_match']
    with pytest.raises(ValueError, match='Duplicate'):
        map_cohort([1], pd.concat([frame, frame.iloc[:1]]), [101])


def test_paths_are_directed_and_apply_threshold_to_each_edge():
    # 10->20(4),10->30(3),11->20(5),20->40(6),30->40(9),40->11(100).
    ids = np.array([10, 11, 20, 30, 40])
    offsets = np.array([0, 2, 3, 4, 5, 6])
    posts = np.array([2, 3, 2, 4, 4, 1])
    contacts = np.array([4, 3, 5, 6, 9, 100])
    edges, paths = trace_ascending_paths(ids, offsets, posts, contacts, [10, 11], [20, 30], {'target': [40]})
    assert edges == [{'pre': 10, 'post': 20, 'contacts': 4}, {'pre': 11, 'post': 20, 'contacts': 5}]
    assert [(p['source'], p['ascending'], p['target']) for p in paths] == [(10, 20, 40), (11, 20, 40)]
    assert paths[0]['sensory_to_ascending_contacts'] == 4
    assert paths[0]['ascending_to_target_contacts'] == 6
    assert not trace_ascending_paths(ids, offsets, posts, contacts, [40], [20, 30], {'target': [10]})[1]
    with pytest.raises(ValueError, match='absent'):
        trace_ascending_paths(ids, offsets, posts, contacts, [999], [20], {'target': [40]})


def test_protocol_targets_exactly_the_eligible_male_cohort_and_runs_with_existing_commands():
    root = Path(__file__).resolve().parents[1]
    cohort = json.loads((root/'experiments/nociception-cohort-v1.json').read_text())
    description = json.loads((root/'experiments/nociception-input-v1.json').read_text())
    chosen = {int(r['matches'][0]['bodyId']) for r in cohort['mapping'] if r['status'] == 'eligible_candidate'}
    assert set(map(int, cohort['ids'])) == chosen and len(chosen) == 24
    assert description['events'][1]['ids'] == cohort['ids']
    # Execute timing on a zero-edge model with exactly this specimen's source IDs.
    graph = Connectome.from_edges(sorted(chosen), [], [], [], snapshot='male-cns:v1.0')
    brain = Brain(graph)
    protocol = Protocol(description, brain)
    for _ in range(70):
        protocol.advance(brain, 100)
    assert protocol.completed and brain.time == .7 and not brain.inputs.size
    assert brain.counts.sum() > 0
    assert [(e['kind'], e['time']) for e in brain.events] == [('release', 0.), ('stimulate', .1), ('release', .4)]


def test_registry_attachment_preserves_central_target_and_rejects_bad_crosswalks():
    import copy
    from nexus.datasets.nociception import attach_cohort
    cohort = {'dataset': 'male-cns:v1.0', 'annotation_source': {'sha256': 'pinned'},
              'ids': ['101'], 'selected_count': 1, 'source_reference': {'paper': 'source'},
              'mapping': [{'status': 'eligible_candidate', 'matches': [{'bodyId': 101}]}]}
    registry = {'snapshot': 'male-cns:v1.0', 'circuits': {'aversion_proxy': {'ids': ['999']}}}
    attached = attach_cohort(registry, cohort, [101, 999], 'pinned')
    assert attached['circuits']['aversion_proxy'] == registry['circuits']['aversion_proxy']
    assert attached['circuits']['nociception_proxy']['ids'] == ['101']
    assert 'nociception_proxy' not in registry['circuits']
    for key, value in [('dataset', '630'), ('ids', ['102']), ('selected_count', 2)]:
        invalid = copy.deepcopy(cohort)
        invalid[key] = value
        with pytest.raises(ValueError):
            attach_cohort(registry, invalid, [101, 999], 'pinned')
    with pytest.raises(ValueError, match='annotations'):
        attach_cohort(registry, cohort, [101, 999], 'other')


def test_existing_pain_button_protocol_matches_the_investigated_input():
    from nexus.brain.scenarios import scenario_protocol
    from nexus.brain.config import default_pack, model_config
    root = Path(__file__).resolve().parents[1]
    cohort = json.loads((root/'experiments/nociception-cohort-v1.json').read_text())
    config = model_config(default_pack())
    assert config.nociception_count == 24 and config.pain_circuit == 'nociception_proxy'
    a, b = [Brain(Connectome.from_edges(cohort['ids'], [], [], [], snapshot='male-cns:v1.0')) for _ in range(2)]
    a.graph.circuits = {'circuits': {'nociception_proxy': {'ids': cohort['ids']}}}
    named = scenario_protocol('aversion', dataset=config.snapshot, pain_circuit=config.pain_circuit,
                              baseline_ms=100, stimulus_ms=300, recovery_ms=300)
    direct = json.loads((root/'experiments/nociception-input-v1.json').read_text())
    p, q = Protocol(named, a), Protocol(direct, b)
    for _ in range(70):
        p.advance(a, 100)
        q.advance(b, 100)
    np.testing.assert_array_equal(a.counts, b.counts)
    np.testing.assert_array_equal(a.v, b.v)
    assert p.completed and q.completed and not a.inputs.size
