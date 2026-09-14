"""Exact type matching, migration equivalence, and causal receptor-sign tests."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from nexus.brain.runtime import Brain, Connectome
from nexus.datasets.histamine import upgrade_pack
from nexus.datasets.male_cns import FILES
from nexus.datasets.male_cns_runtime import MODEL_POLICY, transmitter_signs
from nexus.datasets.receptor_signs import RULES, POLICY_ID, compile_rules, apply_extended
from test_histamine import original_pack


@pytest.mark.parametrize('rule', RULES, ids=[r['id'] for r in RULES])
def test_each_rule_transmits_only_from_confirmed_types_to_its_targets(rule):
    sources = rule['presynaptic_types']
    targets = rule['postsynaptic_types']
    # Same-type nonhistaminergic and ambiguous photoreceptors must remain excluded.
    types = sources + [sources[0], 'R7_unclear', 'R8_unclear', 'R7d', 'R8d', 'T1'] + targets + ['Mi1']
    npre = len(types) - len(targets) - 1
    ids = np.arange(100, 100 + len(types), dtype=np.int64)
    pre = np.repeat(np.arange(npre), len(targets) + 1)
    post = np.tile(np.arange(npre, len(types)), npre)
    graph = Connectome.from_edges(ids, pre, post, np.zeros(len(pre)))
    neurons = pd.DataFrame({'bodyId': ids, 'type': types}).iloc[::-1]
    labels = pd.Series(['histamine'] * npre + ['acetylcholine'] * (len(targets) + 1), index=ids)
    labels.iloc[len(sources)] = 'acetylcholine'
    groups = compile_rules(ids, graph.offsets, graph.posts, neurons, labels, rules=[rule])
    expected = np.flatnonzero((pre < len(sources)) & (post < len(types) - 1))
    np.testing.assert_array_equal(groups[0][1], expected)
    contacts = np.full(len(pre), 20, dtype=np.uint32)
    model = apply_extended(graph.weights, contacts, ids, graph.offsets, graph.posts,
                           neurons, labels, copy.deepcopy(MODEL_POLICY), groups=groups)
    assert model['id'] == POLICY_ID
    np.testing.assert_array_equal(np.flatnonzero(graph.weights), expected)
    np.testing.assert_array_equal(graph.weights[expected], np.full(len(expected), -5.5))
    brain = Brain(graph)
    brain.stimulate([int(ids[0])], 100)
    events = np.zeros((500, 1), dtype=bool)
    events[0] = True
    traces = brain.advance(.05, input_events=events, trace_ids=ids[npre:])['voltage_mv']
    assert np.all(traces[:, :-1].min(axis=0) < -52.1)
    assert np.all(traces[:, -1] == -52)


def test_overlapping_rules_are_rejected():
    ids = np.array([1, 2])
    neurons = pd.DataFrame({'bodyId': ids, 'type': ['R1-R6', 'L3']})
    labels = pd.Series(['histamine', 'acetylcholine'], index=ids)
    duplicate = dict(RULES[0], id='overlapping-rule')
    with pytest.raises(ValueError, match='Overlapping'):
        compile_rules(ids, np.array([0, 1, 1]), np.array([1]), neurons, labels,
                      rules=[RULES[0], duplicate])
    with pytest.raises(ValueError, match='Duplicate'):
        compile_rules(ids, np.array([0, 1, 1]), np.array([1]), neurons, labels,
                      rules=[RULES[0], RULES[0]])


def test_direct_and_incremental_upgrades_have_identical_weights(original_pack):
    pack, structural, raw = original_pack
    original = (pack / 'manifest.json').read_bytes()
    upgrade_pack(pack, structural, raw)
    before = Connectome.load(pack, allow_experimental=True)
    manifest, evidence = upgrade_pack(pack, structural, raw, extended=True)
    after = Connectome.load(pack, allow_experimental=True)
    assert evidence['changed_weights'] == 1 and after.model['id'] == POLICY_ID
    np.testing.assert_array_equal(after.weights, [-2.75, -5.5, -8.25, 2.2, 0])
    np.testing.assert_array_equal(before.weights, [-2.75, -5.5, 0, 2.2, 0])
    assert len(after.model['edge_sign_overrides']) == 2
    prior_manifest = (pack / 'manifest.json').read_bytes()
    with pytest.raises(ValueError, match='already be installed'):
        upgrade_pack(pack, structural, raw, extended=True)
    assert (pack / 'manifest.json').read_bytes() == prior_manifest
    (pack / 'manifest.json').write_bytes(original)
    direct, direct_evidence = upgrade_pack(pack, structural, raw, extended=True)
    assert direct_evidence['changed_weights'] == 3
    assert direct['weights_file'] == manifest['weights_file']
    assert direct['model'] == manifest['model']


@pytest.mark.parametrize('field', ['weight', 'rule', 'policy'])
def test_changed_previous_policy_is_rejected_atomically(original_pack, field):
    from nexus.datasets.male_cns import digest
    pack, structural, raw = original_pack
    upgrade_pack(pack, structural, raw)
    manifest = json.loads((pack / 'manifest.json').read_text())
    if field == 'rule':
        manifest['model']['edge_sign_overrides'][0]['postsynaptic_types'] = ['L3']
    elif field == 'policy':
        manifest['model']['nt_signs']['glutamate'] = 1
    else:
        weights = np.load(pack / manifest['weights_file']).copy()
        weights[0] *= 2
        np.save(pack / 'candidate.npy', weights)
        checksum = digest(pack / 'candidate.npy')
        name = f'weights-{checksum}.npy'
        (pack / 'candidate.npy').rename(pack / name)
        manifest['weights_file'] = name
        manifest['files'][name] = checksum
    (pack / 'manifest.json').write_text(json.dumps(manifest))
    before = (pack / 'manifest.json').read_bytes()
    with pytest.raises(ValueError, match='Prior lamina|prior model|weights differ'):
        upgrade_pack(pack, structural, raw, extended=True)
    assert (pack / 'manifest.json').read_bytes() == before


def test_fresh_weight_policy_matches_incremental_install(original_pack):
    pack, structural, raw = original_pack
    old = Connectome.load(pack, allow_experimental=True)
    contacts = np.load(structural / 'contacts.npy')
    neurons = pd.read_feather(raw / FILES['annotations'])
    _, labels = transmitter_signs(old.ids, pd.read_feather(raw / FILES['neurotransmitters']))
    weights = old.weights.copy()
    fresh = apply_extended(weights, contacts, old.ids, old.offsets, old.posts, neurons, labels, old.model)
    upgrade_pack(pack, structural, raw)
    manifest, _ = upgrade_pack(pack, structural, raw, extended=True)
    np.testing.assert_array_equal(weights, Connectome.load(pack, allow_experimental=True).weights)
    assert fresh == manifest['model']
