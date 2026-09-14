"""Evidence-scoped visual sign rules applied to existing MaleCNS contacts.

These are curated type-level inferences, not per-synapse receptor measurements.
No polarity, receptor or transmitter is inferred from a name prefix.
"""
import copy
import hashlib

import numpy as np

from .histamine import POLICY_ID as LAMINA_POLICY, RULE_ID as LAMINA_RULE
from .histamine import select_edges, apply_override

POLICY_ID = 'male-cns-visual-receptor-lif-v3'
RULES = (
    {'id': 'histamine-R1-R6-to-L3-ort-v1',
     'presynaptic_types': ['R1-R6'], 'postsynaptic_types': ['L3'],
     'sources': ['https://doi.org/10.1523/JNEUROSCI.1654-08.2008'],
     'evidence': 'HCLA reporter expression in L1-L3 and native lamina histamine receptor physiology.',
     'mapping_limit': 'Type-level receptor inference, not a receptor measurement in this specimen.'},
    {'id': 'histamine-R7py-to-Dm8-ort-v1',
     'presynaptic_types': ['R7p', 'R7y'], 'postsynaptic_types': ['Dm8a', 'Dm8b'],
     'sources': ['https://doi.org/10.1016/j.cub.2021.01.105',
                 'https://doi.org/10.1038/s41586-024-07981-1'],
     'evidence': 'R7p/y-driven Dm8 inhibition requires Ort; Dm8-specific Ort rescue restores inhibition.',
     'mapping_limit': 'Dm8 class evidence applies to both connectomic subdivisions. No one-to-one '
                      'Dm8a/b to yellow/pale Dm8 mapping or spectral selectivity is assumed.'},
    {'id': 'histamine-R7py-to-Tm5ab-ort-v1',
     'presynaptic_types': ['R7p', 'R7y'], 'postsynaptic_types': ['Tm5a', 'Tm5b'],
     'sources': ['https://doi.org/10.1016/j.neuron.2008.08.010',
                 'https://doi.org/10.1038/s41586-024-07981-1'],
     'evidence': 'Ort-expressing Tm5a/b types and their R7 anatomical inputs are reported.',
     'mapping_limit': 'A receptor-expression and type-correspondence inference; no fitted Tm5 physiology.'},
    {'id': 'histamine-R8py-to-Ort-targets-v1',
     'presynaptic_types': ['R8p', 'R8y'], 'postsynaptic_types': ['L1', 'Tm5c', 'Tm9', 'Tm20'],
     'sources': ['https://doi.org/10.1038/s41586-023-06681-6',
                 'https://doi.org/10.1016/j.neuron.2008.08.010',
                 'https://doi.org/10.1038/s41586-024-07981-1'],
     'evidence': 'R8-to-L1/Tm9/Tm20 transmission is predominantly histaminergic and hyperpolarizing; '
                 'Ort expression and R8 input support the Tm5c sign.',
     'mapping_limit': 'Minor cholinergic transmission is omitted. This does not assign a global '
                      'R8 sign, model cotransmission, or infer an AMA excitatory pathway.'},
)


def edge_digest(edges):
    return hashlib.sha256(np.asarray(edges, dtype='<i8').tobytes()).hexdigest()


def compile_rules(ids, offsets, posts, neurons, labels, *, rules=RULES):
    groups, seen = [], set()
    names = set()
    for rule in rules:
        if rule['id'] in names:
            raise ValueError('Duplicate receptor rule ID')
        names.add(rule['id'])
        edges = select_edges(ids, offsets, posts, neurons, labels,
                             presynaptic_types=rule['presynaptic_types'],
                             postsynaptic_types=rule['postsynaptic_types'])
        if seen.intersection(edges.tolist()):
            raise ValueError('Overlapping receptor rules')
        seen.update(edges.tolist())
        groups.append((rule, edges))
    return groups


def validate_prior_policy(model, contacts, lamina_edges):
    from .male_cns_runtime import MODEL_POLICY
    for key in ['nt_signs', 'weight_per_contact_mv', 'dynamics', 'experimental']:
        if model.get(key) != MODEL_POLICY[key]:
            raise ValueError(f'Unsupported prior model assumption: {key}')
    if model.get('id') == MODEL_POLICY['id']:
        if model.get('edge_sign_overrides'):
            raise ValueError('Unexpected prior receptor overrides')
    elif model.get('id') == LAMINA_POLICY:
        rules = model.get('edge_sign_overrides', [])
        expected = {'id': LAMINA_RULE, 'presynaptic_type': 'R1-R6',
                    'presynaptic_consensus_nt': 'histamine', 'postsynaptic_types': ['L1', 'L2'],
                    'sign': -1, 'receptor': 'Ort / HCLA', 'edges': len(lamina_edges),
                    'contacts': int(contacts[lamina_edges].sum()),
                    'edge_indices_sha256': edge_digest(lamina_edges)}
        if len(rules) != 1 or any(rules[0].get(k) != v for k, v in expected.items()):
            raise ValueError('Prior lamina rule does not match the scanned graph')
    else:
        raise ValueError('Unsupported prior receptor policy; it may already be installed')


def apply_extended(weights, contacts, ids, offsets, posts, neurons, labels, model, *, groups=None):
    lamina_edges = select_edges(ids, offsets, posts, neurons, labels)
    validate_prior_policy(model, contacts, lamina_edges)
    if groups is None:
        groups = compile_rules(ids, offsets, posts, neurons, labels)
    # Validate every new rule before modifying any weights.
    for _, edges in groups:
        if np.any(weights[edges] != 0) or np.any(contacts[edges] <= 0):
            raise ValueError('Receptor rule requires omitted edges with positive scanned contacts')
    if model['id'] == LAMINA_POLICY:
        expected = -contacts[lamina_edges].astype(np.float64) * model['weight_per_contact_mv']
        if not np.array_equal(weights[lamina_edges], expected):
            raise ValueError('Previous lamina weights changed')
        result = copy.deepcopy(model)
    else:
        result = apply_override(weights, contacts, lamina_edges, model)
    records = result.setdefault('edge_sign_overrides', [])
    for rule, edges in groups:
        if not len(edges):
            continue
        weights[edges] = -contacts[edges].astype(np.float64) * model['weight_per_contact_mv']
        records.append(dict(copy.deepcopy(rule), presynaptic_consensus_nt='histamine', sign=-1,
                            receptor='Ort / HCLA', edges=len(edges),
                            contacts=int(contacts[edges].sum()), edge_indices_sha256=edge_digest(edges)))
    if any(len(edges) for _, edges in groups):
        result['id'] = POLICY_ID
        result['sign_assumptions'] = (
            'Presynaptic fast-transmitter signs with audited receptor-informed histamine exceptions '
            'in lamina and color pathways. Other histamine and monoamine effects remain omitted. '
            'Glutamate remains assumed inhibitory; no global R8 sign is assigned.')
        result['receptor_rule_limits'] = (
            'Exact annotated types and consensus histamine required; unclear and dorsal R7/R8 types excluded. '
            'No per-synapse receptor measurement, graded retinal dynamics, cotransmission, '
            'calibrated strength, or visual perception is reconstructed.')
    if not records:
        result.pop('edge_sign_overrides')
    coverage = result.get('transmitter_coverage', {}).get('histamine')
    if coverage is not None:
        total = sum(rule['edges'] for rule in records)
        coverage['overridden_inhibitory_edges'] = total
        coverage['omitted_edges'] = coverage['outgoing_edges'] - total
    return result
