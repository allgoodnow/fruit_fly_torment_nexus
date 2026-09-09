"""Auditable MANC-to-MaleCNS mapping of published abdominal md candidates."""
import ast
import json

import numpy as np


def literal_assignment(notebook, cell_index, variable):
    """Read a literal from a notebook without executing any notebook code."""
    source = ''.join(notebook['cells'][cell_index]['source'])
    matches = [node.value for node in ast.parse(source).body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == variable for target in node.targets)]
    if len(matches) != 1:
        raise ValueError(f'Expected one assignment for {variable}')
    values = ast.literal_eval(matches[0])
    if (not isinstance(values, list) or not values or any(type(v) is not int or v <= 0 for v in values)
            or len(set(values)) != len(values)):
        raise ValueError('Source cohort must contain unique positive integer IDs')
    return values


def map_cohort(source_ids, annotations, runtime_ids):
    """Exclude missing, ambiguous, unclassified and incompatible correspondences."""
    available = set(map(int, runtime_ids))
    if annotations.bodyId.duplicated().any():
        raise ValueError('Duplicate MaleCNS body IDs')
    rows, selected = [], []
    fields = ['bodyId', 'mancBodyid', 'type', 'mancType', 'class', 'subclass', 'superclass', 'status', 'entryNerve']
    for old in source_ids:
        matches = annotations.loc[annotations.mancBodyid.eq(old)]
        if len(matches) == 0:
            status = 'missing_match'
        elif len(matches) > 1:
            status = 'ambiguous_match'
        elif int(matches.bodyId.iloc[0]) not in available:
            status = 'outside_classified_runtime'
        elif matches.superclass.iloc[0] != 'vnc_sensory' or matches.subclass.iloc[0] != 'abdomen':
            status = 'incompatible_annotation'
        else:
            status = 'eligible_candidate'
            selected.append(int(matches.bodyId.iloc[0]))
        rows.append({'manc_body_id': old, 'status': status,
                     'matches': json.loads(matches[fields].to_json(orient='records'))})
    return sorted(selected), rows


def trace_ascending_paths(ids, offsets, posts, contacts, source_ids, ascending_ids, targets, min_contacts=4):
    """Enumerate two-edge structural paths; thresholds are per connection."""
    if type(min_contacts) is not int or min_contacts < 1:
        raise ValueError('Contact threshold must be a positive integer')
    lookup = {int(root): i for i, root in enumerate(ids)}
    if not set(source_ids) <= lookup.keys() or not set(ascending_ids) <= lookup.keys():
        raise ValueError('Path source or relay absent from graph')
    target_names = {int(root): name for name, roots in targets.items() for root in roots}
    if not target_names.keys() <= lookup.keys():
        raise ValueError('Path target absent from graph')
    ascending = set(ascending_ids)
    incoming = {}
    source_edges = []
    for root in source_ids:
        i = lookup[root]
        for edge in range(offsets[i], offsets[i+1]):
            post, count = int(ids[posts[edge]]), int(contacts[edge])
            if count >= min_contacts and post in ascending:
                incoming.setdefault(post, []).append((root, count))
                source_edges.append({'pre': root, 'post': post, 'contacts': count})
    paths = []
    for relay in sorted(incoming):
        i = lookup[relay]
        for edge in range(offsets[i], offsets[i+1]):
            target, count = int(ids[posts[edge]]), int(contacts[edge])
            if count >= min_contacts and target in target_names:
                for source, first_count in incoming[relay]:
                    paths.append({'source': source, 'ascending': relay, 'target': target,
                                  'target_group': target_names[target],
                                  'sensory_to_ascending_contacts': first_count,
                                  'ascending_to_target_contacts': count})
    return source_edges, sorted(paths, key=lambda p: (p['target_group'], p['source'], p['ascending'], p['target']))


def attach_cohort(registry, cohort, runtime_ids, annotation_sha256):
    """Add the investigated sensory cohort without replacing central readouts."""
    import copy
    import hashlib
    if cohort.get('dataset') != registry.get('snapshot') or cohort.get('dataset') != 'male-cns:v1.0':
        raise ValueError('Nociception cohort belongs to another dataset')
    if cohort['annotation_source']['sha256'] != annotation_sha256:
        raise ValueError('Nociception cohort uses different annotations')
    chosen = [int(i) for i in cohort['ids']]
    eligible = [int(r['matches'][0]['bodyId']) for r in cohort['mapping']
                if r['status'] == 'eligible_candidate' and len(r['matches']) == 1]
    if (not chosen or len(chosen) != len(set(chosen)) or set(chosen) != set(eligible)
            or len(chosen) != cohort['selected_count'] or not set(chosen) <= set(map(int, runtime_ids))):
        raise ValueError('Cohort IDs do not match eligible runtime correspondences')
    updated = copy.deepcopy(registry)
    updated['circuits']['nociception_proxy'] = {
        'ids': [str(i) for i in chosen],
        'selection': 'Unambiguous published abdominal md MANC-to-MaleCNS counterparts',
        'source': cohort['source_reference']['paper'],
        'cohort_canonical_sha256': hashlib.sha256(json.dumps(cohort,sort_keys=True,separators=(',', ':')).encode()).hexdigest(),
        'limitation': 'Cross-specimen sensory candidates; stimulation is an experimental nociception proxy, not validated subjective pain.',
    }
    return updated
