"""Evidence-scoped, exact annotation selection for graded visual transmission."""
import copy

LAMINA_TYPES = ('L1', 'L2')
MEDULLA_TYPES = ('Mi1', 'Tm3', 'Tm1', 'Tm2')


def attach_graded_relays(registry, annotations, ids, *, include_medulla=False):
    if registry.get('snapshot') != 'male-cns:v1.0' or annotations.bodyId.duplicated().any():
        raise ValueError('Unique MaleCNS annotations required')
    rows = annotations.set_index('bodyId').reindex(ids)
    groups = {}
    types = LAMINA_TYPES + MEDULLA_TYPES if include_medulla else LAMINA_TYPES
    for typ in types:
        for side in ('L', 'R'):
            selected = rows[(rows['type'] == typ) & (rows['instance'] == f'{typ}_{side}')
                            & (rows['superclass'] == 'ol_intrinsic')]
            if selected.empty:
                raise ValueError(f'Both exact {typ} eye cohorts are required')
            groups[f'{typ}_{side}'] = [str(int(root)) for root in selected.index]
    result = copy.deepcopy(registry)
    result['graded_relays'] = {
        'format': 'nexus-graded-relays-2' if include_medulla else 'nexus-graded-relays-1', 'groups': groups,
        'selection': 'Exact types ' + '/'.join(types) + ', matching L/R instance, ol_intrinsic superclass',
        'source': 'https://doi.org/10.1016/j.cub.2024.11.064',
        'limit': 'Non-spiking identity evidence only; release and reversal parameters are unfitted.'}
    if include_medulla:
        result['graded_relays']['synapse_model'] = 'conductance-v1'
        result['graded_relays']['medulla_evidence'] = {
            'types': list(MEDULLA_TYPES),
            'source': 'https://doi.org/10.1016/j.cell.2016.05.031',
            'support': 'Voltage imaging and electrophysiology support graded signals in Mi1/Tm3/Tm1/Tm2.',
            'limit': 'The shared release curve is an engineering approximation, not a fit to type-specific recordings.'}
    return result


def relay_ids(registry):
    record = registry.get('graded_relays', {})
    groups = record.get('groups', {})
    version = record.get('format')
    types = LAMINA_TYPES + MEDULLA_TYPES if version == 'nexus-graded-relays-2' else LAMINA_TYPES
    expected = {f'{typ}_{side}' for typ in types for side in ('L', 'R')}
    if (registry.get('snapshot') != 'male-cns:v1.0'
            or version not in ('nexus-graded-relays-1', 'nexus-graded-relays-2') or set(groups) != expected):
        raise ValueError('Install the exact graded-visual-relay cohort first')
    if record.get('synapse_model') not in (None, 'conductance-v1'):
        raise ValueError('Unknown graded synapse model')
    if version == 'nexus-graded-relays-2' and record.get('synapse_model') != 'conductance-v1':
        raise ValueError('Extended graded relays require the conductance model')
    ids = []
    for key in sorted(groups):
        if not groups[key] or any(not isinstance(root, str) or not root.isdecimal() for root in groups[key]):
            raise ValueError('Graded relay IDs must be nonempty decimal-string lists')
        ids.extend(groups[key])
    if len(ids) != len(set(ids)):
        raise ValueError('Graded relay cohorts overlap')
    return ids
