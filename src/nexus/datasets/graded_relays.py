"""Exact annotation selection for an experimental non-spiking L1/L2 model."""
import copy


def attach_graded_relays(registry, annotations, ids):
    if registry.get('snapshot') != 'male-cns:v1.0' or annotations.bodyId.duplicated().any():
        raise ValueError('Unique MaleCNS annotations required')
    rows = annotations.set_index('bodyId').reindex(ids)
    groups = {}
    for typ in ('L1', 'L2'):
        for side in ('L', 'R'):
            selected = rows[(rows['type'] == typ) & (rows['instance'] == f'{typ}_{side}')
                            & (rows['superclass'] == 'ol_intrinsic')]
            if selected.empty:
                raise ValueError('Both exact L1/L2 eye cohorts are required')
            groups[f'{typ}_{side}'] = [str(int(root)) for root in selected.index]
    result = copy.deepcopy(registry)
    result['graded_relays'] = {
        'format': 'nexus-graded-relays-1', 'groups': groups,
        'selection': 'Exact type L1/L2, matching L/R instance, ol_intrinsic superclass',
        'source': 'https://doi.org/10.1016/j.cub.2024.11.064',
        'limit': 'Non-spiking identity evidence only; release and reversal parameters are unfitted.'}
    return result


def relay_ids(registry):
    record = registry.get('graded_relays', {})
    groups = record.get('groups', {})
    expected = {'L1_L', 'L1_R', 'L2_L', 'L2_R'}
    if (registry.get('snapshot') != 'male-cns:v1.0'
            or record.get('format') != 'nexus-graded-relays-1' or set(groups) != expected):
        raise ValueError('Install the exact L1/L2 graded-relay cohort first')
    ids = []
    for key in sorted(groups):
        if not groups[key] or any(not isinstance(root, str) or not root.isdecimal() for root in groups[key]):
            raise ValueError('Graded relay IDs must be nonempty decimal-string lists')
        ids.extend(groups[key])
    if len(ids) != len(set(ids)):
        raise ValueError('Graded relay cohorts overlap')
    return ids
