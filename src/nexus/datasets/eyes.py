"""Specimen-bound, bilateral R1-R6 inputs for pooled eye brightness."""
import copy


def attach_eye_inputs(registry, annotations, transmitters, runtime_ids):
    if registry.get('snapshot') != 'male-cns:v1.0':
        raise ValueError('Eye mapping requires MaleCNS v1.0')
    if annotations.bodyId.duplicated().any() or transmitters.body.duplicated().any():
        raise ValueError('Duplicate annotation or transmitter body IDs')
    available = set(map(int, runtime_ids))
    rows = annotations.loc[annotations.bodyId.isin(available) & annotations.type.eq('R1-R6')].copy()
    labels = transmitters.set_index('body').consensus_nt.reindex(rows.bodyId)
    if rows.empty or not rows.superclass.eq('ol_sensory').all() or not labels.eq('histamine').all():
        raise ValueError('Expected histaminergic R1-R6 sensory cells')
    if not rows.rootSide.isin(['L', 'R']).all():
        raise ValueError('Photoreceptor eye side is unresolved')
    result = copy.deepcopy(registry)
    for side, name in [('L', 'eye_left'), ('R', 'eye_right')]:
        selected = rows.loc[rows.rootSide.eq(side)]
        if selected.empty:
            raise ValueError('Both eye cohorts are required')
        result['circuits'][name] = {
            'ids': [str(i) for i in sorted(map(int, selected.bodyId))],
            'selection': f'type == R1-R6; rootSide == {side}; ol_sensory; consensus_nt == histamine',
            'limitation': 'Pooled eye brightness to scanned R1-R6 cells; no pixel-to-cell '
                          'retinotopy, fitted phototransduction, or graded transmission.'}
    return result
