"""Select annotated MaleCNS LC4 cells without crossing specimen identifiers."""
import copy


def attach_velocity_circuit(registry, annotations, runtime_ids):
    if registry.get('snapshot') != 'male-cns:v1.0':
        raise ValueError('LC4 mapping requires MaleCNS v1.0')
    if annotations.bodyId.duplicated().any():
        raise ValueError('Duplicate annotation body IDs')
    available = set(map(int, runtime_ids))
    size = annotations.loc[annotations.type.eq('LPLC2') & annotations.bodyId.isin(available)]
    if set(map(int, size.bodyId)) != set(map(int, registry['circuits']['looming']['ids'])):
        raise ValueError('LPLC2 registry does not match these annotations')
    rows = annotations.loc[annotations.type.eq('LC4') & annotations.bodyId.isin(available)]
    if rows.empty or not rows.superclass.eq('visual_projection').all():
        raise ValueError('No compatible annotated LC4 cohort')
    result = copy.deepcopy(registry)
    result['circuits']['looming_velocity'] = {
        'ids': [str(i) for i in sorted(map(int, rows.bodyId))],
        'selection': 'type == LC4; classified cells present in this MaleCNS runtime',
        'source': 'https://doi.org/10.1016/j.cub.2019.01.079',
        'limitation': 'Type-matched expansion-speed pathway; input rates and dynamics are unfitted.'}
    return result
