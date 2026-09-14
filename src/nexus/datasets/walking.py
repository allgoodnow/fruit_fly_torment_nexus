"""Specimen-bound BDN2 forward-walking readout from official annotations."""
from copy import deepcopy


def attach_forward_readout(registry, annotations, runtime_ids):
    if registry.get('snapshot') != 'male-cns:v1.0':
        raise ValueError('Forward walking mapping requires MaleCNS v1.0')
    if annotations.bodyId.duplicated().any():
        raise ValueError('Duplicate annotation body IDs')
    rows = annotations.loc[annotations.type.eq('DNg100')]
    if (len(rows) != 2 or set(rows.somaSide) != {'L', 'R'}
            or not rows.superclass.eq('descending_neuron').all()
            or not rows.synonyms.fillna('').str.contains('Sapkal 2024: BDN2', regex=False).all()
            or not set(map(int, rows.bodyId)) <= set(map(int, runtime_ids))):
        raise ValueError('Expected two annotated BDN2/DNg100 cells, one per side, in this pack')
    result = deepcopy(registry)
    result['readouts']['forward_walking'] = [str(int(rows.loc[rows.somaSide.eq(side), 'bodyId'].iloc[0]))
                                             for side in ['L', 'R']]
    result['forward_walking_evidence'] = {
        'selection': 'type DNg100; synonym Sapkal 2024: BDN2; left then right somaSide',
        'source': 'https://doi.org/10.1038/s41586-024-07854-7',
        'limitation': 'Forward drive readout only; no reconstructed VNC-to-muscle controller.'}
    return result
