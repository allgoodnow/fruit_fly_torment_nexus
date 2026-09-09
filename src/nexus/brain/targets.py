"""Exact v630 IDs from the pinned upstream taste-circuit example."""
SUGAR = [str(i) for i in (
    720575940624963786,720575940630233916,720575940637568838,
    720575940638202345,720575940617000768,720575940630797113,
    720575940632889389,720575940621754367,720575940621502051,
    720575940640649691,720575940639332736,720575940616885538,
    720575940639198653,720575940620900446,720575940617937543,
    720575940632425919,720575940633143833,720575940612670570,
    720575940628853239,720575940629176663,720575940611875570)]
MN9 = "720575940660219265"


def readout_ids(name, graph):
    """Resolve readouts in the active specimen; never fall back across datasets."""
    if graph.snapshot == '630':
        if name == 'mn9':
            return [MN9]
        if name == 'sugar':
            return list(SUGAR)
    if graph.snapshot == 'male-cns:v1.0' and graph.circuits is not None:
        if name in graph.circuits['readouts']:
            return list(graph.circuits['readouts'][name])
        reason = graph.circuits.get('unavailable_inputs', {}).get(name)
        if reason:
            raise ValueError(reason)
    raise ValueError(f'No mapped readout {name!r} for dataset {graph.snapshot}')
