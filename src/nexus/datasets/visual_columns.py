"""Infer R1–R6 column identity from independently agreeing L1 and L2 targets."""
from collections import defaultdict, Counter
import copy

import numpy as np


def attach_visual_columns(registry, annotations, ids, offsets, posts, contacts):
    if registry.get('snapshot') != 'male-cns:v1.0' or annotations.bodyId.duplicated().any():
        raise ValueError('Unique MaleCNS annotations are required')
    rows = annotations.set_index('bodyId').reindex(ids)
    lookup = {int(root): i for i, root in enumerate(ids)}
    result = copy.deepcopy(registry)
    mapping = {'format': 'nexus-visual-columns-1', 'eyes': {},
               'criteria': 'L1 and L2 independently agree on a unique strongest column; '
                           'at least 90% of eligible contacts support it; at least 10 contacts.',
               'projection': 'x=sqrt(3)/2*(hex1-hex2), y=-(hex1+hex2)/2; '
                             'full L1/L2 column extent centered in a unit square',
               'limitations': 'Inferred column identity and uncalibrated video projection; '
                              'not a registered camera retina, measured receptive field, or motion detector.'}
    for side, key in [('L', 'eye_left'), ('R', 'eye_right')]:
        target_columns = {}
        for i, row in enumerate(rows.itertuples()):
            if (row.type not in ('L1', 'L2') or row.instance != f'{row.type}_{side}'
                    or not np.isfinite([row.assignedOlHex1, row.assignedOlHex2]).all()):
                continue
            q, r = float(row.assignedOlHex1), float(row.assignedOlHex2)
            if q != int(q) or r != int(r):
                raise ValueError('Column coordinates must be integers')
            target_columns[i] = (row.type, (int(q), int(r)))
        if not target_columns:
            raise ValueError('Both eye column grids are required')
        full_hex = np.unique([pos for _, pos in target_columns.values()], axis=0)
        full_xy = np.column_stack((np.sqrt(3)/2*(full_hex[:, 0]-full_hex[:, 1]),
                                   -(full_hex[:, 0]+full_hex[:, 1])/2))
        low, high = full_xy.min(axis=0), full_xy.max(axis=0)
        span = float((high-low).max())
        if span <= 0:
            raise ValueError('Column grid has no spatial extent')
        selected, excluded = [], Counter()
        for root in registry['circuits'][key]['ids']:
            i = lookup[int(root)]
            if rows.iloc[i].type != 'R1-R6' or rows.iloc[i].rootSide != side:
                raise ValueError('Eye cohort conflicts with source annotations')
            support = {'L1': defaultdict(int), 'L2': defaultdict(int)}
            for edge in range(int(offsets[i]), int(offsets[i+1])):
                target = target_columns.get(int(posts[edge]))
                if target is not None:
                    typ, pos = target
                    support[typ][pos] += int(contacts[edge])
            winners = []
            for typ in ['L1', 'L2']:
                top = sorted(support[typ], key=lambda p: (-support[typ][p], p))
                if not top or (len(top) > 1 and support[typ][top[0]] == support[typ][top[1]]):
                    break
                winners.append(top[0])
            if len(winners) != 2 or winners[0] != winners[1]:
                excluded['missing_or_disagreeing_L1_L2'] += 1
                continue
            pos = winners[0]
            counts = [support[t][pos] for t in ['L1', 'L2']]
            total = sum(sum(group.values()) for group in support.values())
            if sum(counts) < 10 or 10*sum(counts) < 9*total:
                excluded['weak_or_distributed_support'] += 1
                continue
            xy = np.array([np.sqrt(3)/2*(pos[0]-pos[1]), -(pos[0]+pos[1])/2])
            uv = (xy-(low+high)/2)/span + .5
            selected.append({'id': str(root), 'hex': list(pos), 'uv': uv.tolist(),
                             'L1_L2_contacts': counts, 'eligible_contacts': total})
        if not selected:
            raise ValueError('No reliable photoreceptor column assignments')
        mapping['eyes'][side] = {'cells': selected, 'excluded': dict(excluded),
                                 'full_grid_columns': len(full_hex),
                                 'covered_columns': len({tuple(c['hex']) for c in selected})}
    result['visual_columns'] = mapping
    return result
