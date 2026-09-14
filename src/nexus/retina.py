"""Experimental video-plane sampling at inferred visual-column locations."""
import numpy as np


class VisualColumns:
    def __init__(self, registry):
        mapping = registry['visual_columns']
        if registry.get('snapshot') != 'male-cns:v1.0' or mapping.get('format') != 'nexus-visual-columns-1':
            raise ValueError('Unsupported visual-column mapping')
        self.ids, self.uv = {}, {}
        all_ids = set()
        for side, key in [('L', 'eye_left'), ('R', 'eye_right')]:
            cells = mapping['eyes'][side]['cells']
            ids = [str(c['id']) for c in cells]
            uv = np.asarray([c['uv'] for c in cells], dtype=np.float64)
            allowed = set(registry['circuits'][key]['ids'])
            if (not ids or len(set(ids)) != len(ids) or not set(ids) <= allowed
                    or set(ids) & all_ids or uv.shape != (len(ids), 2)
                    or not np.isfinite(uv).all() or ((uv < 0) | (uv > 1)).any()):
                raise ValueError('Invalid visual-column targets or coordinates')
            all_ids.update(ids)
            self.ids[side], self.uv[side] = ids, uv

    def sample(self, frame):
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or min(frame.shape[:2]) == 0:
            raise ValueError('Expected an RGB8 video frame')
        luminance = frame.mean(axis=2) / 255.
        height, width = luminance.shape
        result = {}
        for side, uv in self.uv.items():
            x, y = uv[:, 0]*(width-1), uv[:, 1]*(height-1)
            x0, y0 = x.astype(int), y.astype(int)
            x1, y1 = np.minimum(x0+1, width-1), np.minimum(y0+1, height-1)
            dx, dy = x-x0, y-y0
            result[side] = ((1-dx)*(1-dy)*luminance[y0, x0] + dx*(1-dy)*luminance[y0, x1]
                            + (1-dx)*dy*luminance[y1, x0] + dx*dy*luminance[y1, x1])
        return result
