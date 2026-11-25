"""Ground-truth polylines of the CARLA 3-class 50 m tiles (numpy only, no mmdet).

The tiles come from point2vector_data (``tile.py``): every reference_lines json lists polylines
``{class, class_id, subclass, is_arc, road_id, points}`` in WORLD coordinates. The annotation
cuts each line into pieces of a single sub class (straight / arc) and leaves the ``arc_gap``
(0.1 m) between a straight and the arc it runs into, so one kerb along one road is often two or
three pieces. That cut is the annotation's sub-class bookkeeping, not a break in the line: it is
the CARLA counterpart of the HRDX design elements that ``hrdx_param_gt.chain_parametric`` joins
per alignment. This module joins it the same way:

  * only the requested classes are read (the benchmark's: centerline, curb_model, road_edge);
  * pieces of the same class AND the same ``road_id`` are joined end to end when two of their
    ends lie within ``gap_tol`` (0.15 m: the 0.1 m arc gap, or 0 for clip pieces); a piece is
    reversed when needed (pieces of one road do not share a direction);
  * when the ends touch (< 1 mm) the shared point is kept once, otherwise both are kept;
  * a joined line is ``is_arc`` when any of its pieces is.

One ground-truth instance = one continuous in-tile stretch of one class along one road.
Measured on 600 tiles (2026-11-25): every end pair within 0.5 m of another piece of its class
is on the same road, at 0.0 or 0.1 m.

Shared verbatim by every sister baseline (MapTR, GeMap, PseudoMapTrainer, mapdiffusion,
maptracker, SuperMapNet, CurbNet), so all of them see exactly the same lines.
"""
import json

import numpy as np

CARLA_3CLS_CLASSES = ('centerline', 'curb_model', 'road_edge')


def _ends_join(a, b, tol):
    """How piece b continues piece a: (reverse_a, reverse_b, gap) of the closest end pair, or None."""
    best = None
    for ra, ea in ((False, a[-1]), (True, a[0])):
        for rb, sb in ((False, b[0]), (True, b[-1])):
            g = float(np.linalg.norm(ea[:2] - sb[:2]))
            if g <= tol and (best is None or g < best[2]):
                best = (ra, rb, g)
    return best


def chain_pieces(polylines, classes=CARLA_3CLS_CLASSES, gap_tol=0.15, min_len=0.0):
    """reference_lines['polylines'] -> list of dict(points (N, 3) float64, cls (str), is_arc,
    road_id), one per continuous stretch of one class along one road, in a deterministic order
    (classes in the given order, then roads and pieces in file order)."""
    groups = {}
    order = []
    for pl in polylines:
        c = pl.get('class')
        if c not in classes:
            continue
        p = np.asarray(pl.get('points', []), dtype=np.float64)
        if p.ndim != 2 or len(p) < 2:
            continue
        key = (c, pl.get('road_id'))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((p, bool(pl.get('is_arc', False)) or pl.get('type') == 'arc'))
    out = []
    for c in classes:
        for key in [k for k in order if k[0] == c]:
            pieces = list(groups[key])
            while pieces:
                cur, arc = pieces.pop(0)
                grown = True
                while grown:
                    grown = False
                    for i, (p, a) in enumerate(pieces):
                        j = _ends_join(cur, p, gap_tol)
                        if j is None:
                            continue
                        ra, rb, g = j
                        if ra:
                            cur = cur[::-1]
                        q = p[::-1] if rb else p
                        cur = np.vstack([cur, q[1:] if g < 1e-3 else q])
                        arc = arc or a
                        pieces.pop(i)
                        grown = True
                        break
                out.append(dict(points=np.ascontiguousarray(cur), cls=c, is_arc=arc,
                                road_id=key[1]))
    if min_len > 0:
        out = [o for o in out
               if float(np.linalg.norm(np.diff(o['points'][:, :2], axis=0), axis=1).sum()) >= min_len]
    return out


def tile_gt(ref_json_path, **kw):
    """Convenience: chained GT of one tile's reference_lines file."""
    with open(ref_json_path) as f:
        d = json.load(f)
    return chain_pieces(d.get('polylines', []), **kw)
