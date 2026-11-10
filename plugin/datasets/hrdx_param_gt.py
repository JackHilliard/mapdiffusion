"""Ground-truth curbs from the parametric design equations (numpy only, no mmdet).

Every HRDX tile carries its curbs twice: hand-drawn polylines (class 'curb') and the
alignment elements fitted to them (class 'curb_parametric': straight lines, circular arcs
and cubic B-splines with their parameters, clipped to the tile and sampled every 0.5 m ON
the equations). This module turns the elements into ground-truth polylines:

  * the elements of one alignment are ordered by station (sta_start);
  * consecutive elements are joined into one polyline when the station is continuous
    (|sta_start - previous sta_end| <= sta_tol) and their sampled ends touch (<= gap_tol);
    a jump means the alignment left the tile and came back, so a new polyline starts;
  * the joined points are the equation samples themselves (the shared joint point is kept
    once), so every vertex lies on the design curve.

One ground-truth instance = one continuous in-tile stretch of one design alignment.
Shared by the training dataset (projects/mmdet3d_plugin/datasets/hrdx50m_dataset.py) and
the split tooling, so both see exactly the same curbs.
"""
import json

import numpy as np


def chain_parametric(polylines, sta_tol=0.05, gap_tol=0.10, min_len=0.0):
    """reference_lines['polylines'] -> list of dict(points (N, 3) float64, is_arc, alignment_id,
    sta (start, end)) for the 'curb_parametric' elements, one per continuous stretch."""
    by_al = {}
    for pl in polylines:
        if pl.get('class') != 'curb_parametric':
            continue
        p = np.asarray(pl.get('points', []), dtype=np.float64)
        if p.ndim != 2 or len(p) < 2:
            continue
        by_al.setdefault(pl.get('alignment_id'), []).append((pl, p))
    out = []
    for al, els in by_al.items():
        els.sort(key=lambda t: (float(t[0]['sta_start']), int(t[0].get('element_index', 0))))
        cur, arc, s0, s_prev = None, False, None, None
        for pl, p in els:
            joins = (cur is not None
                     and abs(float(pl['sta_start']) - s_prev) <= sta_tol
                     and float(np.linalg.norm(cur[-1][:2] - p[0][:2])) <= gap_tol)
            if joins:
                same = float(np.linalg.norm(cur[-1][:2] - p[0][:2])) < 1e-3
                cur = np.vstack([cur, p[1:] if same else p])
            else:
                if cur is not None:
                    out.append(dict(points=cur, is_arc=arc, alignment_id=al, sta=(s0, s_prev)))
                cur, arc, s0 = p.copy(), False, float(pl['sta_start'])
            arc = arc or bool(pl.get('is_arc', False)) or pl.get('type') == 'arc'
            s_prev = float(pl['sta_end'])
        if cur is not None:
            out.append(dict(points=cur, is_arc=arc, alignment_id=al, sta=(s0, s_prev)))
    if min_len > 0:
        out = [o for o in out
               if float(np.linalg.norm(np.diff(o['points'][:, :2], axis=0), axis=1).sum()) >= min_len]
    return out


def tile_parametric_gt(ref_json_path, **kw):
    """Convenience: chained parametric curbs of one tile's reference_lines file."""
    with open(ref_json_path) as f:
        d = json.load(f)
    return chain_parametric(d.get('polylines', []), **kw)
