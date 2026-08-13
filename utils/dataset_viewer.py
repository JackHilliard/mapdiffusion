"""Interactive CARLA dataset viewer: browse tiles by town, render top-down
views of the raw LiDAR point cloud, a density heat map, and/or the
reference-line polylines.

Reads the dataset directly (manifest.json / grid_manifest.json +
blocks/*.npz + reference_lines/*.json) -- no model, no checkpoint, no GPU,
and no mmdet3d/torch import at all. Only needs `flask`, `matplotlib`,
`numpy`.

Usage (from the repo root):
  pip install flask matplotlib numpy
  # split-structured dataset (<root>/{train,val,test}/manifest.json)
  python3 utils/dataset_viewer.py --data-root /path/to/carla
  # single-town grid export (<root>/<Town>/grid_tiles/grid_manifest.json)
  python3 utils/dataset_viewer.py \
      --data-root /gel/usr/johil9/Documents/carla/Town10HD/grid_tiles

Then open http://127.0.0.1:5001

--- Two manifest layouts ---
Both are found by the same bounded-depth scan (--data-root itself, plus up
to two directory levels below it), and both put `blocks/` and
`reference_lines/` next to their manifest, so only the metadata differs:

  * `manifest.json`      -- the split-level manifest. Has `split`, `towns`,
                            `tiles_per_town`, and town-prefixed tile names
                            (`town10hd_tile_00000`) carrying their own
                            `town` key.
  * `grid_manifest.json` -- a single town's grid export. NONE of the above:
                            no split, no towns, no per-tile `town`, and
                            bare `tile_00000` names. The town is inferred
                            from `town_ply`'s filename (or the containing
                            directory), and the split from the directory
                            name.

Because grid tile names are only unique *within* their own directory, every
tile is keyed internally by `<dataset key>/<name>`, not by name alone --
two towns' grid exports both contain `tile_00000`.

Where a directory contains both files (the grid exporter writes
`grid_manifest.json`, a later packaging step adds `manifest.json`),
`manifest.json` wins and `grid_manifest.json` only backfills keys it is
missing. Counts are always recomputed from the tile list rather than
trusted: the packaged `manifest.json` in a grid_tiles dir was observed
carrying a stale dataset-wide `n_tiles` (4103) alongside its own 30 tiles.

--- Polyline classes ---
Newer exports label every reference line with a class (`class_id` /
`class`, decoded by the manifest's `class_lookup`, e.g. driving_centerline
/ curb / road_edge / lane_divider / ...). Polylines are coloured by class,
and the class checkboxes filter which are drawn. Older exports have no
class field at all; those render as a single unclassified red set, exactly
as before.

--- Coordinate frames (important) ---
Each tile's .npz stores two different origins, and they are NOT the same:
  * `offset`      -- what `features[:, 0:3]` is actually relative to
                     (verified: points - offset == features xyz, exactly)
  * `tile_center` -- the geometric centre of the tile
The reference_lines/*.json polylines need one of these subtracted to land
in the same frame as the points, and the two differ by a mean of 1.6 m
(max 12.1 m) on the 25 m export, growing with tile size.

Only the RELATIVE choice matters for alignment, and both are fine there so
long as the same shift is applied to the points and the polylines: measured
against real driving-surface returns (label == 0), GT sits a median
0.038 m from the nearest road point either way. Shifting only one of them
is the mistake -- that gives 0.388 m and silently suppresses chamfer AP.

This viewer renders in the **tile-centred** frame, because that is what
this repo trains in: MapDiffusion normalizes map coordinates about the ROI
centre and its head's reference points are sigmoid-bounded, with no clamp
anywhere, so a tile that does not sit on the origin has GT outside the ROI
that the model cannot represent (13.5% of vertices, max 18.6 m, in the
offset frame). tools/data_converter/carla_converter.py subtracts
`tile_center` from the GT and records `tile_shift` (== tile_center -
offset) per sample; LoadCarlaPointsFromFile subtracts `tile_shift` from
`features[:, 0:3]`. This file does both, so points, GT and predictions all
land in one frame and the tile is centred on (0, 0).

The sibling MapTRv2/GeMap trees keep the offset frame and absorb the
overhang downstream by clamping, which is why their copy of this viewer
renders with `offset` and centres the axes on `tile_center - offset`.

World z is the one thing that still needs `offset`: `features` z is
centroid-relative, so the deep scan adds `offset[2]` back, never
`tile_center[2]`.

--- Three tabs ---
`?tab=browse` (the default) is the per-tile viewer described above.
`?tab=stats` answers the other question -- what does the *dataset* look
like -- in two tiers, because their costs differ by three orders of
magnitude:

  * manifest tier -- points/tile, points/m^2, polylines/tile and per-class
    polyline counts, all straight off the tile entries already in memory.
    Zero I/O; always shown.
  * deep tier -- effective (non-degenerate) point count, z-range, per-point
    label mix, |tile_center - offset| drift, and polyline vertex counts and
    arc lengths. Needs every tile's .npz (~12 GB for the 4103-tile train
    split), so it is opt-in behind the "Deep scan" button, runs in a
    background thread with a progress readout, and is cached to disk so it
    only ever runs once per dataset.

The deep tier's cache lives under ~/.cache/mapdiffusion_dataset_viewer/
(override
with --stats-cache), never inside the dataset directory -- that is
bind-mounted read-only in the container workflow.

Why these particular statistics: each one corresponds to a failure this
project has actually hit. `effective vs raw points` exposes the 18
degenerate tiles whose manifest says 5,000,000 points but which hold ~3,200
distinct ones; `near-empty` and `no-GT` find the tiles that crash
extract_lidar_feat with a bare IndexError once voxelization returns zero
voxels; `z-range` finds tiles pushed outside lidar_point_cloud_range;
`drift` is the canary for the GT-frame bug (it should sit near zero now
that the converter subtracts `offset`); the per-class counts are what a
decision to widen the divider-only taxonomy has to be made from; and the
split overlay shows whether train and test are even the same distribution
(they are different towns, so this is not rhetorical).

`?tab=results` scores a training run's predictions per tile and asks what
the failures have in common. Needs `--work-dir` (for a results json) and
an eval GT, which can be EITHER of two files -- they give identical GT:

  * `carla_map_infos_<split>.pkl`, the converter's output. Preferred and
    auto-selected, because it exists as soon as the dataset is converted.
  * `carla_map_gt.json`, the sibling MapTRv2/GeMap format, written by
    their `_format_gt()`. This repo never writes one -- it is read only so
    an existing file from those trees can be used.

Note the pkl's own frame is detected per sample rather than assumed --
this repo and the sibling trees write the same schema in different frames.
See gt_frame_shift().

Building GT from the pkl reproduces `_format_gt()` rather than
approximating it, because that chain is almost an identity:
`gen_vectorized_samples()` wraps each annotation array in a LineString and
keeps it if it has >=2 points and a class mapping to a label != -1;
`LiDARInstanceLines` stores that list untouched; `_format_gt` writes
`np.array(list(gt_vec.coords))[:, :code_size]`. No resampling, no clipping
to pc_range, no reordering. Verified by reproducing an existing
carla_map_gt.json exactly -- 259 tiles, 1210 instances, every coordinate
bit-identical, and identical end-to-end mAP.

Which GT covers which predictions is settled by token overlap, not by
order: splits are disjoint sets of tiles, and nothing in a results json
says which split it came from.

  * It reimplements the repo's chamfer matching in numpy, because this
    viewer runs on the HOST: no torch, no mmdet3d, and neither shapely nor
    scipy. Verified bit-exact against the real implementation -- identical
    TP/FP vectors on all 259 test tiles, and an identical global mAP.
  * The useful structural fact: mean_ap.eval_map() already calls
    custom_tpfp_gen() once PER TILE, and only the AP aggregation is global.
    So the per-tile TP/FP here IS the eval's; the only addition is running
    the AP formula over one tile's detections.
  * "Per-tile AP" is therefore a LOCAL score -- that tile's own ranking
    against its own GT -- not its contribution to the global AP, which is
    not a well-defined per-tile quantity (global ranking interleaves
    detections from every tile). The global number is computed too, with
    the real interleaved ranking, and shown in the header so it can be
    checked against the training log's CarlaMap_chamfer/mAP.
  * Tiles are ranked by any of a dozen metrics and sampled from the top,
    middle and bottom 20% of that ranking. The sample is seeded, so it
    survives a reload but re-rolls on demand; specific tiles can be pinned.
  * Polyline COUNT error (predicted - GT, signed) is tracked separately
    from matched-instance count: "did it find the right number of lines"
    and "did it put them in the right place" are different failures. It
    only exists relative to a score threshold, because the head emits a
    fixed num_vec every tile -- so the threshold is an eval parameter here,
    and count_health() says so out loud when no threshold can separate
    anything (which is exactly what an undertrained checkpoint looks like).
  * The scatters mark tiles the statistics tab flags as suspect in the
    status colour, which is what makes "are the bad tiles just the broken
    ones?" answerable by looking.
"""
import argparse
import csv
import hashlib
import html
import io
import json
import os
import os.path as osp
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib.parse import urlencode

import matplotlib
matplotlib.use('Agg')
# Rendering matplotlib inside a threaded web server needs BOTH of these:
#
#  1. The object-oriented Figure/FigureCanvasAgg API, never pyplot --
#     pyplot is a global state machine and is not thread-safe.
#  2. RENDER_LOCK below, serializing every render.
#
# Neither alone was sufficient here: Flask's dev server is threaded by
# default and a browser requests every tile image on a page concurrently,
# which segfaulted the process outright (no traceback -- SIGSEGV). Switching
# to the OO API still crashed under a concurrent-request stress test; only
# adding the lock made it survive. Sequential curl requests never reproduce
# it, so test this path with genuinely parallel requests if you touch it.
# Rendering is fast enough (~0.1-1 s/tile) that serializing a page's worth
# of images is fine for a local viewer.
import numpy as np
import matplotlib.patheffects as pe
from flask import Flask, abort, make_response, redirect, request, send_file
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

app = Flask(__name__)
STATE = {}
RENDER_LOCK = threading.Lock()

BG = '#0d1117'
BG_PANEL = '#161b22'
BORDER = '#30363d'
TEXT = '#c9d1d9'
TEXT_MUTED = '#8b949e'
ACCENT = '#58a6ff'

# ITU-R BT.709 luma, matching LoadCarlaPointsFromFile's rgb->strength
RGB2GRAY = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# label ids come from manifest['lane_type_lookup']; -1 == unlabeled
LABEL_COLORS = {
    -1: '#484f58', 0: '#58a6ff', 1: '#d29922', 2: '#3fb950', 3: '#bc8cff',
    4: '#f85149', 5: '#39c5cf', 6: '#db6d28', 7: '#e3b341', 8: '#8b949e',
}

# Polyline class ids come from the manifest's `class_lookup`. Keyed by id
# rather than by name so an export that renames a class still renders; the
# UNCLASSED colour is what older, class-free exports use for everything.
CLASS_COLORS = {
    0: '#58a6ff', 1: '#f0883e', 2: '#f85149', 3: '#3fb950',
    4: '#bc8cff', 5: '#39c5cf', 6: '#e3b341', 7: '#ff7ac8',
}
CLASS_FALLBACK = '#d2a8ff'
UNCLASSED_COLOR = '#f85149'
# ...except over the density heat map, where red/orange vanish into
# inferno's own mid-range. Only the unclassified case gets swapped (a
# single colour, so a single substitute works); classed polylines keep
# their palette and rely on the black outline stroke for contrast.
UNCLASSED_COLOR_DENSITY = '#00e5ff'

# 'inferno' rather than a hand-rolled dark ramp: an earlier version started
# the colormap at the page background colour (#0d1117), which made every
# low-density cell literally indistinguishable from empty space -- the heat
# map looked blank. inferno is perceptually uniform and its low end is a
# visible dark purple, so occupied-but-sparse cells still read as occupied.
DENSITY_CMAP = 'inferno'


MANIFEST_NAMES = ('manifest.json', 'grid_manifest.json')
# directories that can hold thousands of files and never a manifest
PRUNE_DIRS = {'blocks', 'reference_lines'}


def discover_datasets(data_root, only=None, max_depth=2):
    """Find every manifest under data_root, at data_root itself or up to
    `max_depth` directories below it. Returns {key: manifest}, where key is
    the manifest directory's path relative to data_root (or its basename
    when the manifest sits in data_root itself).

    Both manifest names are accepted; if a directory has both, see the
    module docstring for why manifest.json wins. `only` filters on either
    the key or the manifest's own split name, so `--split test` still works
    on the split layout and `--split grid_tiles` works on a grid export.
    """
    data_root = data_root.rstrip('/')
    found = {}
    for dirpath, dirnames, filenames in os.walk(data_root):
        rel = osp.relpath(dirpath, data_root)
        depth = 0 if rel == '.' else rel.count(os.sep) + 1
        dirnames[:] = ([] if depth >= max_depth else
                       sorted(d for d in dirnames if d not in PRUNE_DIRS))
        present = [n for n in MANIFEST_NAMES if n in filenames]
        if not present:
            continue
        dirnames[:] = []  # a manifest owns its subtree; don't nest datasets
        man = {}
        # reversed so the preferred name (first in MANIFEST_NAMES) is
        # applied last and its keys win, while the other only backfills
        for name in reversed(present):
            with open(osp.join(dirpath, name)) as f:
                man.update({k: v for k, v in json.load(f).items()
                            if v is not None or k not in man})
        key = osp.basename(data_root) if rel == '.' else rel
        man = normalize_manifest(man, key, dirpath)
        if only and only not in (key, man['split']):
            continue
        found[key] = man
    order = {'train': 0, 'val': 1, 'test': 2}
    return dict(sorted(found.items(),
                        key=lambda kv: (order.get(kv[1]['split'], 99), kv[0])))


def infer_town(man, dirpath):
    """Best-effort town name for a manifest that has no `towns` list --
    i.e. a grid_manifest. `town_ply` points at the source cloud
    (.../Town10HD/Town10HD_full.ply), whose stem names the town; failing
    that, use the containing directory (a grid export lives in
    <Town>/grid_tiles/, so prefer the parent of a generic-looking dir)."""
    ply = man.get('town_ply') or man.get('reference_lines_ply')
    if ply:
        stem = osp.splitext(osp.basename(ply))[0]
        for suffix in ('_full_reference_lines', '_full'):
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
        if stem:
            return stem
    here = osp.basename(dirpath.rstrip('/'))
    if here in ('grid_tiles', 'tiles', 'train', 'val', 'test'):
        parent = osp.basename(osp.dirname(dirpath.rstrip('/')))
        if parent:
            return parent
    return here or 'unknown'


def normalize_manifest(man, key, dirpath):
    """Fill in everything the rest of the viewer assumes exists, so a
    grid_manifest and a split manifest are interchangeable downstream.

    Counts (`n_tiles`, `tiles_per_town`) are always recomputed from the
    tile list rather than read: a packaged manifest.json inside a
    grid_tiles dir was seen reporting the whole dataset's 4103 tiles while
    listing only its own 30.
    """
    man = dict(man)
    man['dir'] = dirpath
    man['key'] = key
    man['split'] = man.get('split') or key
    tiles = [dict(t) for t in man.get('tiles', [])]
    towns = man.get('towns') or []
    if not towns:
        towns = sorted({t['town'] for t in tiles if t.get('town')})
    default_town = towns[0] if len(towns) == 1 else None
    if not towns:
        default_town = infer_town(man, dirpath)
        towns = [default_town]
    counts = {}
    for t in tiles:
        # grid tiles carry no `town` key at all; a one-town manifest can
        # supply it, a multi-town one without per-tile towns cannot, so
        # those fall into a single bucket rather than being invented
        t.setdefault('town', default_town or '(all)')
        counts[t['town']] = counts.get(t['town'], 0) + 1
    man['tiles'] = tiles
    man['towns'] = sorted(counts) or towns
    man['tiles_per_town'] = counts
    man['n_tiles'] = len(tiles)
    man.setdefault('class_lookup', {})
    man.setdefault('lane_type_lookup', {})
    return man


def merged_lookup(datasets, field):
    """Union of one id->name lookup across every loaded manifest. Ids are
    consistent between exports of the same generation, so first writer
    wins; a genuine conflict would mean two incompatible taxonomies, which
    is a data problem, not something to paper over here."""
    out = {}
    for man in datasets.values():
        for k, v in (man.get(field) or {}).items():
            out.setdefault(str(k), v)
    return out


def class_choices(datasets):
    """(key, label) pairs for the class-filter checkboxes: every class in
    the merged lookup, plus an 'unclassified' entry if any loaded dataset
    predates the class field (otherwise its polylines would be filtered out
    with no way to switch them back on)."""
    lookup = merged_lookup(datasets, 'class_lookup')
    out = [(str(k), lookup[str(k)])
           for k in sorted(lookup, key=lambda s: int(s) if s.isdigit() else s)]
    if any(not man.get('class_lookup') for man in datasets.values()):
        out.append(('none', 'unclassified'))
    return out


def class_summary(datasets):
    """One-line 'what classes exist and how many of each' for the header."""
    counts = {}
    for man in datasets.values():
        for name, n in (man.get('polyline_counts_by_class') or {}).items():
            counts[name] = counts.get(name, 0) + n
    if not counts:
        return ('polyline classes: none declared &mdash; reference lines '
                'render as a single unclassified set')
    return 'polyline classes: ' + ', '.join(
        f'<b>{html.escape(k)}</b> ({v:,})' for k, v in sorted(counts.items()))


def build_index(datasets):
    """Flatten every dataset's tiles into one list and group them by
    (dataset, town) for the picker.

    Tiles get a `_uid` of '<dataset key>/<name>' because bare names are NOT
    globally unique across grid exports -- every town's grid_tiles has its
    own tile_00000, and keying by name alone would make one shadow the
    other in the lookup table.
    """
    tiles, groups = [], {}
    for key, man in datasets.items():
        # tile_radius/tile_side are manifest-level, not per-tile, and
        # normalize_manifest doesn't copy them down -- this is the only place
        # a manifest is in scope alongside its own tiles. The area is what
        # makes point counts comparable ACROSS exports: the legacy split is
        # 25 m tiles and the grid export is 60 m ones, so points/tile is not
        # comparable between them while points/m^2 is.
        side = man.get('tile_side')
        if not side:
            side = 2.0 * float(man.get('tile_radius') or 12.5)
        area = float(side) ** 2
        for t in man['tiles']:
            t = dict(t)
            t['_ds'] = key
            t['_split'] = man['split']
            t['_area'] = area
            t['_uid'] = f'{key}/{t["name"]}'
            gkey = f'{key}/{t["town"]}'
            t['_group'] = gkey
            # position within its own group, i.e. exactly the `start` index
            # the browse tab slices with -- what lets the stats tab link
            # straight to a tile it has flagged
            t['_gidx'] = groups.get(gkey, {}).get('count', 0)
            tiles.append(t)
            if gkey not in groups:
                groups[gkey] = {'ds': key, 'split': man['split'],
                                'town': t['town'], 'count': 0}
            groups[gkey]['count'] += 1
    return tiles, groups


def tile_by_uid(uid):
    return STATE['tiles_by_uid'].get(uid)


def ds_dir(ds):
    man = STATE['datasets'].get(ds)
    return man['dir'] if man else None


def load_block(name, ds):
    d = ds_dir(ds)
    path = osp.join(d, 'blocks', f'{name}.npz') if d else None
    if not path or not osp.isfile(path):
        return None
    return np.load(path)


def load_polylines(name, origin, ds):
    """Returns list of (pts, class_id, class_name), where pts is an (N,2)
    array in the same frame as the block's `features` xyz (given the origin
    to subtract -- see module docstring).

    class_id is None for older exports whose polylines carry no class
    field; those callers get a single unclassified set.
    """
    d = ds_dir(ds)
    path = osp.join(d, 'reference_lines',
                     f'{name}_reference_lines.json') if d else None
    if not path or not osp.isfile(path):
        return []
    with open(path) as f:
        rl = json.load(f)
    # per-tile `classes` overrides the manifest's lookup where present --
    # it's written by the same pass that assigned the ids
    lookup = dict(STATE.get('class_lookup', {}))
    lookup.update(rl.get('classes') or {})
    out = []
    for p in rl['polylines']:
        pts = np.asarray(p['points'], dtype=np.float32)
        if pts.shape[0] < 2:
            continue
        cid = p.get('class_id')
        cid = int(cid) if cid is not None else None
        cname = p.get('class') or lookup.get(str(cid)) or (
            'unclassified' if cid is None else f'class {cid}')
        out.append((pts[:, :2] - origin[:2], cid, cname))
    return out


def discover_results(work_dir):
    """Find every prediction json under a training work_dir.

    In this repo both writers land in the same place:
    tools/{train,test}.py set `cfg.data.{val,test}.work_dir = cfg.work_dir`,
    and BaseMapDataset.format_results() writes
    `<work_dir>/submission_vector.json`. So a training run's periodic eval
    and a tools/test.py run produce the SAME path, and each eval overwrites
    the last -- there is no timestamped history to browse. Copy the file
    aside if you want to compare two checkpoints.

    Note `evaluation.jsonfile_prefix`, which mmdet_train.py sets to
    `val/<work_dir>/<ctime>`, is inert here: it reaches
    BaseMapDataset.evaluate() as an unused **kwargs entry, because
    format_results() takes its own `prefix=self.work_dir`. The `val/` root
    is still searched so MapTRv2-style outputs are picked up if present.

    Returns {label: path}, newest first.
    """
    found = {}
    roots = [work_dir, osp.join('val', work_dir)]
    for root in roots:
        if not osp.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                # 'submission' is MapDiffusion's name, 'result' MapTRv2's.
                low = fn.lower()
                if fn.endswith('.json') and ('result' in low
                                             or 'submission' in low):
                    p = osp.join(dirpath, fn)
                    label = osp.relpath(p, work_dir if root == work_dir
                                         else osp.dirname(work_dir))
                    found[label] = p
    return dict(sorted(found.items(),
                        key=lambda kv: osp.getmtime(kv[1]), reverse=True))


def load_results(path):
    """Parse a predictions json into {sample_token: [(pts, score, cls)]}.

    Handles both shapes this project can produce, because they are not the
    same file:

      * MapDiffusion's own `submission_vector.json`, written by
        BaseMapDataset.format_results(). `results` is a DICT keyed by token,
        and each entry holds three parallel lists::

            {"results": {"<token>": {"vectors": [[[x, y], ...], ...],
                                     "scores":  [0.83, ...],
                                     "labels":  [0, ...]}}}

      * the MapTRv2-style `carlamap_results.json` this viewer was ported
        from, where `results` is a LIST of per-sample entries and each
        vector is a dict with `pts` / `confidence_level` / `cls_name`.

    Predicted points are already in the tile-centred metric frame:
    format_results() denormalizes them from (0, 1) with
    `vector * (roi_size + eps) + origin`, `origin = -roi_size/2`. That is
    the frame `_render_tile` draws in and the frame the GT is in, so they
    are plotted and scored as-is, with no shift.
    """
    cached = STATE['results_cache'].get(path)
    if cached is not None:
        return cached
    with open(path) as f:
        blob = json.load(f)
    results = blob.get('results', [])
    out = {}

    if isinstance(results, dict):  # MapDiffusion submission_vector.json
        for token, entry in results.items():
            vecs = []
            vectors = entry.get('vectors', [])
            scores = entry.get('scores', [])
            labels = entry.get('labels', [])
            for i, v in enumerate(vectors):
                pts = np.asarray(v, dtype=np.float32)
                if pts.ndim != 2 or pts.shape[0] < 2:
                    continue
                label = int(labels[i]) if i < len(labels) else 0
                vecs.append((
                    pts[:, :2],
                    float(scores[i]) if i < len(scores) else 1.0,
                    MAPCLASSES[label] if label < len(MAPCLASSES) else str(label),
                ))
            out[token] = vecs
    else:  # MapTRv2-style list
        for entry in results:
            token = entry.get('sample_token')
            vecs = []
            for v in entry.get('vectors', []):
                pts = np.asarray(v.get('pts', []), dtype=np.float32)
                if pts.ndim != 2 or pts.shape[0] < 2:
                    continue
                vecs.append((pts[:, :2],
                              float(v.get('confidence_level', 1.0)),
                              v.get('cls_name', '?')))
            out[token] = vecs

    STATE['results_cache'][path] = out
    return out


def class_key(cid):
    """Stable string key for a class id, used in query strings and in the
    class-filter set. Class-free polylines get 'none' rather than being
    lumped in with class 0."""
    return 'none' if cid is None else str(int(cid))


def class_color(cid, mode='rgb'):
    if cid is None:
        return UNCLASSED_COLOR_DENSITY if mode == 'density' else UNCLASSED_COLOR
    return CLASS_COLORS.get(int(cid), CLASS_FALLBACK)


def style_axes(ax, center, radius):
    ax.set_facecolor(BG)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect('equal')
    ax.tick_params(colors=TEXT_MUTED, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(BORDER)


def render_tile(*args, **kwargs):
    """Thread-safe wrapper -- see RENDER_LOCK's comment at the top."""
    with RENDER_LOCK:
        return _render_tile(*args, **kwargs)


def tile_shift_of(block):
    """`tile_center - offset`, the shift from the block's stored frame into
    the tile-centred one this repo trains in.

    Named to match the `tile_shift` the converter records on every sample
    and that LoadCarlaPointsFromFile subtracts. Returns zeros for a block
    that predates either key, which just leaves the old offset-frame
    behaviour.
    """
    if 'tile_center' not in block or 'offset' not in block:
        return np.zeros(3, dtype=np.float32)
    return (np.asarray(block['tile_center'], dtype=np.float32)
            - np.asarray(block['offset'], dtype=np.float32))


def _render_tile(name, mode, show_polylines,
                  point_size, max_points, log_density=True, ds=None,
                  results_path=None, score_thresh=0.3, classes=None,
                  top_n=None):
    block = load_block(name, ds)
    if block is None:
        return None
    split = STATE['datasets'][ds]['split'] if ds in STATE['datasets'] else ds
    feat = block['features']
    labels = block['labels'] if 'labels' in block else None
    radius = float(block['tile_radius']) if 'tile_radius' in block else 12.5

    # `features[:, 0:3]` is stored relative to `offset`; shift it by
    # `tile_shift` so the points land in the same tile-centred frame as the
    # GT and the predictions below -- exactly what LoadCarlaPointsFromFile
    # does before the model ever sees them.
    tile_shift = tile_shift_of(block)
    xy = feat[:, :2] - tile_shift[:2]

    # `tile_center`, not the block's `offset` -- this is where this viewer
    # diverges from the MapTRv2 original it was ported from, because this
    # repo's converter diverges the same way. Both frames keep the polylines
    # and the points mutually aligned (they differ by one rigid shift), but
    # MapDiffusion normalizes map coordinates about the ROI centre and its
    # head's reference points are sigmoid-bounded, so the tile has to sit on
    # the origin: tools/data_converter/carla_converter.py subtracts
    # tile_center from the GT, and LoadCarlaPointsFromFile subtracts
    # `tile_shift` (== tile_center - offset) from features[:, 0:3].
    #
    # Rendering in that same frame is what lets predictions, which come out
    # of the model tile-centred, be overlaid on the points without a shift.
    origin = (block['tile_center'] if 'tile_center' in block
              else block['offset'])
    # The tile IS centred on (0, 0) in this frame, unlike the offset frame
    # the original had to correct for.
    center = np.zeros(2, dtype=np.float32)

    fig = Figure(figsize=(6, 6))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    fig.patch.set_facecolor(BG)
    style_axes(ax, center, radius)

    n_raw = xy.shape[0]
    subsampled = False
    if mode != 'density' and n_raw > max_points:
        # Only for scatter rendering -- density uses ALL points so the
        # histogram stays quantitatively correct.
        sel = np.random.default_rng(0).choice(n_raw, max_points, replace=False)
        xy_plot = xy[sel]
        labels_plot = labels[sel] if labels is not None else None
        subsampled = True
    else:
        xy_plot = xy
        labels_plot = labels

    if mode == 'density':
        # points per 1 m^2 cell -- uses every point, never the subsample,
        # so the counts stay quantitatively correct
        nbins = max(1, int(round(2 * radius)))
        xedges = np.linspace(center[0] - radius, center[0] + radius, nbins + 1)
        yedges = np.linspace(center[1] - radius, center[1] + radius, nbins + 1)
        H, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=[xedges, yedges])
        # Log norm by default: density spans a huge dynamic range on this
        # dataset (the degenerate 5,000,000-point tiles put ~99.9% of their
        # points in a single cell, so a linear scale maps every other cell
        # to the colormap's zero end and the map reads as blank).
        norm = None
        if log_density and H.max() > 0:
            from matplotlib.colors import LogNorm
            norm = LogNorm(vmin=max(H[H > 0].min(), 1) if (H > 0).any() else 1,
                            vmax=max(H.max(), 2))
        im = ax.imshow(H.T, origin='lower',
                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                        cmap=DENSITY_CMAP, aspect='equal', norm=norm,
                        interpolation='nearest')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('points / m²' + (' (log)' if norm is not None else ''),
                      color=TEXT, fontsize=8)
        cb.ax.tick_params(colors=TEXT_MUTED, labelsize=7)
        cb.outline.set_edgecolor(BORDER)
        occ = int((H > 0).sum())
        title_extra = (f'max {H.max():,.0f} pts/m², median {np.median(H):,.0f}, '
                        f'{occ}/{H.size} cells occupied')
    elif mode == 'rgb':
        # features[:, 3:6] are per-point RGB in 0..1
        rgb = np.clip(feat[:, 3:6], 0.0, 1.0)
        rgb_plot = rgb[sel] if subsampled else rgb
        ax.scatter(xy_plot[:, 0], xy_plot[:, 1], c=rgb_plot, s=point_size,
                    linewidths=0)
        title_extra = 'true RGB colour'
    elif mode == 'intensity':
        strength = feat[:, 3:6] @ RGB2GRAY
        s_plot = strength[sel] if subsampled else strength
        ax.scatter(xy_plot[:, 0], xy_plot[:, 1], c=s_plot, s=point_size,
                    cmap='cividis', linewidths=0, alpha=0.9)
        title_extra = 'intensity (BT.709 luma of RGB)'
    elif mode == 'label' and labels_plot is not None:
        for lab in np.unique(labels_plot):
            msk = labels_plot == lab
            lname = STATE['lane_types'].get(str(int(lab)), 'unlabeled'
                                             if lab == -1 else str(lab))
            ax.scatter(xy_plot[msk, 0], xy_plot[msk, 1],
                        s=point_size, linewidths=0, alpha=0.9,
                        c=LABEL_COLORS.get(int(lab), '#8b949e'),
                        label=f'{lname} ({msk.sum():,})')
        leg = ax.legend(fontsize=6, loc='upper right', facecolor=BG_PANEL,
                         edgecolor=BORDER, labelcolor=TEXT, framealpha=0.9,
                         markerscale=3)
        leg.get_frame().set_linewidth(0.5)
        title_extra = 'coloured by lane label'
    else:  # 'points', and 'label' on a block with no labels array
        ax.scatter(xy_plot[:, 0], xy_plot[:, 1], s=point_size,
                    c='#58a6ff', linewidths=0, alpha=0.6)
        title_extra = ('top-down' if mode != 'label'
                        else 'top-down (no labels in this block)')

    outline = [pe.Stroke(linewidth=3.0, foreground='#000000', alpha=0.6),
                pe.Normal()]
    overlay_handles = []

    if show_polylines:
        gt = load_polylines(name, origin, ds)
        drawn = {}
        for pl, cid, cname in gt:
            if classes is not None and class_key(cid) not in classes:
                continue
            ax.plot(pl[:, 0], pl[:, 1], color=class_color(cid, mode),
                     linewidth=1.8, alpha=0.98, zorder=5,
                     path_effects=outline)
            drawn[(cid, cname)] = drawn.get((cid, cname), 0) + 1
        for (cid, cname), n in sorted(
                drawn.items(),
                key=lambda kv: (kv[0][0] is None, kv[0][0] or 0, kv[0][1])):
            label = f'GT {cname} ({n})' if cid is not None else f'GT ({n})'
            overlay_handles.append(
                Line2D([], [], color=class_color(cid, mode), linewidth=1.8,
                        label=label))

    if results_path:
        # Predictions are deliberately styled to be unmistakable against
        # GT: dashed, thicker, drawn on top (higher zorder). They keep the
        # class palette so a multi-class model's class errors are visible;
        # a class-free result set stays the original bright yellow.
        preds = load_results(results_path).get(name, [])
        pred_drawn = {}
        kept = []
        for pts, score, cls in preds:
            if score < score_thresh:
                continue
            cid = STATE['class_ids'].get(cls)
            if classes is not None and cid is not None \
                    and class_key(cid) not in classes:
                continue
            kept.append((pts, score, cls, cid))
        # Keep only the n highest-scoring predictions, if asked. An absolute
        # score threshold cannot do this job on its own: the model emits a
        # fixed num_vec predictions every time, and their calibration moves
        # as training progresses (a 1-epoch checkpoint tops out around 0.17,
        # so 0.3 hides everything while 0.1 draws all 50 and buries the GT).
        # A count is stable across that; `top_n=len(GT)` in particular asks
        # "the model's best few guesses, as many as there are real lines".
        n_trimmed = 0
        if top_n is not None and len(kept) > top_n:
            kept.sort(key=lambda k: -k[1])
            n_trimmed = len(kept) - top_n
            kept = kept[:top_n]
        for pts, score, cls, cid in kept:
            ax.plot(pts[:, 0], pts[:, 1],
                     color=class_color(cid, mode) if cid is not None else '#ffd60a',
                     linewidth=2.0, alpha=0.95, zorder=6, linestyle='--',
                     path_effects=outline)
            pred_drawn[(cid, cls)] = pred_drawn.get((cid, cls), 0) + 1
        # The legend has to say what was hidden, or a trimmed view is
        # indistinguishable from a model that only made a few predictions.
        cut = (f', top {top_n} of {top_n + n_trimmed} by score'
               if n_trimmed else '')
        if not pred_drawn:
            overlay_handles.append(
                Line2D([], [], color='#ffd60a', linewidth=2.0, linestyle='--',
                        label=f'pred (0 @ score≥{score_thresh:g})'))
        for (cid, cls), n in sorted(pred_drawn.items(),
                                     key=lambda kv: str(kv[0][1])):
            overlay_handles.append(
                Line2D([], [],
                        color=class_color(cid, mode) if cid is not None else '#ffd60a',
                        linewidth=2.0, linestyle='--',
                        label=f'pred {cls} ({n} @ score≥{score_thresh:g}'
                              f'{cut})'))

    if overlay_handles:
        # ax.legend() REPLACES any existing legend, so the label-mode class
        # legend has to be re-added as a standalone artist first, otherwise
        # adding this overlay legend silently removes it.
        existing = ax.get_legend()
        if existing is not None:
            existing.set_zorder(20)
            ax.add_artist(existing)
        leg2 = ax.legend(handles=overlay_handles, fontsize=6, loc='lower left',
                          facecolor=BG_PANEL, edgecolor=BORDER,
                          labelcolor=TEXT, framealpha=0.95)
        leg2.get_frame().set_linewidth(0.5)
        # legends default to zorder 5, but predictions are drawn at 6 and
        # would otherwise scribble straight over the legend box
        leg2.set_zorder(20)

    sub_note = f', showing {max_points:,}' if subsampled else ''
    ax.set_title(f'{name}  [{split}]\n{n_raw:,} pts{sub_note} — {title_extra}',
                  color=TEXT, fontsize=9)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, facecolor=BG)
    # no plt.close() needed -- Figure created via the OO API isn't held by
    # pyplot's global registry, so it's garbage-collected normally
    buf.seek(0)
    return buf


# The stylesheet is shared by both tabs, so it is formatted ONCE here into
# a plain string and passed to each page template as a {css} value. That is
# safe despite this file's doubled-brace convention because str.format never
# re-scans a substituted *value* -- only the template itself. So STYLE_TMPL
# keeps its doubled braces and CSS, below, does not need any.
STYLE_TMPL = """
  :root {{ color-scheme: dark; }}
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 1.5em;
          background: {bg}; color: {text}; }}
  h1 {{ font-size: 1.25em; margin: 0 0 0.2em; }}
  h2 {{ font-size: 0.95em; margin: 1.8em 0 0.6em; font-weight: 600; }}
  .sub {{ color: {muted}; font-size: 0.85em; margin-bottom: 1.2em; }}
  form {{ background: {panel}; border: 1px solid {border}; border-radius: 6px;
          padding: 1em; margin-bottom: 1.5em; display: flex;
          flex-wrap: wrap; gap: 1.2em; align-items: flex-end; }}
  fieldset {{ border: none; margin: 0; padding: 0; }}
  label.top {{ display: block; font-size: 0.7em; text-transform: uppercase;
               letter-spacing: 0.06em; color: {muted}; margin-bottom: 0.35em; }}
  select, input[type=number] {{
    background: {bg}; color: {text}; border: 1px solid {border};
    border-radius: 4px; padding: 0.4em 0.5em; font-size: 0.9em; }}
  .checks label {{ display: block; font-size: 0.85em; margin: 0.15em 0; }}
  button {{ background: {accent}; color: #0d1117; border: none;
            border-radius: 4px; padding: 0.55em 1.4em; font-size: 0.9em;
            font-weight: 600; cursor: pointer; }}
  button.ghost {{ background: transparent; color: {accent};
                  border: 1px solid {border}; }}
  .gallery {{ display: flex; flex-wrap: wrap; gap: 12px; }}
  figure {{ margin: 0; background: {panel}; border: 1px solid {border};
            border-radius: 6px; padding: 8px; }}
  figure img {{ display: block; width: 420px; border-radius: 3px; }}
  figcaption {{ font-size: 0.72em; color: {muted}; margin-top: 5px;
                text-align: center; word-break: break-all; }}
  a {{ color: {accent}; }}
  nav {{ display: flex; gap: 0.4em; margin: 0.9em 0 1.2em;
         border-bottom: 1px solid {border}; }}
  nav a {{ padding: 0.45em 1.1em; font-size: 0.85em; text-decoration: none;
           color: {muted}; border: 1px solid transparent;
           border-bottom: none; border-radius: 5px 5px 0 0;
           position: relative; top: 1px; }}
  nav a.on {{ color: {text}; background: {panel}; border-color: {border};
              border-bottom: 1px solid {panel}; }}
  table.stats {{ border-collapse: collapse; font-size: 0.78em;
                 font-variant-numeric: tabular-nums; margin-bottom: 0.6em; }}
  table.stats th, table.stats td {{ padding: 0.35em 0.75em;
                                    border-bottom: 1px solid {border};
                                    text-align: right; white-space: nowrap; }}
  table.stats th {{ color: {muted}; font-weight: 600; text-align: right;
                    font-size: 0.92em; text-transform: uppercase;
                    letter-spacing: 0.04em; }}
  table.stats td.l, table.stats th.l {{ text-align: left; }}
  table.stats tbody tr:hover {{ background: {panel}; }}
  .chip {{ display: inline-block; font-size: 0.88em; padding: 0.05em 0.5em;
           border-radius: 999px; margin-right: 0.3em;
           border: 1px solid currentColor; }}
  .bar {{ display: inline-block; height: 0.62em; border-radius: 2px;
          background: {accent}; vertical-align: middle;
          margin-right: 0.45em; }}
  .note {{ color: {muted}; font-size: 0.8em; max-width: 60em;
           line-height: 1.5; }}
  .prog {{ height: 6px; border-radius: 3px; background: {border};
           overflow: hidden; margin: 0.5em 0; max-width: 30em; }}
  .prog > div {{ height: 100%; background: {accent}; }}
  .swatch {{ display: inline-block; width: 0.7em; height: 0.7em;
             border-radius: 2px; margin-right: 0.35em; }}
"""

CSS = STYLE_TMPL.format(bg=BG, panel=BG_PANEL, border=BORDER, text=TEXT,
                        muted=TEXT_MUTED, accent=ACCENT)

TABS = (('browse', 'Browse tiles'), ('stats', 'Dataset statistics'),
        ('results', 'Training results'))


def nav_html(active):
    out = []
    for key, label in TABS:
        cls = ' class="on"' if key == active else ''
        out.append(f'<a href="/?tab={key}"{cls}>{label}</a>')
    return '<nav>' + ''.join(out) + '</nav>'


PAGE = """<!doctype html>
<html><head>
<title>CARLA dataset viewer</title>
<style>{css}</style>
</head><body>
<h1>CARLA dataset viewer</h1>
{nav}
<div class="sub">{data_root} &mdash; datasets: <code>{split}</code> &mdash;
  {n_tiles:,} tiles across {n_towns} towns<br>
  showing <b>{town}</b> (<b>{town_split}</b>) tiles {start}&ndash;{end} &mdash;
  representation: <b>{mode}</b><br>{class_summary}</div>

<form method="get">
  <input type="hidden" name="submitted" value="1">
  <fieldset>
    <label class="top">Town</label>
    <select name="town">{town_opts}</select>
  </fieldset>
  <fieldset>
    <label class="top">Tiles to show</label>
    <input type="number" name="count" value="{count}" min="1" max="60" step="1">
  </fieldset>
  <fieldset>
    <label class="top">Start index</label>
    <input type="number" name="start" value="{start}" min="0" step="1">
  </fieldset>
  <fieldset>
    <label class="top">Representation</label>
    <select name="mode">{mode_opts}</select>
  </fieldset>
  <fieldset>
    <label class="top">Point size</label>
    <input type="number" name="point_size" value="{point_size}" min="0.1"
           max="20" step="0.1">
  </fieldset>
  <fieldset class="checks">
    <label class="top">Overlays</label>
    <label><input type="checkbox" name="polylines" value="1" {pl_checked}>
      reference-line polylines</label>
    <label><input type="checkbox" name="linear_density" value="1" {lin_checked}>
      linear density scale (default: log)</label>
  </fieldset>
  {class_fields}
  {pred_fields}
  <button type="submit">Render</button>
</form>

<div class="gallery">
{gallery}
</div>
</body></html>
"""


CLASS_FIELDS = """
  <fieldset class="checks">
    <label class="top">Polyline classes</label>
    {class_boxes}
  </fieldset>
"""

PRED_FIELDS = """
  <fieldset>
    <label class="top">Predictions (work-dir)</label>
    <select name="results">{result_opts}</select>
  </fieldset>
  <fieldset>
    <label class="top">Pred score &ge;</label>
    <input type="number" name="score_thresh" value="{score_thresh}"
           min="0" max="1" step="0.05">
  </fieldset>
"""

NO_WORKDIR_NOTE = """
  <fieldset>
    <label class="top">Predictions</label>
    <div class="meta" style="max-width:22em">
      pass <code>--work-dir &lt;dir&gt;</code> at startup to overlay predicted
      polylines from a results json
    </div>
  </fieldset>
"""

NO_RESULTS_NOTE = """
  <fieldset>
    <label class="top">Predictions</label>
    <div class="meta" style="max-width:30em">
      No predictions found in <code>{work_dir}</code>. This repo writes
      them to <code>&lt;work_dir&gt;/submission_vector.json</code> &mdash; both
      from <code>tools/test.py --eval</code> and from the training-time eval
      hook &mdash; so an absent file means no eval has run in this work-dir
      yet. Note each eval OVERWRITES that one file; copy it aside to compare
      checkpoints.
      {howto}
    </div>
  </fieldset>
"""

HOWTO_WITH_CKPT = """
      <br><br>Generate them from the checkpoint already in this work-dir:
      <br><code style="display:block;white-space:pre-wrap;margin-top:.4em">python3 tools/test.py \\
  {config} \\
  {ckpt} \\
  --launcher none --eval --work-dir {work_dir} \\
  --eta=0.5 --sampling_timesteps=5 --query_threshold=0.5</code>
      then reload this page (results are re-scanned per request).
"""

HOWTO_NO_CKPT = """
      <br><br>There's no <code>.pth</code> checkpoint here either, so train
      first, then run <code>tools/test.py &lt;config&gt; &lt;ckpt&gt;
      --launcher none --eval --work-dir &lt;work_dir&gt;</code>.
"""


def no_results_note(work_dir):
    """Build the 'no predictions' note, filled in with the checkpoint and
    config actually present in this work_dir so the suggested command is
    copy-pasteable rather than a template with placeholders."""
    ckpts = sorted(glob_pth(work_dir))
    cfgs = sorted(f for f in os.listdir(work_dir) if f.endswith('.py')) \
        if osp.isdir(work_dir) else []
    if ckpts:
        # prefer latest.pth / best_*, else whatever's newest
        preferred = next((c for c in ckpts if osp.basename(c) == 'latest.pth'),
                          None) or ckpts[-1]
        cfg = (osp.join(work_dir, cfgs[0]) if cfgs
               else 'plugin/configs/mapdiffusion_carla_lidar_25m.py')
        howto = HOWTO_WITH_CKPT.format(config=cfg, ckpt=preferred,
                                        work_dir=work_dir)
    else:
        howto = HOWTO_NO_CKPT
    return NO_RESULTS_NOTE.format(work_dir=work_dir, howto=howto)


def glob_pth(work_dir):
    out = []
    if not osp.isdir(work_dir):
        return out
    for dirpath, _d, filenames in os.walk(work_dir):
        for fn in filenames:
            if fn.endswith('.pth'):
                out.append(osp.join(dirpath, fn))
    return out


def _safe_int(v, default=None):
    try:
        n = int(v)
    except (TypeError, ValueError):
        return default
    return n if n > 0 else default


def _safe_float(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def resolve_results_path(label):
    """Map a UI label back to a real path, and refuse anything not in the
    discovered set -- the label reaches us straight from a query string, so
    it must never be joined onto a filesystem path unchecked."""
    if not label or not STATE.get('work_dir'):
        return None
    return discover_results(STATE['work_dir']).get(label)


def _opts(values, current, labels=None):
    out = []
    for i, v in enumerate(values):
        lab = labels[i] if labels else v
        sel = ' selected' if str(v) == str(current) else ''
        out.append(f'<option value="{v}"{sel}>{lab}</option>')
    return '\n'.join(out)


def selected_classes(args):
    """Which polyline classes to draw, as a set of class_key() strings, or
    None for 'everything'.

    The hidden `submitted` field exists to tell an unchecked-everything
    submission apart from a bare first load: with checkboxes, both arrive
    as zero `cls` params, and without the marker 'show none' would silently
    render as 'show all'.
    """
    if not args.get('submitted'):
        return None
    return set(args.getlist('cls'))


def class_boxes_html(selected):
    boxes = []
    for key, name in STATE['class_choices']:
        checked = ' checked' if selected is None or key in selected else ''
        color = class_color(None if key == 'none' else int(key))
        boxes.append(
            f'<label><input type="checkbox" name="cls" value="{key}"'
            f'{checked}> <span style="color:{color}">&#9644;</span> '
            f'{html.escape(name)}</label>')
    return '\n'.join(boxes)


@app.route('/')
def index():
    if request.args.get('tab') == 'stats':
        return stats_page()
    if request.args.get('tab') == 'results':
        return results_page()
    towns = list(STATE['groups'].keys())
    town = request.args.get('town', towns[0])
    if town not in STATE['groups']:
        town = towns[0]
    group = STATE['groups'][town]
    town_split = group['split']
    count = max(1, min(60, int(request.args.get('count', 6))))
    start = max(0, int(request.args.get('start', 0)))
    mode = request.args.get('mode', 'rgb')
    point_size = request.args.get('point_size', '1.5')
    polylines = '1' if request.args.get('polylines') else ''
    linear_density = '1' if request.args.get('linear_density') else ''
    results = request.args.get('results', '')
    score_thresh = request.args.get('score_thresh', '0.3')
    classes = selected_classes(request.args)
    # first load defaults to polylines on -- it's the most useful view and
    # makes the frame issue above immediately visible
    if not request.args:
        polylines = '1'

    # re-scan on each request so results produced while the viewer is
    # running show up without a restart
    result_files = (discover_results(STATE['work_dir'])
                     if STATE.get('work_dir') else {})
    if results and results not in result_files:
        results = ''

    tiles = [t for t in STATE['tiles']
             if t['_group'] == town][start:start + count]

    # Unconditional cache-buster. The per-mode query params already make
    # each image a distinct URL, so caching *shouldn't* be able to serve a
    # stale render -- but in practice it did anyway (server verified to
    # return four different PNGs per mode, byte-for-byte, while the browser
    # kept showing one of them). A token that changes every page load makes
    # every image URL unique, which no cache layer can defeat. Rendering is
    # cheap and local, so always re-fetching costs nothing that matters.
    cachebust = f'{time.time():.6f}'

    figs = []
    for t in tiles:
        params = [
            ('name', t['name']),
            ('ds', t['_ds']),
            ('mode', mode),
            ('point_size', point_size),
            ('v', cachebust),
        ]
        if polylines:
            params.append(('polylines', '1'))
        if linear_density:
            params.append(('linear_density', '1'))
        if classes is not None:
            # repeated key, hence a list of pairs rather than a dict
            params.append(('clsfilter', '1'))
            params.extend(('cls', c) for c in sorted(classes))
        if results:
            params.append(('results', results))
            params.append(('score_thresh', score_thresh))
        # urlencode + html.escape is REQUIRED here, not cosmetic. A raw "&"
        # separator in an HTML attribute starts an entity reference, and
        # browsers resolve known entity names even without the closing ";"
        # -- so "&mode=density&gt_frame=offset" was being parsed as
        # mode="density>_frame=offset" (&gt -> ">"), silently corrupting the
        # mode and dropping gt_frame entirely. Every request then fell
        # through to the default branch and rendered "top-down" no matter
        # what was selected. curl never reproduced it because nothing was
        # HTML-parsing the URL; only a real browser triggers it.
        q = '?' + urlencode(params)
        q_attr = html.escape(q, quote=True)
        cap = (f'{t["name"]} — {t["n_points"]:,} pts, '
               f'{t.get("n_polylines", 0)} GT polylines')
        by_class = t.get('n_polylines_by_class') or {}
        if len(by_class) > 1 or (by_class and classes is not None):
            cap += ' (' + ', '.join(f'{k}: {v}' for k, v in
                                     sorted(by_class.items())) + ')'
        if results:
            preds = load_results(result_files[results]).get(t['name'])
            if preds is None:
                cap += ' — <span style="color:#d29922">no preds for this tile</span>'
            else:
                kept = sum(1 for _p, s, _c in preds if s >= float(score_thresh))
                cap += f', {kept} preds'
        figs.append(
            f'<figure><a href="/tile.png{q_attr}" target="_blank">'
            f'<img src="/tile.png{q_attr}"></a>'
            f'<figcaption>{cap}</figcaption></figure>')

    # NB: not named `html` -- that shadows the stdlib `html` module used
    # above for attribute escaping.
    page_html = PAGE.format(
        css=CSS, nav=nav_html('browse'),
        data_root=osp.abspath(STATE['data_root']),
        split=', '.join(STATE['datasets'].keys()),
        n_tiles=len(STATE['tiles']), n_towns=len(towns),
        town=group['town'], town_split=town_split, mode=mode,
        end=start + len(tiles),
        class_summary=STATE['class_summary'],
        town_opts=_opts(towns, town,
                         [f'{STATE["groups"][t]["town"]}  '
                          f'({STATE["groups"][t]["split"]}'
                          f', {STATE["groups"][t]["count"]:,} tiles)'
                          for t in towns]),
        class_fields=CLASS_FIELDS.format(
            class_boxes=class_boxes_html(classes)),
        count=count, start=start, point_size=point_size,
        mode_opts=_opts(['rgb', 'label', 'points', 'density', 'intensity'],
                         mode,
                         ['true RGB colour', 'lane label',
                          'top-down (flat colour)',
                          'density heat map (1 m² bins)', 'intensity']),
        pl_checked='checked' if polylines else '',
        lin_checked='checked' if linear_density else '',
        pred_fields=(
            NO_WORKDIR_NOTE if not STATE.get('work_dir')
            else no_results_note(STATE['work_dir'])
            if not result_files
            else PRED_FIELDS.format(
                score_thresh=score_thresh,
                result_opts=_opts([''] + list(result_files.keys()), results,
                                   ['(none)'] + list(result_files.keys())))),
        gallery='\n'.join(figs) or
                f'<p style="color:{TEXT_MUTED}">no tiles for that town/range</p>',
    )
    resp = make_response(page_html)
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    return resp


@app.route('/tile.png')
def tile_png():
    name = request.args.get('name')
    ds = request.args.get('ds')
    # tile names repeat across grid exports, so the (dataset, name) pair is
    # what identifies a tile -- and validating the pair against the index is
    # also what keeps `ds` from being joined onto a path unchecked
    if not name or not ds or tile_by_uid(f'{ds}/{name}') is None:
        abort(404)
    try:
        point_size = float(request.args.get('point_size', 1.5))
    except ValueError:
        point_size = 1.5
    buf = render_tile(
        name,
        mode=request.args.get('mode', 'rgb'),
        show_polylines=bool(request.args.get('polylines')),
        point_size=point_size,
        max_points=STATE['max_points'],
        log_density=request.args.get('linear_density') != '1',
        ds=ds,
        results_path=resolve_results_path(request.args.get('results')),
        score_thresh=_safe_float(request.args.get('score_thresh'), 0.3),
        classes=(set(request.args.getlist('cls'))
                 if request.args.get('clsfilter') else None),
        top_n=_safe_int(request.args.get('top_n')),
    )
    if buf is None:
        abort(404)
    resp = send_file(buf, mimetype='image/png')
    # Every render is cheap and always reflects on-disk data; caching only
    # ever causes confusion here (e.g. "changing the representation does
    # nothing" when the browser is quietly serving a stale image).
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    return resp


# --------------------------------------------------------------- statistics
#
# Colour policy for this tab, decided once here so every chart reads as one
# system:
#
#  * A distribution compared ACROSS groups is a horizontal box plot in a
#    single hue -- never one hue per town. Six towns sits at the categorical
#    soft cap, and colouring by town would spend the only free channel on
#    information the axis labels already carry.
#  * Categorical colour is reserved for the two charts whose subject really
#    IS a category (polyline class, lane label), where it reuses the browse
#    tab's own CLASS_COLORS / LABEL_COLORS so the two tabs agree.
#  * The train/test comparison is two shades of one hue, not two hues.
#  * CRITICAL is a status colour: it marks flagged/suspect marks only, and
#    is never used as "series N".
BOX_HUE = ACCENT
BOX_HUE_ALT = '#a5d6ff'      # eval splits: same hue, lighter step
CRITICAL = '#f85149'
MUTED_MARK = '#484f58'
# fixed order, never cycled -- there are at most three splits
SERIES = ('#58a6ff', '#f0883e', '#3fb950')

# a tile whose points collapse to under this fraction of distinct cells is
# degenerate: 4.99M of town01_tile_00000's 5M points share one xy position
DEGENERATE_RATIO = 0.01
DEGENERATE_MIN_RAW = 100000
# under this many surviving cells the LiDAR voxelizer can produce zero
# voxels, which crashes extract_lidar_feat with a bare IndexError
NEAR_EMPTY_CELLS = 200
NEAR_EMPTY_RAW = 2000

SCAN = {'thread': None, 'done': 0, 'total': 0, 'error': None,
        'started': None, 'finished': None}
SCAN_LOCK = threading.Lock()


def unique_cells(pts, grid):
    """How many grid cells the points occupy -- i.e. how many points would
    survive GridSamplePoints, which is what the model actually sees.

    Same integer coordinate packing + single ``unique`` as
    ``GridSamplePoints`` in
    plugin/datasets/pipelines/loading.py, in numpy rather
    than torch: this viewer deliberately imports neither torch nor mmdet3d
    so it can run outside the container.

    `grid` is per-axis, because the model's own lidar_voxel_size is
    anisotropic ([0.1, 0.1, 0.4]); using one scalar would answer a question
    the training pipeline never asks.

    The count will not match GridSamplePoints' output exactly, and is not
    meant to: that transform phases its grid on point_cloud_range's own
    origin and clamps out-of-range coordinates into the edge cells, whereas
    this phases on the tile's own minimum. A half-cell shift moves points
    across cell boundaries either way, so the two differ by a few percent
    (on town01_tile_00000: 4,132 here vs ~3,242 measured through the real
    transform). What is robust -- and what this statistic is for -- is the
    order of magnitude: 5,000,000 raw points collapsing to ~4,000 distinct
    cells is a 1200x reduction no phasing choice can explain away.
    """
    if pts.shape[0] == 0:
        return 0
    grid = np.asarray(grid, dtype=np.float64)[:pts.shape[1]]
    g = np.floor((pts - pts.min(axis=0)) / grid).astype(np.int64)
    dims = g.max(axis=0) + 1
    key = g[:, 0]
    for i in range(1, g.shape[1]):
        key = key * dims[i] + g[:, i]
    return int(np.unique(key).size)


def scan_tile(t, grid):
    """Everything about one tile that needs its .npz read. Returns
    (uid, record); never raises -- a tile that fails is recorded with an
    `error` so one bad file can't abort a multi-thousand-tile scan."""
    uid, name, ds = t['_uid'], t['name'], t['_ds']
    rec = {}
    try:
        block = load_block(name, ds)
        if block is None:
            return uid, {'error': 'blocks/%s.npz missing' % name}
        try:
            # Two origins, deliberately kept apart:
            #   `origin`      -- the block's `offset`, the frame
            #                    features[:, 0:3] is stored in. Only the z
            #                    component is used below, to recover WORLD z;
            #                    tile_center would give the wrong answer
            #                    there.
            #   `poly_origin` -- `tile_center`, the tile-centred frame this
            #                    repo's converter, point loader and renderer
            #                    all work in. Polylines are shifted by this.
            origin = (np.asarray(block['offset'], dtype=np.float64)
                       if 'offset' in block else np.zeros(3))
            poly_origin = origin
            ctr = None
            if 'tile_center' in block:
                tc = np.asarray(block['tile_center'], dtype=np.float64)
                poly_origin = tc
                # In the tile-centred frame the tile sits on the origin, so
                # the out-of-bounds test below measures from zero.
                ctr = np.zeros(2, dtype=np.float32)
                # `tile_shift` -- how far the cloud's centroid (`offset`)
                # sits from the tile's geometric centre. Not an error here:
                # it is exactly what LoadCarlaPointsFromFile subtracts, and
                # it grows with tile size (mean 1.6 m, max 12.1 m on the 25 m
                # export). Kept because it is the size of the correction the
                # offset-frame pipelines in the sibling repos never applied.
                rec['drift'] = float(np.linalg.norm(tc[:2] - origin[:2]))
            radius = (float(block['tile_radius'])
                       if 'tile_radius' in block else None)
            # the expensive member -- everything above is a few bytes
            xyz = np.asarray(block['features'][:, :3], dtype=np.float32)
            rec['n_raw'] = int(xyz.shape[0])
            if rec['n_raw']:
                z = xyz[:, 2]
                rec['z_min'] = float(z.min())
                rec['z_max'] = float(z.max())
                # Recorded in WORLD z, by adding the tile's own origin back
                # on, and this is not cosmetic.
                #
                # `features` is stored relative to `offset`, and `offset` is
                # the cloud's CENTROID -- verified directly: mean(features)
                # is 0 on all three axes for every tile checked, to float32
                # rounding. So the centroid-relative mean z is identically
                # ~0 (measured spread across the 4103-tile train split:
                # -0.07 to 0.17 m, i.e. nothing), and plotting anything
                # against it would be plotting against rounding noise.
                #
                # In world terms both are real: z_mean is the elevation of
                # the tile's centroid, z_median the elevation of its road
                # surface (most returns are ground). Their DIFFERENCE is the
                # vertical skew -- how far the road sits below the centre of
                # mass of everything above it -- and that difference is the
                # only part of this the model can actually see, since its
                # input is centroid-relative. Both are in results.csv.
                rec['z_median'] = float(np.median(z)) + float(origin[2])
                rec['z_mean'] = float(z.mean()) + float(origin[2])
                rec['n_effective'] = unique_cells(xyz, grid)
                rec['n_xy_cells'] = unique_cells(xyz[:, :2], grid)
            else:
                rec['n_effective'] = rec['n_xy_cells'] = 0
            if 'labels' in block:
                lab = np.asarray(block['labels']).astype(np.int64).ravel()
                if lab.size:
                    # +1 so the -1 'unlabeled' id survives bincount
                    counts = np.bincount(lab + 1)
                    rec['labels'] = {str(i - 1): int(c)
                                      for i, c in enumerate(counts) if c}
        finally:
            block.close()

        # reference lines are small json, cheap next to the above
        pl, gt_out = [], 0
        for pts, cid, _cname in load_polylines(
                name, poly_origin.astype(np.float32), ds):
            length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            pl.append([-1 if cid is None else int(cid),
                        int(pts.shape[0]), round(length, 2)])
            if radius is not None and ctr is not None:
                # square tile, so chebyshev distance is the right test
                gt_out += int((np.abs(pts - ctr).max(axis=1)
                                > radius + 1.0).sum())
        rec['pl'] = pl
        rec['n_gt'] = len(pl)
        rec['gt_out'] = gt_out
    except Exception as e:                        # noqa: BLE001 - see docstring
        rec['error'] = f'{type(e).__name__}: {e}'
    return uid, rec


# Bump whenever scan_tile() starts recording a NEW field. Without this, a
# warm cache from an older build is loaded happily, every tile looks
# "already scanned", and the charts that need the new field render empty --
# the silent-staleness failure this file keeps running into. A mismatch
# drops the cache and re-scans, which is a minute for 4,362 tiles.
#   1 -> original; 2 -> adds z_median / z_mean; 3 -> those in WORLD z;
#   4 -> polylines and `drift` in the tile-centred frame (see scan_tile)
CACHE_VERSION = 4


def cache_path_for(ds):
    """One cache file per dataset *directory*, so a scan done via
    --data-root <root> is reused by a later --data-root <root>/train."""
    d = osp.abspath(ds_dir(ds) or ds)
    return osp.join(STATE['cache_dir'],
                     hashlib.sha1(d.encode()).hexdigest()[:16] + '.json')


def load_deep_cache():
    """Populate STATE['deep'] from disk. Entries scanned at a different
    --scan-grid are dropped: n_effective is only meaningful relative to the
    grid size it was measured at."""
    loaded = 0
    for ds in STATE['datasets']:
        path = cache_path_for(ds)
        if not osp.isfile(path):
            continue
        try:
            with open(path) as f:
                blob = json.load(f)
        except (ValueError, OSError) as e:
            print(f'  [stats] ignoring unreadable cache {path}: {e}')
            continue
        # `grid` was a scalar in the first version of this cache; anything
        # that isn't the current per-axis grid is simply re-scanned
        cached_grid = blob.get('grid')
        if not isinstance(cached_grid, list):
            cached_grid = [cached_grid]
        if cached_grid != list(STATE['scan_grid']):
            print(f'  [stats] {path}: cached at grid='
                  f'{blob.get("grid")}, want {list(STATE["scan_grid"])}'
                  f' -- ignored')
            continue
        # version 1 predates the key, hence the default
        if int(blob.get('version', 1)) != CACHE_VERSION:
            print(f'  [stats] {path}: cache schema v'
                  f'{blob.get("version", 1)}, want v{CACHE_VERSION} '
                  f'(newer fields missing) -- ignored, re-scan to rebuild')
            continue
        for name, rec in (blob.get('tiles') or {}).items():
            uid = f'{ds}/{name}'
            if uid in STATE['tiles_by_uid']:
                STATE['deep'][uid] = rec
                loaded += 1
    return loaded


def save_deep_cache(ds):
    """Write atomically -- a scan of the full train split takes minutes and
    the viewer is routinely Ctrl-C'd mid-run."""
    recs = {}
    for t in STATE['tiles']:
        if t['_ds'] != ds:
            continue
        rec = STATE['deep'].get(t['_uid'])
        if rec is not None:
            recs[t['name']] = rec
    if not recs:
        return
    path = cache_path_for(ds)
    os.makedirs(osp.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump({'grid': list(STATE['scan_grid']), 'dir': ds_dir(ds),
                    'version': CACHE_VERSION, 'tiles': recs}, f)
    os.replace(tmp, path)


def _scan_worker(todo):
    grid = STATE['scan_grid']
    try:
        with ThreadPoolExecutor(max_workers=STATE['scan_workers']) as ex:
            for uid, rec in ex.map(lambda t: scan_tile(t, grid), todo):
                STATE['deep'][uid] = rec
                with SCAN_LOCK:
                    SCAN['done'] += 1
                    done = SCAN['done']
                if done % 500 == 0:
                    for ds in STATE['datasets']:
                        save_deep_cache(ds)
    except Exception as e:                        # noqa: BLE001
        with SCAN_LOCK:
            SCAN['error'] = f'{type(e).__name__}: {e}'
    finally:
        for ds in STATE['datasets']:
            save_deep_cache(ds)
        with SCAN_LOCK:
            SCAN['finished'] = time.time()


def start_scan(rescan=False):
    """Kick off a background deep scan of every tile missing from
    STATE['deep'] (or of everything, if rescan). Returns False if one is
    already running."""
    with SCAN_LOCK:
        if SCAN['thread'] is not None and SCAN['thread'].is_alive():
            return False
        # --scan-stride samples the flat tile list; groups keep their
        # relative sizes, so the distributions stay representative
        pool = STATE['tiles'][::max(1, STATE['scan_stride'])]
        todo = [t for t in pool if rescan or t['_uid'] not in STATE['deep']]
        SCAN.update(done=0, total=len(todo), error=None,
                     started=time.time(), finished=None)
        SCAN['thread'] = threading.Thread(target=_scan_worker, args=(todo,),
                                           daemon=True)
        SCAN['thread'].start()
    return True


def scan_running():
    th = SCAN['thread']
    return th is not None and th.is_alive()


def stat_rows():
    """One row per tile: manifest fields plus whatever the deep scan has
    recorded for it (None if it hasn't been scanned). Pure lookup, no I/O."""
    deep = STATE['deep']
    rows = []
    for t in STATE['tiles']:
        area = float(t.get('_area') or 1.0)
        n_pts = int(t.get('n_points') or 0)
        rows.append({
            'uid': t['_uid'], 'name': t['name'], 'ds': t['_ds'],
            'split': t['_split'], 'group': t['_group'], 'gidx': t['_gidx'],
            'town': STATE['groups'][t['_group']]['town'],
            'n_points': n_pts, 'area': area, 'density': n_pts / area,
            'n_polylines': int(t.get('n_polylines') or 0),
            'by_class': t.get('n_polylines_by_class') or {},
            'deep': deep.get(t['_uid']),
        })
    return rows


def group_rows(rows, group_by):
    """Ordered {label: [row]}. 'town' gives one series per (dataset, town);
    'split' collapses to train/val/test, which is the comparison that
    actually bears on eval numbers -- train and test here are different
    towns, so a distribution shift between them is entirely possible."""
    out = {}
    for r in rows:
        if group_by == 'split':
            label = r['split']
        else:
            g = STATE['groups'][r['group']]
            label = f"{g['town']} ({g['split']})"
        out.setdefault(label, []).append(r)
    return out


def is_eval_label(label):
    return 'test' in label.lower() or 'val' in label.lower()


def fmt_n(v, unit=''):
    if v is None:
        return '&mdash;'
    if isinstance(v, float) and not v.is_integer():
        return f'{v:,.2f}{unit}' if abs(v) < 1000 else f'{v:,.0f}{unit}'
    return f'{int(v):,}{unit}'


def med(vals):
    return float(np.median(vals)) if len(vals) else None


def grid_str():
    g = STATE['scan_grid']
    return '×'.join(f'{v:g}' for v in g) if len(set(g)) > 1 else f'{g[0]:g}'


# ---- plotting ------------------------------------------------------------

def stat_fig(height=4.2, width=7.6):
    fig = Figure(figsize=(width, height))
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(BG)
    ax = fig.subplots()
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    for side, spine in ax.spines.items():
        spine.set_color(BORDER)
        if side in ('top', 'right'):
            spine.set_visible(False)
    # solid hairline one shade off the surface; never dashed
    ax.grid(True, color=BORDER, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def abs_margins(fig, left=1.9, right=1.15, top=0.5, bottom=0.7):
    """subplots_adjust in INCHES rather than figure fractions.

    These figures vary in height with the number of groups, and a fractional
    bottom margin shrinks in absolute terms as the figure grows shorter --
    which silently clipped the x-axis label off every one-group plot.
    """
    w, h = fig.get_size_inches()
    fig.subplots_adjust(left=left / w, right=1.0 - right / w,
                         top=1.0 - top / h, bottom=bottom / h)


def finish(fig, ax, title, xlabel='', tight=False):
    ax.set_title(title, color=TEXT, fontsize=10, loc='left', pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_MUTED, fontsize=8)
    if tight:
        # must run AFTER the title and label exist, or it reserves no room
        # for them and they get clipped off the canvas
        fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, facecolor=BG)
    buf.seek(0)
    return buf


def legend(ax, loc='lower right', handles=None):
    leg = ax.legend(handles=handles, fontsize=7.5, loc=loc,
                     facecolor=BG_PANEL, edgecolor=BORDER, labelcolor=TEXT,
                     framealpha=0.95)
    leg.get_frame().set_linewidth(0.5)
    return leg


def box_by_group(fig, ax, groups, values, log=False, fmt='{:,.0f}',
                  top_margin=0.5):
    """Horizontal box plot, one row per group, single hue. `values(rows)`
    returns the flat list of numbers for a group. Medians are direct-labelled
    in a column at the right; nothing else is, per the 'no number on every
    mark' rule.

    Returns (labels, boxes) so a caller that legitimately needs categorical
    colour (the per-class length chart) can recolour the patches, or None if
    no group had any data.
    """
    labels, series = [], []
    for label, rows in groups.items():
        vals = [v for v in values(rows) if v is not None and np.isfinite(v)]
        if log:
            vals = [v for v in vals if v > 0]
        if vals:
            labels.append(label)
            series.append(np.asarray(vals, dtype=np.float64))
    if not series:
        return None
    pos = np.arange(len(series))[::-1]  # first group at the top
    bp = ax.boxplot(
        series, vert=False, positions=pos, widths=0.55, patch_artist=True,
        flierprops=dict(marker='.', markersize=2.5, alpha=0.45,
                         markerfacecolor=MUTED_MARK, markeredgecolor='none'),
        medianprops=dict(color=BG, linewidth=1.4),
        whiskerprops=dict(color=BORDER, linewidth=1.0),
        capprops=dict(color=BORDER, linewidth=1.0))
    for patch, label in zip(bp['boxes'], labels):
        patch.set_facecolor(BOX_HUE_ALT if is_eval_label(label) else BOX_HUE)
        patch.set_edgecolor('none')
    ax.set_yticks(pos)
    ax.set_yticklabels(labels, fontsize=8, color=TEXT)
    ax.set_ylim(-0.7, len(series) - 0.3)
    if log:
        ax.set_xscale('log')
    ax.grid(False, axis='y')
    for y, s in zip(pos, series):
        ax.annotate(fmt.format(float(np.median(s))), xy=(1.015, y),
                     xycoords=('axes fraction', 'data'), va='center',
                     ha='left', fontsize=7.5, color=TEXT_MUTED,
                     annotation_clip=False)
    ax.annotate('median', xy=(1.015, 1.02), xycoords='axes fraction',
                 fontsize=7, color=TEXT_MUTED, annotation_clip=False)
    abs_margins(fig, top=top_margin)
    return labels, bp['boxes']


def annotate_below(ax, x, text):
    """Label a vertical reference line in a band opened up beneath the last
    box row. The top of these axes belongs to the title, and the reference
    lines land mid-plot, so a label up there collides with it."""
    ax.set_ylim(-1.5, ax.get_ylim()[1])
    ax.annotate(text, xy=(x, -1.35), xytext=(6, 0),
                 textcoords='offset points', fontsize=7, color=CRITICAL,
                 ha='left', va='bottom')


def hist_by_group(fig, ax, groups, values, log=False, bins=36):
    """Overlaid step histograms, normalised to share-of-tiles so groups of
    very different size compare directly. Used for the split comparison,
    where distribution *shape* is the point; capped at three series."""
    per = {}
    for label, rows in groups.items():
        vals = [v for v in values(rows) if v is not None and np.isfinite(v)]
        if log:
            vals = [v for v in vals if v > 0]
        if vals:
            per[label] = np.asarray(vals, dtype=np.float64)
    if not per or len(per) > len(SERIES):
        return False
    lo = min(float(v.min()) for v in per.values())
    hi = max(float(v.max()) for v in per.values())
    if log:
        edges = np.logspace(np.log10(max(lo, 1e-9)), np.log10(max(hi, lo * 10)),
                             bins)
        ax.set_xscale('log')
    else:
        edges = np.linspace(lo, hi if hi > lo else lo + 1, bins)
    for i, (label, vals) in enumerate(per.items()):
        ax.hist(vals, bins=edges, histtype='step', linewidth=2.0,
                 color=SERIES[i], weights=np.full(vals.size, 1.0 / vals.size),
                 label=f'{label} ({vals.size:,} tiles)')
    ax.set_ylabel('share of tiles', color=TEXT_MUTED, fontsize=8)
    legend(ax, 'upper right')
    return True


def box_height(n_groups):
    """Plot area grows with the number of rows; the fixed chrome (title,
    x-axis band) is added by abs_margins, so this only needs the data band."""
    return max(2.6, 1.3 + 0.45 * max(1, n_groups))


def dist_plot(groups, group_by, values, title, xlabel, log=False,
               fmt='{:,.0f}'):
    if group_by == 'split':
        fig, ax = stat_fig(height=4.2)
        if hist_by_group(fig, ax, groups, values, log=log):
            return finish(fig, ax, title, xlabel, tight=True)
    fig, ax = stat_fig(height=box_height(len(groups)))
    if box_by_group(fig, ax, groups, values, log=log, fmt=fmt) is None:
        return None
    return finish(fig, ax, title, xlabel)


def class_order():
    """(id, name) in manifest id order -- which is also the CLASS_COLORS
    index, so a class keeps its colour across both tabs. A class-free export
    yields [], and callers degrade to a single unclassified set."""
    lookup = STATE['class_lookup']
    ids = sorted(int(k) for k in lookup if str(k).lstrip('-').isdigit())
    return [(i, lookup[str(i)]) for i in ids]


def stacked_bars(fig, ax, labels, seg_names, matrix, colors, normalize,
                  xlabel):
    """Horizontal stacked bars -- part-to-whole per group, gone horizontal
    because the category names are long. Segments are separated by a 2px
    surface-coloured gap rather than an outline."""
    pos = np.arange(len(labels))[::-1]
    matrix = np.asarray(matrix, dtype=np.float64)
    if normalize:
        totals = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, np.where(totals == 0, 1, totals)) * 100.0
    left = np.zeros(len(labels))
    for j, name in enumerate(seg_names):
        w = matrix[:, j]
        ax.barh(pos, w, left=left, height=0.6, color=colors[j],
                 edgecolor=BG, linewidth=1.2,
                 label=f'{name} ({int(matrix[:, j].sum()):,})'
                       if not normalize else name)
        left += w
    ax.set_yticks(pos)
    ax.set_yticklabels(labels, fontsize=8, color=TEXT)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.grid(False, axis='y')
    ax.set_xlabel(xlabel, color=TEXT_MUTED, fontsize=8)
    # A little headroom so the longest bar doesn't run into the axes edge.
    ax.set_xlim(0, ax.get_xlim()[1] * 1.02)
    # Legend BELOW the plot, anchored in FIGURE coordinates: at eight
    # segments it is far too tall to sit inside the axes without covering the
    # very bars it describes, and anchoring it to the axes makes its position
    # depend on the group count.
    ncol = min(3, max(1, len(seg_names)))
    nrow = int(np.ceil(len(seg_names) / float(ncol)))
    abs_margins(fig, right=0.4, bottom=0.85 + 0.16 * nrow)
    leg = ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.01),
                     bbox_transform=fig.transFigure, ncol=ncol, fontsize=7.5,
                     facecolor=BG_PANEL, edgecolor=BORDER, labelcolor=TEXT,
                     framealpha=0.95)
    leg.get_frame().set_linewidth(0.5)
    return nrow


def plot_classes(groups, group_by, normalize):
    order = class_order()
    labels = list(groups.keys())
    if order:
        seg_names = [n for _i, n in order]
        colors = [CLASS_COLORS.get(i, CLASS_FALLBACK) for i, _n in order]
        matrix = []
        for rows in groups.values():
            counts = {}
            for r in rows:
                for name, k in (r['by_class'] or {}).items():
                    counts[name] = counts.get(name, 0) + int(k)
            matrix.append([counts.get(n, 0) for n in seg_names])
    else:
        # class-free export: everything is one unclassified set. Degrade to
        # a single-series bar of total GT rather than inventing categories.
        seg_names = ['unclassified']
        colors = [UNCLASSED_COLOR]
        matrix = [[sum(r['n_polylines'] for r in rows)]
                  for rows in groups.values()]
    fig, ax = stat_fig(height=box_height(len(labels))
                        + 0.16 * np.ceil(len(seg_names) / 3.0))
    stacked_bars(fig, ax, labels, seg_names, matrix, colors, normalize,
                  'share of GT polylines (%)' if normalize
                  else 'GT polylines')
    return finish(fig, ax,
                   'GT polylines by class' + (' (normalised)' if normalize
                                               else ''))


def plot_labels(groups, group_by, normalize):
    """Per-point lane-label mix. Folds to the eight largest labels so the
    chart never exceeds the categorical ceiling."""
    totals = {}
    per_group = {}
    for label, rows in groups.items():
        acc = {}
        for r in rows:
            d = (r['deep'] or {}).get('labels') or {}
            for lid, c in d.items():
                acc[lid] = acc.get(lid, 0) + int(c)
                totals[lid] = totals.get(lid, 0) + int(c)
        per_group[label] = acc
    if not totals:
        return None
    keep = [k for k, _v in sorted(totals.items(), key=lambda kv: -kv[1])][:8]
    seg_names, colors = [], []
    for lid in keep:
        seg_names.append(STATE['lane_types'].get(
            lid, 'unlabeled' if lid == '-1' else lid))
        colors.append(LABEL_COLORS.get(int(lid), CLASS_FALLBACK))
    other = [k for k in totals if k not in keep]
    if other:
        seg_names.append('other')
        # deliberately not MUTED_MARK -- that is exactly LABEL_COLORS[-1],
        # so 'other' would be indistinguishable from 'unlabeled'
        colors.append('#8b949e')
    labels = list(per_group.keys())
    matrix = []
    for label in labels:
        acc = per_group[label]
        row = [acc.get(lid, 0) for lid in keep]
        if other:
            row.append(sum(acc.get(k, 0) for k in other))
        matrix.append(row)
    fig, ax = stat_fig(height=box_height(len(labels))
                        + 0.16 * np.ceil(len(seg_names) / 3.0))
    stacked_bars(fig, ax, labels, seg_names, matrix, colors, normalize,
                  'share of points (%)' if normalize else 'points')
    return finish(fig, ax, 'LiDAR points by lane label'
                   + (' (normalised)' if normalize else ''))


def plot_effective(groups, group_by):
    """Raw points vs points that survive grid sampling, log-log against
    y = x. This is the single clearest picture of the degenerate tiles: they
    fall orders of magnitude below the diagonal.

    Emphasis, not categorical -- one muted hue for every tile plus the
    status colour for the flagged ones. Colouring by town here would be an
    all-pairs form well past its three-series cap.
    """
    raw, eff, bad = [], [], []
    for rows in groups.values():
        for r in rows:
            d = r['deep'] or {}
            if 'n_effective' not in d or not d.get('n_raw'):
                continue
            raw.append(d['n_raw'])
            eff.append(max(d['n_effective'], 1))
            bad.append(is_degenerate(d))
    if not raw:
        return None
    raw = np.asarray(raw, dtype=np.float64)
    eff = np.asarray(eff, dtype=np.float64)
    bad = np.asarray(bad, dtype=bool)
    fig, ax = stat_fig(height=5.0, width=6.4)
    lo = max(1.0, min(raw.min(), eff.min()) * 0.7)
    hi = raw.max() * 1.4
    ax.plot([lo, hi], [lo, hi], color=BORDER, linewidth=1.0, zorder=1)
    ax.annotate('y = x  (every raw point distinct)', xy=(hi, hi),
                 xytext=(-6, 6), textcoords='offset points', ha='right',
                 fontsize=7.5, color=TEXT_MUTED)
    ax.scatter(raw[~bad], eff[~bad], s=11, c='#768390', linewidths=0,
                alpha=0.75, zorder=2, label=f'tiles ({int((~bad).sum()):,})')
    if bad.any():
        ax.scatter(raw[bad], eff[bad], s=30, c=CRITICAL, linewidths=0,
                    zorder=3, label=f'degenerate ({int(bad.sum()):,})')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(f'distinct {grid_str()} m cells occupied',
                   color=TEXT_MUTED, fontsize=8)
    legend(ax, 'upper left')
    return finish(fig, ax, 'Effective vs raw points per tile',
                   'raw points in the .npz', tight=True)


def plot_zrange(groups, group_by):
    """Per-group z extent against the model's own z bounds. A tile whose
    points fall outside those bounds contributes nothing to the voxel grid,
    which is how gotcha #13's empty-voxel crash happens."""
    labels, spans = [], []
    for label, rows in groups.items():
        zmins = [(r['deep'] or {}).get('z_min') for r in rows]
        zmaxs = [(r['deep'] or {}).get('z_max') for r in rows]
        zmins = [z for z in zmins if z is not None]
        zmaxs = [z for z in zmaxs if z is not None]
        if not zmins:
            continue
        labels.append(label)
        spans.append((min(zmins), float(np.median(zmins)),
                       float(np.median(zmaxs)), max(zmaxs)))
    if not spans:
        return None
    fig, ax = stat_fig(height=box_height(len(labels)) + 0.45)
    pos = np.arange(len(labels))[::-1]
    for y, (lo, mlo, mhi, hi) in zip(pos, spans):
        ax.plot([lo, hi], [y, y], color=BORDER, linewidth=1.2, zorder=2)
        ax.barh([y], [mhi - mlo], left=[mlo], height=0.42, color=BOX_HUE,
                 zorder=3)
    zlo, zhi = STATE['pc_range_z']
    for x, name in ((zlo, f'z min {zlo:g}'), (zhi, f'z max {zhi:g}')):
        ax.axvline(x, color=CRITICAL, linewidth=1.0, zorder=4)
        ax.annotate(name, xy=(x, 1.005), xycoords=('data', 'axes fraction'),
                     fontsize=7, color=CRITICAL, ha='center',
                     annotation_clip=False)
    ax.set_yticks(pos)
    ax.set_yticklabels(labels, fontsize=8, color=TEXT)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.grid(False, axis='y')
    # legend below the axes (figure coords) -- inside, it lands on top of the
    # very range bars it is describing
    abs_margins(fig, right=0.4, top=0.8, bottom=1.05)
    leg = ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.01),
                     bbox_transform=fig.transFigure, ncol=3, fontsize=7.5,
                     facecolor=BG_PANEL, edgecolor=BORDER, labelcolor=TEXT,
                     framealpha=0.95, handles=[
                         Line2D([], [], color=BOX_HUE, linewidth=6,
                                 label='median z_min .. median z_max'),
                         Line2D([], [], color=BORDER, linewidth=1.2,
                                 label='full extent across tiles'),
                         Line2D([], [], color=CRITICAL, linewidth=1.0,
                                 label='lidar_point_cloud_range z bounds')])
    leg.get_frame().set_linewidth(0.5)
    return finish(fig, ax, 'Per-tile z extent vs the model z range',
                   'z in the tile-local (offset) frame, m')


def polyline_values(rows, index):
    out = []
    for r in rows:
        for p in (r['deep'] or {}).get('pl') or []:
            out.append(p[index])
    return out


def plot_vertices(groups, group_by):
    fig, ax = stat_fig(height=box_height(len(groups)))
    if box_by_group(fig, ax, groups, lambda rows: polyline_values(rows, 1),
                     fmt='{:,.0f}', top_margin=0.85) is None:
        return None
    k = STATE['num_pts_per_vec']
    ax.axvline(k, color=CRITICAL, linewidth=1.0, zorder=4)
    # below the last box rather than above the first: the top of the axes is
    # where the title lives, and the reference line lands mid-plot, so a
    # centred label up there collides with it
    annotate_below(ax, k,
                    f'num_pts_per_vec = {k}\nleft: upsampled   right: dropped')
    return finish(fig, ax, 'Vertices per GT polyline', 'vertices')


def plot_lengths(groups, group_by):
    """Grouped by polyline class rather than by town -- class is what the
    length distribution actually varies with, and it reuses the class
    palette."""
    order = class_order()
    names = {i: n for i, n in order}
    per = {}
    for rows in groups.values():
        for r in rows:
            for cid, _nv, length in (r['deep'] or {}).get('pl') or []:
                key = names.get(cid, 'unclassified' if cid < 0 else f'class {cid}')
                per.setdefault((cid, key), []).append(length)
    if not per:
        return None
    keys = sorted(per, key=lambda kv: (kv[0] < 0, kv[0]))
    fig, ax = stat_fig(height=box_height(len(keys)))
    grouped = {k[1]: per[k] for k in keys}
    drawn = box_by_group(fig, ax, grouped, lambda v: v, log=True,
                          fmt='{:,.1f} m')
    if drawn is None:
        return None
    # the one box plot that IS categorical -- class is the subject here, so
    # it reuses the browse tab's class palette rather than the single hue
    name_to_cid = {name: cid for cid, name in keys}
    for patch, label in zip(drawn[1], drawn[0]):
        cid = name_to_cid.get(label, -1)
        patch.set_facecolor(class_color(None if cid < 0 else cid))
    return finish(fig, ax, 'GT polyline length by class',
                   'arc length (m, log)')


def is_degenerate(d):
    raw, eff = d.get('n_raw') or 0, d.get('n_effective')
    return bool(raw >= DEGENERATE_MIN_RAW and eff is not None
                 and eff < DEGENERATE_RATIO * raw)


PLOTS = {
    'points': dict(
        title='Points per tile', deep=False,
        note='Raw .npz point count. The stack at the right edge is the '
             '5,000,000-point cap &mdash; see the effective-points chart '
             'before trusting any of it.'),
    'density': dict(
        title='Points per m&sup2;', deep=False,
        note='The cross-export comparable version: the legacy split is 25 m '
             'tiles and the grid export 60 m ones, so raw counts are not '
             'comparable between them and this is.'),
    'polylines': dict(
        title='GT polylines per tile', deep=False,
        note='Tiles at zero contribute only negatives to the loss.'),
    'classes': dict(
        title='GT polylines by class', deep=False,
        note='What a decision to widen the divider-only taxonomy has to be '
             'made from. Switch to normalised to compare balance across '
             'towns of different size.'),
    'effective': dict(
        title='Effective vs raw points', deep=True,
        note='Points surviving grid sampling at the model\'s own voxel size. '
             'Tiles far below the diagonal are the degenerate ones.'),
    'zrange': dict(
        title='z extent vs model z range', deep=True,
        note='Points outside these bounds are dropped by the voxelizer; a '
             'tile entirely outside them voxelizes to nothing and crashes '
             'extract_lidar_feat.'),
    'drift': dict(
        title='|tile_center &minus; offset|', deep=True,
        note='How far apart a tile\'s two candidate origins are. GT is built '
             'in the <code>offset</code> frame, so this is the size of the '
             'error anything reverting to <code>tile_center</code> would '
             'introduce &mdash; compare it against the 0.5/1.0/1.5 m chamfer '
             'thresholds. Not a defect in itself.'),
    'vertices': dict(
        title='Vertices per GT polyline', deep=True,
        note='VectorizeMap resamples every instance to a fixed sample_num '
             '(num_points, 20 in the shipped configs), so this shows how '
             'much of the GT is being up- or down-sampled.'),
    'lengths': dict(
        title='GT polyline length by class', deep=True,
        note='Very short instances are the ones chamfer distance scores most '
             'erratically.'),
    'labels': dict(
        title='Points by lane label', deep=True,
        note='Per-point lane-type mix. The current pipeline collapses all of '
             'these into one divider class.'),
}


def render_stat(*args, **kwargs):
    """Thread-safe wrapper -- same RENDER_LOCK as the tile renderer, for the
    same reason (see its comment at the top of the file)."""
    with RENDER_LOCK:
        return _render_stat(*args, **kwargs)


def _render_stat(kind, group_by, normalize=False):
    rows = stat_rows()
    groups = group_rows(rows, group_by)
    if kind == 'points':
        return dist_plot(groups, group_by,
                          lambda rs: [r['n_points'] for r in rs],
                          'Points per tile', 'raw points (log)', log=True)
    if kind == 'density':
        return dist_plot(groups, group_by,
                          lambda rs: [r['density'] for r in rs],
                          'Points per m² of tile', 'points / m² (log)',
                          log=True, fmt='{:,.0f}')
    if kind == 'polylines':
        zero = sum(1 for r in rows if r['n_polylines'] == 0)
        buf = dist_plot(groups, group_by,
                         lambda rs: [r['n_polylines'] for r in rs],
                         f'GT polylines per tile — {zero:,} tiles have none',
                         'polylines')
        return buf
    if kind == 'classes':
        return plot_classes(groups, group_by, normalize)
    if kind == 'effective':
        return plot_effective(groups, group_by)
    if kind == 'zrange':
        return plot_zrange(groups, group_by)
    if kind == 'drift':
        fig, ax = stat_fig(height=box_height(len(groups)))
        if box_by_group(fig, ax, groups,
                         lambda rs: [(r['deep'] or {}).get('drift')
                                      for r in rs], fmt='{:,.3f} m',
                         top_margin=0.75) is None:
            return None
        for x in (0.5, 1.0, 1.5):
            ax.axvline(x, color=CRITICAL, linewidth=0.9, alpha=0.8, zorder=4)
        annotate_below(ax, 1.5, 'chamfer thresholds\n0.5 / 1.0 / 1.5 m')
        return finish(fig, ax, '|tile_center − offset| per tile', 'metres')
    if kind == 'vertices':
        return plot_vertices(groups, group_by)
    if kind == 'lengths':
        return plot_lengths(groups, group_by)
    if kind == 'labels':
        return plot_labels(groups, group_by, normalize)
    return None


# ---- flagging ------------------------------------------------------------

def tile_flags(r):
    """Reason chips for one tile, worst first. Manifest-tier reasons are
    always available; the rest need a deep scan."""
    out = []
    d = r['deep'] or {}
    if d.get('error'):
        out.append(('error', d['error']))
    if is_degenerate(d):
        out.append(('degenerate',
                     f"{d['n_raw']:,} raw points collapse to "
                     f"{d['n_effective']:,} distinct cells"))
    eff = d.get('n_effective')
    if (eff is not None and eff < NEAR_EMPTY_CELLS) or \
            (eff is None and r['n_points'] < NEAR_EMPTY_RAW):
        out.append(('near-empty',
                     f"{eff:,} occupied cells" if eff is not None
                     else f"{r['n_points']:,} raw points"))
    if r['n_polylines'] == 0 and not d.get('n_gt'):
        out.append(('no-GT', 'no reference-line polylines'))
    zlo, zhi = STATE['pc_range_z']
    if d.get('z_min') is not None and (d['z_min'] < zlo or d['z_max'] > zhi):
        out.append(('z-out-of-range',
                     f"z {d['z_min']:.1f} .. {d['z_max']:.1f} vs "
                     f"{zlo:g} .. {zhi:g}"))
    # NB: drift is deliberately NOT a flag. It measures how far apart a
    # tile's two candidate origins are, which is a property of the tile's
    # geometry, not a defect -- on the 60 m grid export the median is 7.3 m
    # and every tile would light up for nothing. What it tells you is the
    # size of the error the old tile_center frame would introduce, so it
    # stays a chart. The actionable version of the same concern is
    # gt-outside-tile below, which fires on real misplacement.
    if d.get('gt_out'):
        out.append(('gt-outside-tile', f"{d['gt_out']} vertices"))
    return out


SEVERITY = {'error': 0, 'degenerate': 1, 'near-empty': 2, 'no-GT': 3,
            'z-out-of-range': 4, 'gt-outside-tile': 5}
CHIP_COLORS = {'error': CRITICAL, 'degenerate': CRITICAL,
                'near-empty': CRITICAL, 'no-GT': '#d29922',
                'z-out-of-range': '#d29922', 'gt-outside-tile': '#d29922'}


def suspect_rows(rows):
    out = []
    for r in rows:
        flags = tile_flags(r)
        if flags:
            out.append((r, flags))
    out.sort(key=lambda rf: (SEVERITY.get(rf[1][0][0], 9), -len(rf[1]),
                              rf[0]['uid']))
    return out


# ---- page ----------------------------------------------------------------

STATS_PAGE = """<!doctype html>
<html><head>
<title>CARLA dataset statistics</title>{refresh}
<style>{css}</style>
</head><body>
<h1>CARLA dataset viewer</h1>
{nav}
<div class="sub">{data_root} &mdash; {n_tiles:,} tiles across {n_groups}
  town(s) in {n_ds} dataset(s) &mdash; grouped by <b>{group_by}</b><br>
  {class_summary}</div>

<form method="get">
  <input type="hidden" name="tab" value="stats">
  <fieldset>
    <label class="top">Group by</label>
    <select name="group_by">{group_opts}</select>
  </fieldset>
  <fieldset>
    <label class="top">Suspect tiles shown</label>
    <input type="number" name="top" value="{top}" min="0" max="2000" step="10">
  </fieldset>
  <fieldset class="checks">
    <label class="top">Options</label>
    <label><input type="checkbox" name="normalize" value="1" {norm_checked}>
      normalise stacked charts to %</label>
  </fieldset>
  <button type="submit">Apply</button>
</form>

{scan_panel}

<h2>Summary</h2>
{summary}
<div class="note">Every number here is also in
  <a href="/stats.csv">stats.csv</a>, one row per tile.</div>

<h2>Distributions</h2>
<div class="gallery">
{gallery}
</div>

<h2>Suspect tiles{suspect_count}</h2>
{suspects}
</body></html>
"""

SCAN_IDLE = """
<form method="get" action="/scan">
  <fieldset>
    <label class="top">Deep scan</label>
    <div class="note" style="max-width:44em">{state}</div>
  </fieldset>
  <button type="submit">{button}</button>
</form>
"""

SCAN_BUSY = """
<div class="note" style="background:{panel};border:1px solid {border};
     border-radius:6px;padding:1em;margin-bottom:1.5em">
  <b style="color:{text}">Deep scan running</b> &mdash; {done:,} / {total:,}
  tiles ({pct:.0f}%){rate}
  <div class="prog"><div style="width:{pct:.1f}%"></div></div>
  This page refreshes itself every 3 s. It is safe to leave; the scan
  continues in the background and its results are cached to disk.
</div>
"""


def scan_panel_html():
    n_deep = len(STATE['deep'])
    n_tiles = len(STATE['tiles'])
    if scan_running():
        done, total = SCAN['done'], max(1, SCAN['total'])
        elapsed = time.time() - (SCAN['started'] or time.time())
        rate = ''
        if done > 5 and elapsed > 0:
            per = elapsed / done
            rate = (f' &mdash; {done / elapsed:.1f} tiles/s, about '
                    f'{(total - done) * per / 60:.1f} min left')
        return SCAN_BUSY.format(panel=BG_PANEL, border=BORDER, text=TEXT,
                                 done=done, total=SCAN['total'],
                                 pct=100.0 * done / total, rate=rate)
    if SCAN['error']:
        state = (f'Last scan failed: <code>{html.escape(SCAN["error"])}</code>. '
                 f'{n_deep:,} of {n_tiles:,} tiles have deep statistics.')
        button = 'Retry deep scan'
    elif n_deep >= n_tiles and n_tiles:
        state = (f'All {n_tiles:,} tiles scanned at grid='
                 f'{grid_str()} m. Cached under '
                 f'<code>{html.escape(STATE["cache_dir"])}</code>, so this '
                 f'survives a restart.')
        button = 'Re-scan everything'
    elif n_deep:
        state = (f'{n_deep:,} of {n_tiles:,} tiles scanned. Scanning the rest '
                 f'reads only the tiles still missing.')
        button = f'Scan remaining {n_tiles - n_deep:,}'
    else:
        state = ('Reads every tile\'s .npz to add effective point counts, '
                 'z-range, label mix, origin drift and polyline geometry. '
                 'This is the expensive tier &mdash; the full train split is '
                 'about 12 GB &mdash; but it runs in the background and is '
                 'cached, so it only ever happens once.')
        button = f'Deep scan {n_tiles:,} tiles'
    return SCAN_IDLE.format(state=state, button=button)


def summary_html(groups):
    order = class_order()
    have_deep = any(r['deep'] for rs in groups.values() for r in rs)
    head = ['group', 'tiles', 'points<br>median', 'points<br>max',
            'pts/m&sup2;<br>median', 'GT<br>median', 'GT<br>total',
            'no GT']
    if have_deep:
        head += ['scanned', 'effective<br>median', 'worst<br>eff/raw',
                 'drift<br>median', 'z min', 'z max']
    body = []
    for label, rows in groups.items():
        pts = [r['n_points'] for r in rows]
        dens = [r['density'] for r in rows]
        gts = [r['n_polylines'] for r in rows]
        nz = sum(1 for g in gts if g == 0)
        cells = [
            f'<td class="l">{html.escape(label)}</td>',
            f'<td>{len(rows):,}</td>',
            f'<td>{fmt_n(med(pts))}</td>',
            f'<td>{max(pts):,}</td>',
            f'<td>{fmt_n(med(dens))}</td>',
            f'<td>{fmt_n(med(gts))}</td>',
            f'<td>{sum(gts):,}</td>',
            f'<td>{nz:,} ({100.0 * nz / max(1, len(rows)):.0f}%)</td>',
        ]
        if have_deep:
            ds = [r['deep'] for r in rows if r['deep']]
            eff = [d['n_effective'] for d in ds if d.get('n_effective') is not None]
            ratios = [d['n_effective'] / d['n_raw'] for d in ds
                      if d.get('n_raw') and d.get('n_effective') is not None]
            drifts = [d['drift'] for d in ds if d.get('drift') is not None]
            zmin = [d['z_min'] for d in ds if d.get('z_min') is not None]
            zmax = [d['z_max'] for d in ds if d.get('z_max') is not None]
            cells += [
                f'<td>{len(ds):,}</td>',
                f'<td>{fmt_n(med(eff))}</td>',
                f'<td>{min(ratios):.3f}</td>' if ratios else '<td>&mdash;</td>',
                f'<td>{med(drifts):.3f}</td>' if drifts else '<td>&mdash;</td>',
                f'<td>{min(zmin):.1f}</td>' if zmin else '<td>&mdash;</td>',
                f'<td>{max(zmax):.1f}</td>' if zmax else '<td>&mdash;</td>',
            ]
        body.append('<tr>' + ''.join(cells) + '</tr>')

    tables = ['<table class="stats"><thead><tr>'
              + ''.join(f'<th class="l">{h}</th>' if i == 0 else f'<th>{h}</th>'
                         for i, h in enumerate(head))
              + '</tr></thead><tbody>' + '\n'.join(body) + '</tbody></table>']

    if order:
        # per-class counts get their own table: eight more columns would make
        # the summary unreadable, and this one is a table on purpose (past
        # ~7 categories a table beats more colours)
        chead = ['group'] + [
            f'<span class="swatch" style="background:'
            f'{CLASS_COLORS.get(i, CLASS_FALLBACK)}"></span>{html.escape(n)}'
            for i, n in order]
        cbody = []
        for label, rows in groups.items():
            counts = {}
            for r in rows:
                for name, k in (r['by_class'] or {}).items():
                    counts[name] = counts.get(name, 0) + int(k)
            cbody.append('<tr><td class="l">' + html.escape(label) + '</td>'
                          + ''.join(f'<td>{counts.get(n, 0):,}</td>'
                                     for _i, n in order) + '</tr>')
        tables.append('<table class="stats"><thead><tr>'
                       + ''.join(f'<th class="l">{h}</th>' if i == 0
                                  else f'<th>{h}</th>'
                                  for i, h in enumerate(chead))
                       + '</tr></thead><tbody>' + '\n'.join(cbody)
                       + '</tbody></table>')
    return '\n'.join(tables)


def suspects_html(rows, top):
    susp = suspect_rows(rows)
    if not susp:
        return ('<div class="note">Nothing flagged.'
                + ('' if STATE['deep'] else
                   ' Only the manifest-tier checks have run &mdash; run a deep '
                   'scan for the degenerate-tile, z-range and drift checks.')
                + '</div>'), 0
    out = ['<table class="stats"><thead><tr><th class="l">tile</th>'
           '<th class="l">group</th><th>points</th><th>GT</th>'
           '<th class="l">flags</th><th class="l">detail</th></tr></thead>'
           '<tbody>']
    for r, flags in susp[:top]:
        # deliberately NOT submitted=1: with no cls params that would mean
        # "show no classes", and the linked tile would render without GT
        q = urlencode([('tab', 'browse'), ('town', r['group']),
                        ('start', r['gidx']), ('count', 1),
                        ('mode', 'rgb'), ('polylines', '1')])
        chips = ''.join(
            f'<span class="chip" style="color:'
            f'{CHIP_COLORS.get(k, TEXT_MUTED)}">{html.escape(k)}</span>'
            for k, _d in flags)
        detail = '; '.join(html.escape(str(d)) for _k, d in flags)
        out.append(
            f'<tr><td class="l"><a href="/{html.escape("?" + q, quote=True)}">'
            f'{html.escape(r["name"])}</a></td>'
            f'<td class="l">{html.escape(r["town"])} ({html.escape(r["split"])})</td>'
            f'<td>{r["n_points"]:,}</td><td>{r["n_polylines"]}</td>'
            f'<td class="l">{chips}</td>'
            f'<td class="l" style="white-space:normal">{detail}</td></tr>')
    out.append('</tbody></table>')
    if len(susp) > top:
        out.append(f'<div class="note">Showing {top:,} of {len(susp):,} '
                    f'flagged tiles &mdash; raise "suspect tiles shown", or '
                    f'take the full list from <a href="/stats.csv">'
                    f'stats.csv</a>.</div>')
    return '\n'.join(out), len(susp)


def stats_page():
    group_by = request.args.get('group_by', 'town')
    if group_by not in ('town', 'split'):
        group_by = 'town'
    normalize = bool(request.args.get('normalize'))
    try:
        top = max(0, min(2000, int(request.args.get('top', 50))))
    except ValueError:
        top = 50

    rows = stat_rows()
    groups = group_rows(rows, group_by)
    have_deep = bool(STATE['deep'])

    cachebust = f'{time.time():.6f}'
    figs = []
    for kind, spec in PLOTS.items():
        if spec['deep'] and not have_deep:
            continue
        params = [('plot', kind), ('group_by', group_by), ('v', cachebust)]
        if normalize:
            params.append(('normalize', '1'))
        # urlencode + html.escape is REQUIRED, not cosmetic -- see the long
        # comment in index() about "&gt" being resolved as an entity
        q = html.escape('?' + urlencode(params), quote=True)
        figs.append(
            f'<figure style="max-width:560px">'
            f'<a href="/stat.png{q}" target="_blank">'
            f'<img src="/stat.png{q}" style="width:540px"></a>'
            f'<figcaption style="text-align:left;word-break:normal">'
            f'{spec["note"]}</figcaption></figure>')
    if not have_deep:
        figs.append(
            f'<figure style="max-width:560px;width:540px"><figcaption '
            f'style="text-align:left;word-break:normal">'
            f'Six more charts &mdash; effective vs raw points, z extent, '
            f'origin drift, polyline vertices and lengths, and the per-point '
            f'label mix &mdash; appear here once a deep scan has run.'
            f'</figcaption></figure>')

    suspects, n_susp = suspects_html(rows, top)
    page = STATS_PAGE.format(
        css=CSS, nav=nav_html('stats'),
        refresh=('<meta http-equiv="refresh" content="3">'
                  if scan_running() else ''),
        data_root=osp.abspath(STATE['data_root']),
        n_tiles=len(rows), n_groups=len(STATE['groups']),
        n_ds=len(STATE['datasets']), group_by=group_by,
        class_summary=STATE['class_summary'],
        group_opts=_opts(['town', 'split'], group_by,
                          ['town (one row per town)',
                           'split (train vs test overlay)']),
        top=top, norm_checked='checked' if normalize else '',
        scan_panel=scan_panel_html(),
        summary=summary_html(groups),
        gallery='\n'.join(figs),
        suspect_count=f' &mdash; {n_susp:,} flagged' if n_susp else '',
        suspects=suspects,
    )
    resp = make_response(page)
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    return resp


@app.route('/stat.png')
def stat_png():
    kind = request.args.get('plot')
    if kind not in PLOTS:
        abort(404)
    group_by = request.args.get('group_by', 'town')
    if group_by not in ('town', 'split'):
        group_by = 'town'
    buf = render_stat(kind, group_by,
                       normalize=bool(request.args.get('normalize')))
    if buf is None:
        abort(404)
    resp = send_file(buf, mimetype='image/png')
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    return resp


@app.route('/scan')
def scan_route():
    started = start_scan(rescan=len(STATE['deep']) >= len(STATE['tiles']))
    if not started:
        print('[stats] scan already running; ignoring request')
    return redirect('/?tab=stats')


@app.route('/stats.csv')
def stats_csv():
    rows = stat_rows()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['uid', 'name', 'dataset', 'split', 'town', 'n_points',
                 'tile_area_m2', 'points_per_m2', 'n_polylines',
                 'n_effective', 'n_xy_cells', 'effective_over_raw',
                 'z_min', 'z_max', 'drift_m', 'gt_vertices_outside_tile',
                 'flags'])
    for r in rows:
        d = r['deep'] or {}
        raw, eff = d.get('n_raw'), d.get('n_effective')
        w.writerow([
            r['uid'], r['name'], r['ds'], r['split'], r['town'],
            r['n_points'], f"{r['area']:.1f}", f"{r['density']:.3f}",
            r['n_polylines'], eff if eff is not None else '',
            d.get('n_xy_cells', ''),
            f'{eff / raw:.5f}' if raw and eff is not None else '',
            d.get('z_min', ''), d.get('z_max', ''),
            f"{d['drift']:.4f}" if d.get('drift') is not None else '',
            d.get('gt_out', ''),
            ' '.join(k for k, _v in tile_flags(r)),
        ])
    resp = make_response(buf.getvalue())
    resp.headers['Content-Type'] = 'text/csv; charset=utf-8'
    resp.headers['Content-Disposition'] = 'attachment; filename=stats.csv'
    return resp


# ----------------------------------------------------------- training results
#
# This tab answers "which tiles is the model bad at, and what do they have in
# common?". That needs a PER-TILE quality score, which the repo's own eval
# does not report -- it prints one dataset-wide mAP.
#
# The good news is that the split is natural: mean_ap.eval_map() calls
# custom_tpfp_gen() once per tile and only the AP *aggregation* is global. So
# every tile's TP/FP vector here is exactly the one the real eval computes;
# the only thing this file adds is running the same AP formula over one
# tile's detections instead of all of them.
#
# What "per-tile AP" therefore means, precisely: rank that tile's own
# predictions by confidence, match them greedily against that tile's own GT,
# and integrate the resulting PR curve against that tile's GT count. It is a
# LOCAL score -- the tile's own AP, not its contribution to the global one,
# which is not a well-defined per-tile quantity because global ranking
# interleaves detections from every tile. The two agree on what a good tile
# looks like, which is all this tab needs. The global number is computed too
# (with the real interleaved ranking) and shown in the header, so it can be
# checked against the training log's CarlaMap_chamfer/mAP.
#
# Reimplemented in numpy rather than imported because this viewer runs on the
# HOST, outside the container: there is no torch, no mmdet3d, and neither
# shapely nor scipy. Verified against the real implementation -- see
# eval_matches()'s note on the buffer prefilter, and CLAUDE.md.

# mean_ap.py's `thresholds` for metric='chamfer'.
CHAMFER_THRESHOLDS = (0.5, 1.0, 1.5)
# mean_ap.format_res_gt_by_classes()'s num_fixed_sample_pts: every GT line is
# resampled to this many equally spaced points before matching. Predictions
# are NOT resampled (eval_use_same_gt_sample_num_flag defaults False), so they
# arrive at whatever fixed_ptsnum_per_pred_line the model emitted.
GT_NUM_SAMPLE = 100
# The threshold at which "correctly predicted polylines" is counted. 1.0 m is
# the middle of the three, and the one whose AP tracks the mean most closely.
TP_THRESHOLD = 1.0

REval = {'key': None, 'rows': None}
REVAL_LOCK = threading.Lock()


def resample_line(pts, n=GT_NUM_SAMPLE):
    """Resample a polyline to n equally spaced points along its arc length.

    Equivalent to shapely's
        [line.interpolate(d) for d in np.linspace(0, line.length, n)]
    which is what get_cls_results() does to every GT line -- LineString
    interpolation is linear between vertices, same as this.
    """
    pts = np.asarray(pts, dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        # a degenerate all-identical-vertices line; shapely returns the same
        # point n times here too
        return np.repeat(pts[:1], n, axis=0)
    want = np.linspace(0.0, total, n)
    idx = np.clip(np.searchsorted(cum, want, side='right') - 1,
                   0, len(seg) - 1)
    frac = (want - cum[idx]) / np.maximum(seg[idx], 1e-12)
    return pts[idx] + frac[:, None] * (pts[idx + 1] - pts[idx])


def chamfer_score_matrix(preds, gts):
    """(num_preds, num_gts) matrix of NEGATED symmetric chamfer distance.

    Mirrors tpfp_chamfer.custom_polyline_score(metric='chamfer'):
        ab = cdist(pred_pts, gt_pts).min(-1).mean()   # pred -> gt
        ba = cdist(pred_pts, gt_pts).min(-2).mean()   # gt -> pred
        score = -(ab + ba) / 2
    Negated because the caller compares against -threshold, so "higher is
    better" holds for both chamfer and iou.

    The one deliberate difference: the real version prefilters pairs with an
    STRtree, buffers each line by 2 m, and leaves non-intersecting pairs at
    the sentinel -100; this computes every pair. That cannot change any
    match. Two lines whose 2 m buffers do not intersect are more than 4 m
    apart at every point, so both ab and ba exceed 4 and the score is below
    -4 -- past even the loosest -1.5 threshold. Computing them outright just
    avoids needing shapely on the host (and avoids gotcha #6, the shapely
    2.0 STRtree API break, entirely).
    """
    n_p, n_g = len(preds), len(gts)
    out = np.full((n_p, n_g), -100.0)
    if not n_p or not n_g:
        return out
    gt_stack = np.asarray(gts, dtype=np.float64)          # (G, g, 2)
    flat = gt_stack.reshape(-1, 2)                        # (G*g, 2)
    g_pts = gt_stack.shape[1]
    # One pred at a time: the full 4-D broadcast would be
    # P*G*p*g floats, which on a dense tile is large enough to matter and
    # buys nothing -- this is already vectorised over every GT.
    for i, pred in enumerate(preds):
        pred = np.asarray(pred, dtype=np.float64)
        d = np.linalg.norm(pred[:, None, :] - flat[None, :, :], axis=-1)
        d = d.reshape(pred.shape[0], n_g, g_pts)          # (p, G, g)
        ab = d.min(axis=2).mean(axis=0)                   # (G,)
        ba = d.min(axis=0).mean(axis=1)                   # (G,)
        out[i] = -(ab + ba) / 2.0
    return out


def tpfp_from_matrix(matrix, scores, threshold):
    """Per-tile TP/FP, mirroring tpfp.custom_tpfp_gen() line for line.

    Greedy by descending confidence; each prediction takes its single
    best-scoring GT (argmax over the row, chosen BEFORE any matching, exactly
    as upstream does), and a GT already claimed by a higher-scoring
    prediction turns the later one into a false positive rather than letting
    it fall through to its second choice.
    """
    n_p, n_g = matrix.shape
    tp = np.zeros(n_p, dtype=np.float64)
    fp = np.zeros(n_p, dtype=np.float64)
    if n_g == 0:
        fp[...] = 1
        return tp, fp, np.full(n_p, -1, dtype=int)
    if n_p == 0:
        return tp, fp, np.zeros(0, dtype=int)
    thr = -abs(threshold)
    row_max = matrix.max(axis=1)
    row_argmax = matrix.argmax(axis=1)
    matched = np.full(n_p, -1, dtype=int)
    covered = np.zeros(n_g, dtype=bool)
    for i in np.argsort(-scores):
        if row_max[i] >= thr and not covered[row_argmax[i]]:
            covered[row_argmax[i]] = True
            tp[i] = 1
            matched[i] = row_argmax[i]
        else:
            fp[i] = 1
    return tp, fp, matched


def average_precision(recalls, precisions):
    """mean_ap.average_precision(mode='area'), for a single scale."""
    mrec = np.concatenate([[0.0], recalls, [1.0]])
    mpre = np.concatenate([[0.0], precisions, [0.0]])
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ind = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[ind + 1] - mrec[ind]) * mpre[ind + 1]))


def ap_from_tpfp(tp, fp, scores, num_gts):
    """Rank by score, accumulate, integrate. Used for both the per-tile AP
    (one tile's detections) and the global one (every tile's, interleaved --
    which is what eval_map does and what the training log reports)."""
    if num_gts == 0:
        # AP is undefined without GT: the recall denominator is zero. Upstream
        # sidesteps this by only ever aggregating over the whole dataset.
        return None
    if len(tp) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    ctp = np.cumsum(np.asarray(tp)[order])
    cfp = np.cumsum(np.asarray(fp)[order])
    eps = np.finfo(np.float32).eps
    return average_precision(ctp / max(num_gts, eps),
                              ctp / np.maximum(ctp + cfp, eps))


# The map classes, in order -- the index IS the label id, matching the
# configs' `cat2id` ({'divider': 0}) and the `labels` in
# submission_vector.json. The CARLA taxonomy is divider-only today; if that
# widens, this has to widen with it, in the same order, or every id shifts.
MAPCLASSES = ('divider',)


def load_gt(path):
    """{sample_token: [(pts, type)]} from either GT source.

    Two files can serve as eval GT, and they agree by construction:

      * carla_map_gt.json -- what _format_gt() writes to the config's
        `map_ann_file`. Only exists once an eval or tools/test.py run has
        happened.
      * carla_map_infos_<split>.pkl -- what the converter writes. Exists as
        soon as the dataset has been converted, i.e. always, and long
        before any training has run.

    The pkl is preferred when both are offered because a fresh dataset has
    no json yet, and because the json is the one that goes stale: _format_gt
    skips regeneration if the file already exists, so a json left over from
    a previous sample set or GT frame silently scores against the wrong GT.
    """
    cached = STATE['gt_cache'].get(path)
    if cached is not None:
        return cached
    out = (load_gt_pkl(path) if path.endswith('.pkl')
           else load_gt_json(path))
    STATE['gt_cache'][path] = out
    return out


def load_gt_json(path):
    """Parse carla_map_gt.json, as written by
    CustomCarlaLocalMapDataset._format_gt()."""
    with open(path) as f:
        blob = json.load(f)
    out = {}
    for entry in blob.get('GTs', []):
        vecs = []
        for v in entry.get('vectors', []):
            pts = np.asarray(v.get('pts', []), dtype=np.float64)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            vecs.append((pts[:, :2], int(v.get('type', 0))))
        out[entry.get('sample_token')] = vecs
    return out


def load_gt_pkl(path):
    """Build the same GT straight from the converter's annotation pkl.

    This reproduces _format_gt() rather than approximating it. That is
    possible because the chain from pkl to GT json turns out to be almost
    an identity:

        gen_vectorized_samples() wraps each annotation array in a
        LineString and keeps it if it has >= 2 points and its class maps to
        a label other than -1; LiDARInstanceLines stores that list
        untouched (`self.instance_list = instance_line_list`); and
        _format_gt writes `np.array(list(gt_vec.coords))[:, :code_size]`.

    So: no resampling, no clipping to pc_range, no reordering -- just the
    pkl's own arrays, filtered by point count and class, truncated to 2D.
    Verified by reproducing an existing carla_map_gt.json exactly (259
    tiles, 1210 instances, every coordinate bit-identical).

    Plain pickle + numpy: the pkl holds only dicts, strings and numpy
    arrays, so this does NOT need mmcv or mmdet3d, which is what makes it
    loadable on the host.
    """
    with open(path, 'rb') as f:
        blob = pickle.load(f)
    samples = blob.get('samples') if isinstance(blob, dict) else None
    if samples is None:
        raise ValueError(
            f'{path} does not look like a CARLA annotation pkl (no '
            f'"samples" key; found {list(blob)[:6] if isinstance(blob, dict) else type(blob).__name__})')
    out = {}
    frames = set()
    for s in samples:
        ann = s.get('annotation') or {}
        shift = gt_frame_shift(s, frames)
        vecs = []
        for label, cls in enumerate(MAPCLASSES):
            for inst in ann.get(cls, []):
                pts = np.asarray(inst, dtype=np.float64)
                # the <2-point filter is gen_vectorized_samples's own
                if pts.ndim != 2 or pts.shape[0] < 2:
                    continue
                vecs.append((pts[:, :2] - shift, label))
        out[s.get('sample_idx')] = vecs
    if frames - {'tile_center'}:
        print(f'[gt] {osp.basename(path)}: annotations are in the '
              f'{"/".join(sorted(frames))} frame; converted to the '
              'tile-centred frame this repo scores in')
    return out


def gt_frame_shift(sample, frames=None):
    """xy shift that brings one pkl sample's GT into the tile-centred frame.

    Annotation pkls from this repo and from the sibling MapTRv2/GeMap trees
    look identical -- same `samples` list, same `annotation['divider']`
    arrays -- but are written in DIFFERENT frames, and nothing in the arrays
    themselves says which. Scoring the wrong one is silent: every polyline
    is off by `tile_center - offset`, which is 1.6 m on average and up to
    12.1 m on the 25 m export, so tiles just look badly predicted.

    Both roots get searched here (data/carla is usually the shared tile
    export, which is where the sibling repos put their pkls too), so the
    frame is detected per sample from the keys the writers leave behind:

      * `tile_shift`        -- this repo's converter. Already tile-centred.
      * `annotation_origin` -- the sibling converters since GeMap 7f860b0,
                               where it records the block `offset` the GT was
                               shifted by. Convert with
                               `tile_center - annotation_origin`.
      * neither             -- a pre-7f860b0 pkl, which subtracted
                               `tile_center`, i.e. already tile-centred.
                               (Verified against the copy in data/carla:
                               bit-identical to this repo's own.)
    """
    if sample.get('tile_shift') is not None:
        if frames is not None:
            frames.add('tile_center')
        return np.zeros(2)
    origin = sample.get('annotation_origin')
    center = sample.get('tile_center')
    if origin is None or center is None:
        if frames is not None:
            frames.add('tile_center')
        return np.zeros(2)
    origin = np.asarray(origin, dtype=np.float64)[:2]
    center = np.asarray(center, dtype=np.float64)[:2]
    if np.allclose(origin, center):
        if frames is not None:
            frames.add('tile_center')
        return np.zeros(2)
    if frames is not None:
        frames.add('offset')
    return center - origin


def discover_gt(work_dir=None):
    """Every usable eval GT, best first.

    Annotation pkls come FIRST, deliberately. A freshly converted dataset
    has no carla_map_gt.json at all -- that file only appears once an eval
    or tools/test.py run has written it -- whereas the pkl is the
    converter's own output and always exists. The pkl is also the safer of
    the two when both are present: _format_gt() skips regeneration if the
    json already exists, so a json left over from an earlier sample set or
    GT frame silently scores against the wrong GT, while the pkl is
    regenerated every time the converter runs.

    Searched: data/carla_infos (this repo's ann_file) and data/carla,
    and anything named like a GT under the work-dir.
    """
    cands = []
    for ann_dir in ('data/carla_infos', 'data/carla'):
        if not osp.isdir(ann_dir):
            continue
        cands.extend(osp.join(ann_dir, fn)
                      for fn in sorted(os.listdir(ann_dir))
                      if fn.startswith('carla_map_infos')
                      and fn.endswith('.pkl'))
    cands.append('data/carla/carla_map_gt.json')  # MapTRv2-style, if present
    for root in filter(None, [work_dir, osp.join('val', work_dir or '_')]):
        if not osp.isdir(root):
            continue
        for dirpath, _d, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith('.pkl') and 'infos' in fn:
                    cands.append(osp.join(dirpath, fn))
                elif fn.endswith('.json') and 'gt' in fn.lower():
                    cands.append(osp.join(dirpath, fn))
    seen, out = set(), []
    for c in cands:
        c = osp.abspath(c)
        if c not in seen and osp.isfile(c):
            seen.add(c)
            out.append(c)
    return out


def best_gt_for(results_path, gt_files):
    """Pick the GT file that actually covers these predictions.

    There is normally one annotation pkl per split, and nothing in a results
    json says which split it came from. Defaulting to the first candidate
    would happily score test predictions against the train pkl and report
    every tile as missing GT, which reads like a broken tab rather than the
    wrong dropdown entry. Token overlap settles it unambiguously: splits are
    disjoint sets of tiles.
    """
    if len(gt_files) == 1:
        return gt_files[0]
    tokens = set(load_results(results_path))
    best, best_n = gt_files[0], -1
    for path in gt_files:
        try:
            n = len(tokens & set(load_gt(path)))
        except Exception:  # noqa: BLE001 -- a broken candidate must not
            continue      # take down the page; the others may be fine
        if n > best_n:
            best, best_n = path, n
    return best


def eval_tile(preds, gts, score_thresh=0.0):
    """Every per-tile number this tab can sort or plot by.

    `preds` is [(pts, score, cls)] straight from load_results(); `gts` is
    [(pts, type)] from load_gt(). Classes are not separated: the CARLA
    taxonomy is divider-only, so class-conditional AP would be the same
    number three times. If a real multi-class taxonomy lands, this is the
    place that has to grow a per-class loop (mirroring eval_map's).

    `score_thresh` affects ONLY the count metrics (n_pred_kept,
    count_error). AP, TP and recall deliberately ignore it, because AP
    integrates over every detection at every operating point -- applying a
    threshold first would silently redefine the metric and stop it matching
    the training log.
    """
    pred_pts = [p for p, _s, _c in preds]
    scores = np.array([s for _p, s, _c in preds], dtype=np.float64)
    gt_pts = [resample_line(g, GT_NUM_SAMPLE) for g, _t in gts]
    n_gt, n_pred = len(gt_pts), len(pred_pts)

    matrix = chamfer_score_matrix(pred_pts, gt_pts)
    per_thr, tpfp = {}, {}
    for thr in CHAMFER_THRESHOLDS:
        tp, fp, matched = tpfp_from_matrix(matrix, scores, thr)
        tpfp[thr] = (tp, fp, matched)
        per_thr[thr] = ap_from_tpfp(tp, fp, scores, n_gt)

    aps = [v for v in per_thr.values() if v is not None]
    tp, fp, matched = tpfp[TP_THRESHOLD]
    n_tp = int(tp.sum())
    # Chamfer distance of the matched pairs -- how good the hits are, as
    # opposed to how many there are. Sign flipped back to a real distance.
    dists = [-matrix[i, m] for i, m in enumerate(matched) if m >= 0]

    # How many polylines the model actually claims exist here. The raw
    # count is useless on its own -- the head emits a fixed num_vec every
    # time, so n_pred is the same constant for every tile -- which is why
    # this is the thresholded count, and why count_error moves with the
    # score threshold. See count_health() for the case where no threshold
    # separates anything.
    n_kept = int((scores >= score_thresh).sum())

    return {
        'n_gt': n_gt, 'n_pred': n_pred, 'n_tp': n_tp,
        'n_pred_kept': n_kept,
        # Signed, so over- and under-prediction stay distinguishable:
        # 0 = exactly right, -1 = one too few, +1 = one too many.
        'count_error': n_kept - n_gt,
        'ap': float(np.mean(aps)) if aps else None,
        'ap_thr': per_thr,
        'recall': (n_tp / n_gt) if n_gt else None,
        # Precision over ALL predictions, unthresholded, to stay consistent
        # with AP -- which also integrates over every detection. The score
        # threshold in this tab only ever affects what is DRAWN.
        'precision': (n_tp / n_pred) if n_pred else None,
        'chamfer': float(np.median(dists)) if dists else None,
        # kept per threshold so global_ap() can pool them across tiles
        'scores': scores,
        'tp_thr': {t: (v[0], v[1]) for t, v in tpfp.items()},
    }


def eval_key(results_path, gt_path, score_thresh):
    # score_thresh is part of the key because it changes count_error
    return (results_path, gt_path, round(float(score_thresh), 6))


def evaluate_run(results_path, gt_path, score_thresh=0.0):
    """Per-tile metrics for one (results, GT) pair, cached in memory.

    Serialised and cached because it is O(preds x gts x points) over every
    tile -- fast (a few seconds for the 259-tile test split) but not
    something to redo for each of a dozen images on one page load.
    """
    key = eval_key(results_path, gt_path, score_thresh)
    with REVAL_LOCK:
        if REval['key'] == key:
            return REval['rows']
        preds_by_token = load_results(results_path)
        gt_by_token = load_gt(gt_path)
        # A results json identifies a tile by bare `sample_idx`, which is
        # unique only WITHIN a dataset -- two grid exports both contain
        # tile_00000 (the reason the rest of this viewer keys by
        # '<dataset>/<name>'). Nothing in the json says which export it came
        # from, so an ambiguous name is resolved to the first match and
        # counted, and the count is surfaced on the page rather than left to
        # silently attach predictions to the wrong town's tile.
        by_name = {}
        for r in stat_rows():
            by_name.setdefault(r['name'], []).append(r)
        ambiguous = sum(1 for v in by_name.values() if len(v) > 1)

        rows, missing_gt, missing_tile = [], 0, 0
        t0 = time.time()
        for token, preds in preds_by_token.items():
            gts = gt_by_token.get(token)
            if gts is None:
                missing_gt += 1
                continue
            cands = by_name.get(token)
            base = cands[0] if cands else None
            if base is None:
                # a results json from a different dataset than --data-root
                missing_tile += 1
                continue
            m = eval_tile(preds, gts, score_thresh)
            row = dict(base)
            row.update(m)
            row['token'] = token
            d = row.get('deep') or {}
            for k in DEEP_FIELDS:
                row[k] = d.get(k)
            rows.append(row)

        glob = global_ap(rows)
        REval.update(key=key, rows=rows, global_ap=glob, ambiguous=ambiguous,
                      missing_gt=missing_gt, missing_tile=missing_tile,
                      n_gt_total=sum(r['n_gt'] for r in rows),
                      n_pred_total=sum(r['n_pred'] for r in rows),
                      elapsed=time.time() - t0)
        return rows


def global_ap(rows):
    """The real, interleaved-ranking AP -- every tile's detections pooled and
    ranked together, exactly as eval_map does. Reported so this tab can be
    checked against the training log rather than taken on faith."""
    out = {}
    n_gts = sum(r['n_gt'] for r in rows)
    for thr in CHAMFER_THRESHOLDS:
        tp = np.concatenate([r['tp_thr'][thr][0] for r in rows]) if rows \
            else np.zeros(0)
        fp = np.concatenate([r['tp_thr'][thr][1] for r in rows]) if rows \
            else np.zeros(0)
        sc = np.concatenate([r['scores'] for r in rows]) if rows \
            else np.zeros(0)
        out[thr] = ap_from_tpfp(tp, fp, sc, n_gts)
    vals = [v for v in out.values() if v is not None]
    out['mean'] = float(np.mean(vals)) if vals else None
    return out


# ---- what can be ranked, and what can be plotted against it ---------------
#
# `better` is what makes "Best / Median / Worst" meaningful for keys where
# high is not good: chamfer distance is a distance, so its best tiles are its
# smallest. Keys with better=None are descriptive (how big is this tile) and
# are offered as sort keys anyway, because "show me the worst tiles by GT
# count" is a legitimate thing to ask.
RMETRICS = {
    'ap': dict(label='AP (mean of 0.5/1.0/1.5 m)', fmt='{:.3f}',
                better='high', axis='per-tile AP'),
    'ap_0.5': dict(label='AP @ 0.5 m', fmt='{:.3f}', better='high',
                    axis='per-tile AP @ 0.5 m'),
    'ap_1.0': dict(label='AP @ 1.0 m', fmt='{:.3f}', better='high',
                    axis='per-tile AP @ 1.0 m'),
    'ap_1.5': dict(label='AP @ 1.5 m', fmt='{:.3f}', better='high',
                    axis='per-tile AP @ 1.5 m'),
    # Renamed from "correctly predicted polylines", which read as "did it
    # predict the right NUMBER" -- a different question, now answered by
    # count_error below.
    'n_tp': dict(label='matched polylines (TP @ 1.0 m)', fmt='{:.0f}',
                  better='high', axis='matched polylines (TP @ 1.0 m)'),
    'count_error': dict(
        label='polyline count error (pred − GT)', fmt='{:+.0f}',
        better='zero', axis='count error (predicted − GT)'),
    'n_pred_kept': dict(label='predictions above score threshold',
                         fmt='{:.0f}', better=None,
                         axis='predictions ≥ score threshold'),
    'recall': dict(label='recall (TP / GT)', fmt='{:.3f}', better='high',
                    axis='recall @ 1.0 m'),
    'precision': dict(label='precision (TP / predictions)', fmt='{:.3f}',
                       better='high', axis='precision @ 1.0 m'),
    'chamfer': dict(label='median chamfer of matched pairs', fmt='{:.3f} m',
                     better='low', axis='median matched chamfer (m)'),
    'n_gt': dict(label='GT polylines', fmt='{:.0f}', better=None,
                  axis='GT polylines'),
    'n_pred': dict(label='predictions (raw, = num_vec)', fmt='{:.0f}',
                    better=None, axis='raw predictions'),
    'n_points': dict(label='raw points', fmt='{:,.0f}', better=None,
                      axis='raw points'),
    'density': dict(label='points / m²', fmt='{:,.1f}', better=None,
                     axis='points / m²'),
}

# The quality axes the scatters are drawn against. AP is the eval's own
# summary; matched-polyline count is the one a human can check by eye
# against a rendered tile, and unlike AP it does not collapse to 0 for a
# tile whose single GT line was missed; count error asks the separate
# question of whether the model even knows HOW MANY lines are present,
# independently of where it puts them.
RQUALITY = ('ap', 'n_tp', 'count_error')

RX_VARS = {
    'density': dict(label='points / m² of tile', log=True, deep=False,
                     note='Sparse tiles are the ones with least evidence per '
                          'metre of road. The comparable-across-exports '
                          'version of raw point count.'),
    'n_points': dict(label='raw points in tile', log=True, deep=False,
                      note='Raw .npz count. The column at 5,000,000 is the '
                           'export cap, and most of those tiles are '
                           'degenerate -- see the effective-points version.'),
    'n_gt': dict(label='GT polylines in tile', log=False, deep=False,
                  note='Whether the model simply does worse where there is '
                       'more map to get right. Also the denominator of '
                       'recall, so a strong trend here can be an artefact of '
                       'the metric rather than of the model.'),
    'n_effective': dict(label='effective points (post grid-sampling)',
                         log=True, deep=True,
                         note='What the model actually sees, after '
                              'GridSamplePoints. The honest version of '
                              '"how much LiDAR did this tile have".'),
    'z_median': dict(label='tile median z, world (m)', log=False, deep=True,
                      note='Elevation of the tile\'s road surface &mdash; the '
                           'median is robust, and most returns are ground. '
                           'Against AP this asks whether hilly parts of a '
                           'town are harder. Note the model cannot see '
                           'absolute elevation (its input is '
                           'centroid-relative), so a trend here is about '
                           '<i>where</i> the tile is, not its height as '
                           'such.'),
    'z_mean': dict(label='tile mean z, world (m)', log=False, deep=True,
                    note='Elevation of the tile\'s centroid. Because '
                         '<code>offset</code> IS the centroid, this is very '
                         'nearly <code>offset[2]</code> itself, and the '
                         'centroid-relative mean is identically zero &mdash; '
                         'which is why this is reported in world z. Its gap '
                         'from the median is the vertical skew, and that gap '
                         'is the only part of this the model actually sees.'),
    'count_error': dict(label='polyline count error (predicted − GT)',
                         log=False, deep=False,
                         note='Does getting the NUMBER of polylines right '
                              'go with getting them in the right place? A '
                              'peak at count error 0 would say the two go '
                              'together; a flat line says the count is '
                              'independent of placement quality.'),
}


# Deep-scan fields copied onto each evaluated row so they can be plotted
# against prediction quality. Keep in step with the deep=True entries in
# RX_VARS (and with CACHE_VERSION, if a new one is added to scan_tile).
DEEP_FIELDS = ('n_effective', 'z_median', 'z_mean')


def r_metric(row, key):
    """One metric off an evaluated row, or None where it is undefined."""
    if key.startswith('ap_'):
        return row['ap_thr'].get(float(key[3:]))
    return row.get(key)


def corr(x, y):
    """(pearson, spearman, n) over the finite pairs.

    Spearman is just Pearson on the ranks, which is worth having alongside:
    these relationships are expected to be monotonic but not linear (AP is
    bounded in [0,1] and point count spans three decades), and Pearson alone
    would understate them.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None, None, len(x)
    pear = float(np.corrcoef(x, y)[0, 1])
    spear = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    return pear, spear, len(x)


def rankdata(a):
    """Average-tie ranks -- scipy.stats.rankdata, which isn't available on
    the host. Ties matter here: n_gt and n_tp are small integers with heavy
    ties, and ordinal ranking would fabricate an ordering within them."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    srt = a[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def binned_median(x, y, nbins=10):
    """Median y per x-quantile bin -- the trend line on the scatters.

    Quantile bins rather than equal-width ones because every x here is
    heavily skewed (point counts span three decades); equal-width bins would
    put almost every tile in the first bin and draw a line through noise.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 20:
        return None, None
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    if len(edges) < 3:
        return None, None
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    cx, cy = [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() >= 3:
            cx.append(float(np.median(x[sel])))
            cy.append(float(np.median(y[sel])))
    return (np.array(cx), np.array(cy)) if len(cx) >= 2 else (None, None)


def coincident_counts(x, y):
    """How many points share each (x, y), returned per point.

    Exact equality is the right test: the axes where this matters
    (GT count, TP count) are integers, and the continuous ones effectively
    never collide, so this is a no-op there.
    """
    seen = {}
    for xi, yi in zip(x, y):
        seen[(xi, yi)] = seen.get((xi, yi), 0) + 1
    return np.array([seen[(xi, yi)] for xi, yi in zip(x, y)],
                     dtype=np.float64)


def plot_scatter(rows, xkey, ykey):
    """Quality against one tile property, one point per tile.

    Emphasis rather than categorical colour: every tile is one muted hue,
    and only the tiles the dataset-statistics tab already flags as suspect
    are drawn in the status colour. That makes the question "are the bad
    tiles the broken ones?" readable straight off the chart, which is the
    whole reason this tab sits next to that one.
    """
    xs = RX_VARS[xkey]
    ym = RMETRICS[ykey]
    pts = [(r, r.get(xkey), r_metric(r, ykey)) for r in rows]
    pts = [(r, x, y) for r, x, y in pts
           if x is not None and y is not None and np.isfinite(x)
           and np.isfinite(y) and (x > 0 or not xs['log'])]
    if len(pts) < 3:
        return None
    x = np.array([p[1] for p in pts], dtype=np.float64)
    y = np.array([p[2] for p in pts], dtype=np.float64)
    flagged = np.array([bool(tile_flags(p[0])) for p in pts])

    fig, ax = stat_fig(height=4.0, width=6.2)
    # Marker AREA is proportional to how many tiles share that exact
    # coordinate. Several of these axis pairs are small integers -- GT count
    # against TP count is a lattice -- so a plain scatter drew 259 tiles as
    # about 40 visible dots and hid its own distribution completely. Sizing
    # by the coincident count is exact, unlike jitter, which would invent
    # positions on an axis whose integer values are the whole point.
    mult = coincident_counts(x, y)
    sizes = 11.0 * mult
    ax.scatter(x[~flagged], y[~flagged], s=sizes[~flagged], c=MUTED_MARK,
                alpha=0.75, linewidths=0, zorder=3)
    if flagged.any():
        ax.scatter(x[flagged], y[flagged], s=sizes[flagged] * 1.4, c=CRITICAL,
                    alpha=0.9, linewidths=0, zorder=4)
    # Explicit fixed-size handles: the scatter's own would be drawn at
    # whatever size the largest coincident stack happens to be, which turns
    # the legend into a row of blobs and implies a meaning the swatch does
    # not have.
    handles = [Line2D([], [], marker='o', linestyle='', color=MUTED_MARK,
                       markersize=4.5,
                       label=f'tile ({int((~flagged).sum())})')]
    if flagged.any():
        handles.append(Line2D([], [], marker='o', linestyle='', color=CRITICAL,
                               markersize=4.5,
                               label=f'flagged by stats tab '
                                     f'({int(flagged.sum())})'))
    if mult.max() > 1:
        handles.append(Line2D([], [], marker='o', linestyle='',
                               color=MUTED_MARK, markersize=7.5,
                               label=f'…{int(mult.max())} coincident tiles '
                                     f'(area ∝ count)'))
    bx, by = binned_median(x, y)
    if bx is not None:
        ax.plot(bx, by, color=ACCENT, linewidth=1.8, zorder=5,
                 marker='o', markersize=3.2)
        handles.append(Line2D([], [], color=ACCENT, linewidth=1.8, marker='o',
                               markersize=3.2, label='median per decile'))
    if xs['log']:
        ax.set_xscale('log')
    # A signed error has a meaningful origin, so mark it -- "is the cloud
    # centred on zero?" is the whole question for count error, and without
    # the rule the eye has no reference to judge it against.
    if RMETRICS.get(xkey, {}).get('better') == 'zero':
        ax.axvline(0, color=TEXT_MUTED, linewidth=0.9, alpha=0.9, zorder=2)
    if ym['better'] == 'zero':
        ax.axhline(0, color=TEXT_MUTED, linewidth=0.9, alpha=0.9, zorder=2)
    ax.set_ylabel(ym['axis'], color=TEXT_MUTED, fontsize=8)
    pear, spear, n = corr(x, y)
    stat = ('too few points' if pear is None else
            f'Spearman ρ = {spear:+.2f}   Pearson r = {pear:+.2f}   n = {n:,}')
    ax.set_title(f'{ym["axis"]} vs {xs["label"]}\n{stat}',
                  color=TEXT, fontsize=9.5, loc='left', pad=9)
    legend(ax, loc='best', handles=handles)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, facecolor=BG)
    buf.seek(0)
    return buf


def plot_counts(rows):
    """Predicted polyline count against GT count, with the identity line.

    The honest counterpart to count_error-vs-GT-count: neither axis is
    defined in terms of the other, so the correlation here is a real one.
    Distance from the diagonal IS the count error, read vertically.
    """
    x = np.array([r['n_gt'] for r in rows], dtype=np.float64)
    y = np.array([r['n_pred_kept'] for r in rows], dtype=np.float64)
    if len(x) < 3:
        return None
    fig, ax = stat_fig(height=4.0, width=6.2)
    mult = coincident_counts(x, y)
    ax.scatter(x, y, s=11.0 * mult, c=MUTED_MARK, alpha=0.75, linewidths=0,
                zorder=3)
    hi = max(x.max(), y.max())
    ax.plot([0, hi], [0, hi], color=ACCENT, linewidth=1.3, linestyle='--',
             zorder=4)
    bx, by = binned_median(x, y)
    handles = [Line2D([], [], marker='o', linestyle='', color=MUTED_MARK,
                       markersize=4.5, label=f'tile ({len(x):,})'),
               Line2D([], [], color=ACCENT, linewidth=1.3, linestyle='--',
                       label='exactly right (y = x)')]
    if mult.max() > 1:
        handles.insert(1, Line2D([], [], marker='o', linestyle='',
                                  color=MUTED_MARK, markersize=7.5,
                                  label=f'…{int(mult.max())} coincident '
                                        f'tiles (area ∝ count)'))
    if bx is not None:
        ax.plot(bx, by, color='#f0883e', linewidth=1.8, marker='o',
                 markersize=3.2, zorder=5)
        handles.append(Line2D([], [], color='#f0883e', linewidth=1.8,
                               marker='o', markersize=3.2,
                               label='median per decile'))
    exact = int((x == y).sum())
    pear, spear, n = corr(x, y)
    stat = ('too few points' if pear is None else
            f'Spearman ρ = {spear:+.2f}   Pearson r = {pear:+.2f}   n = {n:,}')
    ax.set_ylabel('predictions ≥ score threshold', color=TEXT_MUTED,
                   fontsize=8)
    ax.set_title(f'Predicted vs GT polyline count — {exact:,} exactly right\n'
                  f'{stat}', color=TEXT, fontsize=9.5, loc='left', pad=9)
    legend(ax, loc='best', handles=handles)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, facecolor=BG)
    buf.seek(0)
    return buf


def plot_quality_by_group(rows, ykey, group_by):
    """Per-tile quality distribution per town/split -- the check on whether a
    single bad town is carrying the whole number."""
    groups = group_rows(rows, group_by)
    fig, ax = stat_fig(height=box_height(len(groups)))
    ym = RMETRICS[ykey]
    got = box_by_group(fig, ax, groups,
                        lambda rs: [v for v in (r_metric(r, ykey) for r in rs)
                                    if v is not None],
                        fmt=ym['fmt'].replace(',', ''))
    if got is None:
        return None
    return finish(fig, ax, f'{ym["axis"]}, by {group_by}', ym['axis'])


def plot_pr_curve(rows):
    """The GLOBAL precision-recall curve, one line per chamfer threshold.

    Interleaved ranking across every tile -- i.e. the real eval, not the
    per-tile approximation the rest of this tab sorts by. It is here as the
    cross-check: the AP printed in its legend is directly comparable to the
    training log's CarlaMap_chamfer/*_AP_thr_* entries.

    Ordered scale (0.5 < 1.0 < 1.5 m), so it gets one hue in three steps
    rather than three categorical colours.
    """
    n_gts = sum(r['n_gt'] for r in rows)
    if not rows or not n_gts:
        return None
    # one hue in three steps, low->high threshold. The darkest step still has
    # to clear the page background (#0d1117); a genuinely dark navy reads as
    # an absent line here, which is why this ramp starts mid-hue rather than
    # at the bottom of the scale.
    shades = ['#1f6feb', ACCENT, '#a5d6ff']
    fig, ax = stat_fig(height=4.0, width=6.2)
    for i, thr in enumerate(CHAMFER_THRESHOLDS):
        tp = np.concatenate([r['tp_thr'][thr][0] for r in rows])
        fp = np.concatenate([r['tp_thr'][thr][1] for r in rows])
        sc = np.concatenate([r['scores'] for r in rows])
        order = np.argsort(-sc)
        ctp, cfp = np.cumsum(tp[order]), np.cumsum(fp[order])
        eps = np.finfo(np.float32).eps
        rec = ctp / max(n_gts, eps)
        prec = ctp / np.maximum(ctp + cfp, eps)
        ap = ap_from_tpfp(tp, fp, sc, n_gts)
        ax.plot(rec, prec, color=shades[i], linewidth=1.6,
                 label=f'{thr:g} m — AP {ap:.4f}')
    ax.set_xlim(0, max(0.05, ax.get_xlim()[1]))
    ax.set_ylim(0, 1.02)
    ax.set_ylabel('precision', color=TEXT_MUTED, fontsize=8)
    legend(ax, loc='upper right')
    return finish(fig, ax,
                   'Global precision-recall (all tiles ranked together)',
                   'recall', tight=True)


def plot_quality_hist(rows, ykey):
    """How per-tile quality is distributed. The mass at exactly zero is the
    number this tab exists to explain: tiles where nothing was matched at
    all."""
    vals = [v for v in (r_metric(r, ykey) for r in rows) if v is not None]
    if len(vals) < 3:
        return None
    ym = RMETRICS[ykey]
    vals = np.asarray(vals, dtype=np.float64)
    fig, ax = stat_fig(height=3.6, width=6.2)
    if ym['better'] == 'zero':
        # Integer counts: one bin per value, edges offset by a half so each
        # bar is centred on its own integer. Shared bin edges would smear
        # "exactly right" together with "one too many".
        lo, hi = int(np.floor(vals.min())), int(np.ceil(vals.max()))
        bins = np.arange(lo - 0.5, hi + 1.5, 1.0) if hi - lo <= 60 else 30
        n_zero = int((vals == 0).sum())
        label = 'exactly right'
    else:
        bins = 30
        n_zero = int((vals <= 0).sum())
        label = 'at zero'
    ax.hist(vals, bins=bins, color=BOX_HUE, alpha=0.85)
    if ym['better'] == 'zero':
        ax.axvline(0, color=TEXT_MUTED, linewidth=0.9, alpha=0.9, zorder=4)
    ax.set_ylabel('tiles', color=TEXT_MUTED, fontsize=8)
    pct = 100.0 * n_zero / len(vals)
    return finish(fig, ax,
                   f'{ym["axis"]} across {len(vals):,} tiles — '
                   f'{n_zero:,} {label} ({pct:.0f}%)',
                   ym['axis'], tight=True)


RPLOTS = {}
for _yk in RQUALITY:
    for _xk, _xs in RX_VARS.items():
        if _xk == _yk:
            continue  # a metric against itself is a diagonal line
        _note = _xs['note']
        if {_xk, _yk} == {'n_gt', 'count_error'}:
            # count_error is DEFINED as n_pred_kept - n_gt, so plotting it
            # against n_gt has -n_gt on both axes and is negatively
            # correlated by construction: if the kept count were constant,
            # this would read exactly -1.00 while telling you nothing.
            _note = (
                'Careful: <b>partly circular</b>. count_error is defined as '
                '(predicted &minus; GT), so it contains &minus;GT and this '
                'correlation is negative by construction &mdash; a constant '
                'predicted count alone would produce &rho; = &minus;1. Read '
                'the predicted-vs-GT chart instead, where the diagonal is '
                'the honest reference.')
        RPLOTS[f'{_yk}__{_xk}'] = dict(
            kind='scatter', x=_xk, y=_yk, deep=_xs['deep'], note=_note)
RPLOTS['counts'] = dict(
    kind='counts', deep=False,
    note='Does the model know how many polylines a tile has? Points on the '
         'diagonal got the count exactly right; below it under-predicts, '
         'above it over-predicts. This is the non-circular version of count '
         'error against GT count -- neither axis is defined in terms of the '
         'other.')
RPLOTS['pr'] = dict(kind='pr', deep=False,
                     note='The cross-check on everything else here: this is '
                          'the real interleaved-ranking eval, so its AP '
                          'should match the training log.')
for _yk in RQUALITY:
    RPLOTS[f'hist__{_yk}'] = dict(kind='hist', y=_yk, deep=False,
                                   note='')
RPLOTS['bygroup'] = dict(kind='bygroup', deep=False,
                          note='Whether one town is carrying the whole '
                               'number. Train and test are different towns '
                               'here, so this is not rhetorical.')


def render_placeholder(spec, kind):
    """An explicit 'no data' panel, in place of a broken image."""
    with RENDER_LOCK:
        fig, ax = stat_fig(height=3.2, width=6.2)
        ax.axis('off')
        why = ('this needs the deep scan on the dataset-statistics tab'
               if spec.get('deep') else
               'no tile has a value for it in this run')
        ax.text(0.5, 0.5, f'Nothing to plot for “{kind}”.\n{why}.',
                 transform=ax.transAxes, ha='center', va='center',
                 color=TEXT_MUTED, fontsize=10, wrap=True)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=110, facecolor=BG)
        buf.seek(0)
        return buf


def render_res(*args, **kwargs):
    """Thread-safe wrapper -- same RENDER_LOCK as every other renderer."""
    with RENDER_LOCK:
        return _render_res(*args, **kwargs)


def _render_res(kind, rows, group_by, ykey):
    spec = RPLOTS[kind]
    if spec['kind'] == 'scatter':
        return plot_scatter(rows, spec['x'], spec['y'])
    if spec['kind'] == 'counts':
        return plot_counts(rows)
    if spec['kind'] == 'pr':
        return plot_pr_curve(rows)
    if spec['kind'] == 'hist':
        return plot_quality_hist(rows, spec['y'])
    if spec['kind'] == 'bygroup':
        return plot_quality_by_group(rows, ykey, group_by)
    return None


# ---- best / median / worst banding ---------------------------------------

BANDS = (('best', 'Best'), ('median', 'Median'), ('worst', 'Worst'))
BAND_FRAC = 0.2


def rank_rows(rows, sort_key):
    """Rows carrying a value for `sort_key`, best first."""
    m = RMETRICS[sort_key]
    scored = [(r, r_metric(r, sort_key)) for r in rows]
    scored = [(r, v) for r, v in scored if v is not None]
    if m['better'] == 'zero':
        # A signed error: best is nearest zero in either direction, worst is
        # furthest. Sorting by the raw value would instead rank "20 too few"
        # as the best tile on the page.
        scored.sort(key=lambda rv: abs(rv[1]))
    else:
        scored.sort(key=lambda rv: rv[1], reverse=m['better'] != 'low')
    return [r for r, _v in scored]


def band_rows(ranked, band):
    """The candidate pool for one band.

    Bands are the top/middle/bottom 20% of the ranking rather than single
    extremes, so that "Best" is a sample of good tiles and not the same one
    tile every time. With fewer than ~15 tiles the bands overlap; that is
    fine and better than showing nothing.
    """
    n = len(ranked)
    if not n:
        return []
    w = max(1, int(round(n * BAND_FRAC)))
    if band == 'best':
        return ranked[:w]
    if band == 'worst':
        return ranked[-w:]
    mid = n // 2
    lo = max(0, mid - w // 2)
    return ranked[lo:lo + w]


def pick_band(ranked, band, count, seed, pinned):
    """Which tiles to actually render for a band.

    Explicit picks win and keep their given order; the rest of the slots are
    filled by a seeded random sample of the band, so the page is stable
    across reloads (an unseeded sample would reshuffle every image request
    and make the gallery disagree with its own captions) but re-rollable
    with the Shuffle button.
    """
    pool = band_rows(ranked, band)
    by_uid = {r['uid']: r for r in pool}
    out = [by_uid[u] for u in pinned if u in by_uid]
    rest = [r for r in pool if r['uid'] not in {r2['uid'] for r2 in out}]
    need = max(0, count - len(out))
    if need and rest:
        # hashlib, not hash() -- str hashing is salted per process, so the
        # built-in would reshuffle the gallery on every viewer restart
        digest = hashlib.sha1(f'{seed}/{band}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:4], 'big'))
        idx = rng.permutation(len(rest))[:need]
        out.extend(rest[i] for i in sorted(idx))
    return out[:count]


# ---- the page ------------------------------------------------------------

RESULTS_PAGE = """<!doctype html>
<html><head>
<title>CARLA training results</title>
<style>{css}</style>
</head><body>
<h1>CARLA dataset viewer</h1>
{nav}
<div class="sub">{header}</div>

{setup}
</body></html>
"""

RESULTS_BODY = """
<form method="get">
  <input type="hidden" name="tab" value="results">
  <fieldset>
    <label class="top">Predictions</label>
    <select name="results">{result_opts}</select>
  </fieldset>
  <fieldset>
    <label class="top">Ground truth</label>
    <select name="gt">{gt_opts}</select>
  </fieldset>
  <fieldset>
    <label class="top">Rank tiles by</label>
    <select name="sort">{sort_opts}</select>
  </fieldset>
  <fieldset>
    <label class="top">Tiles per category</label>
    <input type="number" name="count" value="{count}" min="1" max="12" step="1">
  </fieldset>
  <fieldset>
    <label class="top">Group charts by</label>
    <select name="group_by">{group_opts}</select>
  </fieldset>
  <fieldset>
    <label class="top">Representation</label>
    <select name="mode">{mode_opts}</select>
  </fieldset>
  <fieldset>
    <label class="top">Prediction counts&colon; score &ge;</label>
    <input type="number" name="score_thresh" value="{score_thresh}"
           min="0" max="1" step="0.01">
  </fieldset>
  <fieldset>
    <label class="top">Predictions drawn</label>
    <select name="top_n">{topn_opts}</select>
  </fieldset>
  <input type="hidden" name="seed" value="{seed}">
  <button type="submit">Apply</button>
</form>

<h2>Run summary</h2>
{summary}

<h2>What predicts prediction quality?</h2>
<div class="note" style="margin-bottom:1em">One point per tile. The trend
  line is the median per decile of the x axis, not a fit &mdash; these
  relationships are monotonic at best, and a straight line through them
  would imply more than the data supports. Tiles the
  <a href="/?tab=stats">dataset-statistics</a> tab flags as suspect are drawn
  in red, so a cluster of them at the bottom of a chart means the data, not
  the model, is what is being measured there.</div>
<div class="gallery">
{gallery}
</div>

<h2>Tile browser &mdash; ranked by {sort_label}</h2>
<div class="note" style="margin-bottom:1em">Each category samples the
  top / middle / bottom {band_pct:.0f}% of the {n_ranked:,} ranked tiles.
  The selection is random but seeded, so it survives a reload; use
  <b>Shuffle</b> for a new draw, or pin specific tiles per category below.
  GT is solid red, predictions dashed yellow.</div>
{bands}
"""

BAND_BLOCK = """
<h2 style="margin-top:1.6em">{title} &mdash; {label}</h2>
<form method="get" style="margin-bottom:0.8em">
{hidden}
  <fieldset>
    <label class="top">Pin tiles (ctrl-click for several, empty = random)</label>
    <select name="pick_{band}" multiple size="6"
            style="min-width:26em">{options}</select>
  </fieldset>
  <button type="submit">Show pinned</button>
  <button type="submit" name="seed" value="{next_seed}" class="ghost">Shuffle</button>
</form>
<div class="gallery">
{gallery}
</div>
"""

NO_GT_NOTE = """
<div class="note" style="background:{panel};border:1px solid {border};
     border-radius:6px;padding:1.2em;max-width:60em">
  <b style="color:{text}">No evaluation ground truth found.</b><br><br>
  This tab scores predictions against either of two files, both of which
  give identical GT:<br><br>
  &bull; <code>data/carla_infos/carla_map_infos_&lt;split&gt;.pkl</code> &mdash;
  the converter's own output. <b style="color:{text}">Preferred</b>, because
  it exists as soon as the dataset has been converted, and is regenerated
  every time the converter runs.<br>
  &bull; <code>data/carla/carla_map_gt.json</code> &mdash; written by
  <code>_format_gt()</code> to the config's <code>map_ann_file</code>, but
  only once an eval or <code>tools/test.py</code> run has happened, so a
  freshly converted dataset does not have one yet.<br><br>
  Neither was found. Point at one explicitly with
  <code>--gt-json &lt;path&gt;</code> (it takes either format), or generate
  the pkl:<br>
  <code style="display:block;white-space:pre-wrap;margin-top:.4em">python3 tools/data_converter/carla_converter.py \\
  --data-root data/carla --out-dir data/carla_infos --split &lt;split&gt;</code>
  <br>
  <b style="color:{text}">Note:</b> if you do use the json, delete and
  regenerate it whenever the sample set or GT frame changes &mdash;
  <code>_format_gt()</code> skips regeneration if the file exists, so a
  stale one silently scores against the wrong GT. The pkl does not have
  that failure mode.
</div>
"""

NO_PRED_NOTE = """
<div class="note" style="background:{panel};border:1px solid {border};
     border-radius:6px;padding:1.2em;max-width:60em">
  <b style="color:{text}">{title}</b><br><br>
  {inner}
</div>
"""


def results_summary_html(rows):
    """Headline numbers, including the global AP -- the one directly
    comparable to the training log."""
    g = REval.get('global_ap') or {}
    n_gt = sum(r['n_gt'] for r in rows)
    n_pred = sum(r['n_pred'] for r in rows)
    n_tp = sum(r['n_tp'] for r in rows)
    aps = [r['ap'] for r in rows if r['ap'] is not None]
    zero = sum(1 for r in rows if r['ap'] == 0)

    cells = [
        ('tiles scored', f'{len(rows):,}'),
        ('GT polylines', f'{n_gt:,}'),
        ('predictions', f'{n_pred:,}'),
        ('correct @ 1.0 m', f'{n_tp:,}'),
        ('recall @ 1.0 m', f'{n_tp / n_gt:.3f}' if n_gt else '&mdash;'),
        ('global mAP', f'{g.get("mean"):.4f}' if g.get('mean') is not None
         else '&mdash;'),
    ]
    for thr in CHAMFER_THRESHOLDS:
        v = g.get(thr)
        cells.append((f'global AP @ {thr:g} m',
                       f'{v:.4f}' if v is not None else '&mdash;'))
    cells.append(('median per-tile AP',
                   f'{np.median(aps):.4f}' if aps else '&mdash;'))
    cells.append(('tiles at AP 0', f'{zero:,}'))
    errs = np.array([r['count_error'] for r in rows], dtype=np.float64)
    exact = int((errs == 0).sum())
    cells.append(('count exactly right',
                   f'{exact:,} ({100.0 * exact / len(rows):.0f}%)'))
    cells.append(('median count error', f'{np.median(errs):+.0f}'))

    head = ''.join(f'<th>{k}</th>' for k, _v in cells)
    body = ''.join(f'<td>{v}</td>' for _k, v in cells)
    note = ('<div class="note">Global AP pools every tile\'s detections and '
            'ranks them together &mdash; the real eval, directly comparable '
            'to the training log\'s <code>CarlaMap_chamfer/*</code>. The '
            'per-tile AP everything below sorts by is a <i>local</i> score '
            '(one tile\'s own ranking against its own GT); the two answer '
            'different questions and will not be equal.</div>')
    miss = []
    if REval.get('missing_gt'):
        miss.append(f'{REval["missing_gt"]:,} predicted tiles had no GT entry')
    if REval.get('missing_tile'):
        miss.append(f'{REval["missing_tile"]:,} had no matching tile under '
                     f'--data-root')
    if REval.get('ambiguous'):
        miss.append(f'{REval["ambiguous"]:,} tile names exist in more than '
                     f'one loaded dataset and were resolved to the first '
                     f'(restrict with --split to disambiguate)')
    if miss:
        note += (f'<div class="note" style="color:{CRITICAL}">Skipped: '
                 + '; '.join(miss) + '.</div>')
    note += count_health(rows)
    return (f'<table class="stats"><thead><tr>{head}</tr></thead>'
            f'<tbody><tr>{body}</tr></tbody></table>{note}')


def count_health(rows):
    """Warn when the count metrics cannot mean anything at this threshold.

    The detection head emits a fixed `num_vec` predictions for EVERY tile,
    so the raw count carries no information and count_error is only as
    meaningful as the score threshold that trims it. Two ways that fails,
    both silent without this check:

      * the threshold keeps all (or none) of the predictions on nearly
        every tile, so count_error is just a constant minus n_gt;
      * the scores barely differ across the whole run, in which case NO
        threshold separates anything. That is what an undertrained
        checkpoint looks like -- confidences collapse to almost one value
        -- and it is a fact about the run, not about the tiles.

    Reported rather than worked around: silently substituting some other
    definition of "how many polylines" would make the number look
    informative when it is not.
    """
    if not rows:
        return ''
    n_raw = np.array([r['n_pred'] for r in rows], dtype=np.float64)
    n_kept = np.array([r['n_pred_kept'] for r in rows], dtype=np.float64)
    scores = np.concatenate([r['scores'] for r in rows
                              if len(r['scores'])]) if rows else np.zeros(0)
    all_kept = float((n_kept == n_raw).mean())
    none_kept = float((n_kept == 0).mean())
    msgs = []
    if all_kept > 0.95:
        msgs.append(
            f'the score threshold keeps <b>every</b> prediction on '
            f'{all_kept * 100:.0f}% of tiles, so the count is the head\'s '
            f'fixed num_vec and the count error is just that constant minus '
            f'the GT count')
    elif none_kept > 0.95:
        msgs.append(
            f'the score threshold discards <b>every</b> prediction on '
            f'{none_kept * 100:.0f}% of tiles, so the count error is just '
            f'&minus;(GT count)')
    if len(scores) > 1:
        spread = float(scores.max() - scores.min())
        if spread < 0.05:
            msgs.append(
                f'every confidence in this run lies between '
                f'{scores.min():.4f} and {scores.max():.4f} (a spread of '
                f'{spread:.4f}), so <b>no</b> threshold separates confident '
                f'predictions from unconfident ones &mdash; a sign of an '
                f'undertrained checkpoint rather than of the data')
    if not msgs:
        return ''
    return (f'<div class="note" style="color:{CRITICAL};max-width:60em">'
            f'<b>Count metrics are not informative here:</b> '
            + '; and '.join(msgs) + '. The AP, TP and recall columns are '
            'unaffected &mdash; they ignore the threshold entirely.</div>')


def results_page():
    work_dir = STATE.get('work_dir')
    result_files = discover_results(work_dir) if work_dir else {}
    gt_files = discover_gt(work_dir)
    if STATE.get('gt_json'):
        gt_files = [STATE['gt_json']] + [g for g in gt_files
                                          if g != STATE['gt_json']]

    def shell(setup, header):
        page = RESULTS_PAGE.format(css=CSS, nav=nav_html('results'),
                                    header=header, setup=setup)
        resp = make_response(page)
        resp.headers['Cache-Control'] = ('no-store, no-cache, '
                                          'must-revalidate, max-age=0')
        return resp

    if not result_files:
        if not work_dir:
            inner = ('Start the viewer with <code>--work-dir &lt;dir&gt;</code> '
                     'to point it at a training run.')
        else:
            # Same situation the browse tab explains, but that version is
            # wrapped in a <fieldset> for its filter form, so the shared
            # part is the how-to rather than the markup.
            ckpts = sorted(glob_pth(work_dir))
            cfgs = sorted(f for f in os.listdir(work_dir)
                           if f.endswith('.py')) if osp.isdir(work_dir) else []
            if ckpts:
                ckpt = next((c for c in ckpts
                              if osp.basename(c) == 'latest.pth'), None) \
                    or ckpts[-1]
                cfg = (osp.join(work_dir, cfgs[0]) if cfgs else
                       'plugin/configs/'
                       'mapdiffusion_carla_lidar_25m.py')
                howto = HOWTO_WITH_CKPT.format(config=cfg, ckpt=ckpt,
                                                work_dir=work_dir)
            else:
                howto = HOWTO_NO_CKPT
            inner = (f'Nothing matching <code>*result*.json</code> under '
                     f'<code>{html.escape(work_dir)}</code>. Training writes '
                     f'only checkpoints and logs there; its eval hook writes '
                     f'predictions to <code>val/&lt;work_dir&gt;/&lt;timestamp'
                     f'&gt;/</code> relative to the CWD training ran from, '
                     f'which for a container run is usually not bind-mounted, '
                     f'so they were discarded.{howto}')
        return shell(NO_PRED_NOTE.format(panel=BG_PANEL, border=BORDER,
                                          text=TEXT, inner=inner,
                                          title='No predictions to score.'),
                      'no predictions found')
    if not gt_files:
        return shell(NO_GT_NOTE.format(panel=BG_PANEL, border=BORDER,
                                        text=TEXT), 'no eval GT found')

    results = request.args.get('results') or next(iter(result_files))
    if results not in result_files:
        results = next(iter(result_files))
    gt = request.args.get('gt')
    if gt not in gt_files:
        gt = best_gt_for(result_files[results], gt_files)
    sort_key = request.args.get('sort', 'ap')
    if sort_key not in RMETRICS:
        sort_key = 'ap'
    group_by = request.args.get('group_by', 'town')
    if group_by not in ('town', 'split'):
        group_by = 'town'
    mode = request.args.get('mode', 'points')
    count = max(1, min(12, int(request.args.get('count', 3) or 3)))
    seed = request.args.get('seed', '0')
    score_thresh = request.args.get('score_thresh', '0.1')
    # 'gt' -- draw as many predictions as the tile has GT lines -- is the
    # default because the model emits a fixed num_vec every time (50 here)
    # and drawing all of them buries the GT under a haystack of
    # near-zero-confidence guesses. See the top_n note in _render_tile.
    top_n = request.args.get('top_n', 'gt')
    if top_n not in ('gt', 'all') and _safe_int(top_n) is None:
        top_n = 'gt'

    rows = evaluate_run(result_files[results], gt,
                         _safe_float(score_thresh, 0.0))
    if not rows:
        # Almost always a split mismatch (test predictions against the train
        # pkl), not a missing file -- so say that, and say which candidate
        # would have worked, rather than showing the how-to-generate-GT note.
        best = best_gt_for(result_files[results], gt_files)
        n_over = len(set(load_results(result_files[results]))
                     & set(load_gt(best))) if best else 0
        alt = (f'<code>{html.escape(osp.relpath(best))}</code> covers '
               f'{n_over:,} of them &mdash; select it above.'
               if best and best != gt and n_over else
               'No other candidate covers them either, so the predictions '
               'and this dataset are probably from different exports.')
        return shell(
            NO_PRED_NOTE.format(
                panel=BG_PANEL, border=BORDER, text=TEXT,
                title='This ground truth does not cover these predictions.',
                inner=(f'<code>{html.escape(osp.relpath(gt))}</code> has no '
                       f'ground truth for any of the '
                       f'{len(load_results(result_files[results])):,} tiles '
                       f'in these predictions. Splits are disjoint sets of '
                       f'tiles, so this is normally the wrong split\'s '
                       f'annotation file. {alt}')),
            'predictions and GT share no tiles')

    have_deep = any(r.get(k) is not None for r in rows for k in DEEP_FIELDS)
    cachebust = f'{time.time():.6f}'
    figs = []
    for kind, spec in RPLOTS.items():
        if spec['deep'] and not have_deep:
            continue
        params = [('plot', kind), ('results', results), ('gt', gt),
                   ('group_by', group_by), ('sort', sort_key),
                   ('score_thresh', score_thresh), ('v', cachebust)]
        # urlencode + html.escape is REQUIRED, not cosmetic -- see the long
        # comment in index() about "&gt" being resolved as an entity
        q = html.escape('?' + urlencode(params), quote=True)
        cap = spec['note']
        figs.append(
            f'<figure style="max-width:560px">'
            f'<a href="/res.png{q}" target="_blank">'
            f'<img src="/res.png{q}" style="width:540px"></a>'
            f'<figcaption style="text-align:left;word-break:normal">'
            f'{cap}</figcaption></figure>')
    if not have_deep:
        figs.append(
            f'<figure style="max-width:560px;width:540px"><figcaption '
            f'style="text-align:left;word-break:normal">Quality against '
            f'<b>effective point count</b> (what survives grid sampling, as '
            f'opposed to what the manifest claims) and against the tile\'s '
            f'<b>median</b> and <b>mean z</b> appears here once a '
            f'<a href="/?tab=stats">deep scan</a> has run &mdash; all three '
            f'need every tile\'s .npz read.</figcaption></figure>')

    ranked = rank_rows(rows, sort_key)
    hidden = ''.join(
        f'<input type="hidden" name="{k}" value="{html.escape(str(v), True)}">'
        for k, v in [('tab', 'results'), ('results', results), ('gt', gt),
                      ('sort', sort_key), ('count', count), ('mode', mode),
                      ('group_by', group_by), ('top_n', top_n),
                      ('score_thresh', score_thresh), ('seed', seed)])
    bands = []
    for band, title in BANDS:
        pinned = request.args.getlist(f'pick_{band}')
        chosen = pick_band(ranked, band, count, seed, pinned)
        pool = band_rows(ranked, band)
        opts = []
        for r in pool[:200]:
            v = r_metric(r, sort_key)
            sel = ' selected' if r['uid'] in pinned else ''
            opts.append(
                f'<option value="{html.escape(r["uid"], True)}"{sel}>'
                f'{html.escape(r["name"])} — '
                f'{RMETRICS[sort_key]["fmt"].format(v)}</option>')
        if len(pool) > 200:
            opts.append('<option disabled>… '
                         f'{len(pool) - 200:,} more (band is larger than this '
                         'list)</option>')
        bands.append(BAND_BLOCK.format(
            title=title, band=band, hidden=hidden.replace(
                f'name="seed" value="{seed}"',
                f'name="seed" value="{seed}"'),
            label=band_label(band, pool, sort_key),
            options='\n'.join(opts) or '<option disabled>no tiles</option>',
            next_seed=str(int(seed) + 1 if seed.isdigit() else 1),
            gallery=tile_gallery(chosen, mode, results, score_thresh,
                                  sort_key, cachebust, top_n)))

    body = RESULTS_BODY.format(
        result_opts=_opts(list(result_files), results),
        gt_opts=_opts(gt_files, gt, [osp.relpath(g) for g in gt_files]),
        sort_opts=_opts(list(RMETRICS), sort_key,
                         [RMETRICS[k]['label'] for k in RMETRICS]),
        group_opts=_opts(['town', 'split'], group_by,
                          ['town', 'split (train vs test)']),
        mode_opts=_opts(['points', 'rgb', 'label', 'density', 'intensity'],
                         mode, ['top-down (flat colour)', 'true RGB colour',
                                'lane label', 'density heat map',
                                'intensity']),
        count=count, seed=seed, score_thresh=score_thresh,
        topn_opts=_opts(['gt', '1', '3', '5', '10', '20', 'all'], top_n,
                         ['as many as GT lines', 'top 1 by score', 'top 3',
                          'top 5', 'top 10', 'top 20', 'all predictions']),
        summary=results_summary_html(rows),
        gallery='\n'.join(figs),
        sort_label=RMETRICS[sort_key]['label'],
        band_pct=100 * BAND_FRAC, n_ranked=len(ranked),
        bands='\n'.join(bands))

    header = (f'{html.escape(results)} scored against '
              f'<code>{html.escape(osp.relpath(gt))}</code> &mdash; '
              f'{len(rows):,} tiles in {REval.get("elapsed", 0):.1f}s')
    return shell(body, header)


def band_label(band, pool, sort_key):
    if not pool:
        return 'no tiles'
    fmt = RMETRICS[sort_key]['fmt']
    vals = [r_metric(r, sort_key) for r in pool]
    vals = [v for v in vals if v is not None]
    if not vals:
        return f'{len(pool):,} tiles'
    return (f'{len(pool):,} tiles, {RMETRICS[sort_key]["label"]} '
            f'{fmt.format(min(vals))} to {fmt.format(max(vals))}')


def tile_gallery(chosen, mode, results, score_thresh, sort_key, cachebust,
                  top_n='all'):
    """Renders through the existing /tile.png route, so these tiles look
    exactly like the browse tab's and the GT/pred overlay logic lives in one
    place."""
    figs = []
    for r in chosen:
        params = [('name', r['name']), ('ds', r['ds']), ('mode', mode),
                   ('polylines', '1'), ('results', results),
                   ('score_thresh', score_thresh), ('point_size', '1.5'),
                   ('v', cachebust)]
        # 'gt' resolves per tile, which is the point of it
        n = r['n_gt'] if top_n == 'gt' else _safe_int(top_n)
        if n:
            params.append(('top_n', str(n)))
        q = html.escape('?' + urlencode(params), quote=True)
        bits = []
        for k in (sort_key, 'ap', 'n_tp', 'n_gt', 'n_pred'):
            if k in bits:
                continue
            v = r_metric(r, k)
            if v is None:
                continue
            bits.append(f'{RMETRICS[k]["label"].split(" (")[0]} '
                         f'<b>{RMETRICS[k]["fmt"].format(v)}</b>')
        flags = ' '.join(f'<span class="chip" style="color:{CHIP_COLORS[k]}">'
                          f'{k}</span>' for k, _t in tile_flags(r))
        browse = html.escape(
            '/?' + urlencode([('town', r['group']), ('start', r['gidx']),
                               ('count', 1), ('mode', mode),
                               ('polylines', '1')]), quote=True)
        figs.append(
            f'<figure><a href="/tile.png{q}" target="_blank">'
            f'<img src="/tile.png{q}"></a>'
            f'<figcaption style="text-align:left">'
            f'<a href="{browse}">{html.escape(r["name"])}</a><br>'
            + ' &middot; '.join(bits[:4]) + (f'<br>{flags}' if flags else '')
            + '</figcaption></figure>')
    return '\n'.join(figs) or (f'<p style="color:{TEXT_MUTED}">'
                                f'no tiles in this band</p>')


@app.route('/res.png')
def res_png():
    kind = request.args.get('plot')
    if kind not in RPLOTS:
        abort(404)
    work_dir = STATE.get('work_dir')
    result_files = discover_results(work_dir) if work_dir else {}
    results = request.args.get('results')
    gt = request.args.get('gt')
    # both reach us from a query string, so neither is trusted: results must
    # be one of the discovered labels and gt one of the discovered paths
    if results not in result_files or gt not in discover_gt(work_dir) \
            + ([STATE['gt_json']] if STATE.get('gt_json') else []):
        abort(404)
    group_by = request.args.get('group_by', 'town')
    if group_by not in ('town', 'split'):
        group_by = 'town'
    sort_key = request.args.get('sort', 'ap')
    if sort_key not in RMETRICS:
        sort_key = 'ap'
    rows = evaluate_run(result_files[results], gt,
                         _safe_float(request.args.get('score_thresh'), 0.0))
    buf = render_res(kind, rows, group_by, sort_key)
    if buf is None:
        # A chart with nothing to draw (a deep-tier field not scanned yet,
        # or a metric undefined for every tile) must not 404: the page has
        # already emitted the <img>, so a 404 shows a broken-image icon and
        # looks like a bug rather than an empty result.
        buf = render_placeholder(RPLOTS[kind], kind)
    resp = send_file(buf, mimetype='image/png')
    resp.headers['Cache-Control'] = ('no-store, no-cache, must-revalidate, '
                                      'max-age=0')
    return resp


@app.route('/results.csv')
def results_csv():
    if REval.get('rows') is None:
        abort(404)
    rows = REval['rows']
    buf = io.StringIO()
    w = csv.writer(buf)
    keys = ['ap', 'ap_0.5', 'ap_1.0', 'ap_1.5', 'n_tp', 'recall',
            'precision', 'chamfer', 'n_gt', 'n_pred', 'n_pred_kept',
            'count_error']
    w.writerow(['uid', 'name', 'dataset', 'split', 'town', 'n_points',
                 'points_per_m2'] + list(DEEP_FIELDS) + ['z_skew'] + keys
                + ['flags'])
    for r in rows:
        # median - mean: how far the road sits below the centre of mass.
        # The one z quantity the model's centroid-relative input encodes.
        zmed, zmean = r.get('z_median'), r.get('z_mean')
        skew = ('' if zmed is None or zmean is None
                else f'{zmed - zmean:.4f}')
        w.writerow([r['uid'], r['name'], r['ds'], r['split'], r['town'],
                     r['n_points'], f"{r['density']:.3f}"]
                    + [('' if r.get(k) is None else f'{r[k]:.6g}')
                       for k in DEEP_FIELDS]
                    + [skew]
                    + [('' if r_metric(r, k) is None
                        else f'{r_metric(r, k):.6g}') for k in keys]
                    + [' '.join(k for k, _v in tile_flags(r))])
    resp = make_response(buf.getvalue())
    resp.headers['Content-Type'] = 'text/csv; charset=utf-8'
    resp.headers['Content-Disposition'] = 'attachment; filename=results.csv'
    return resp


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--split', default=None,
                    help='restrict to a single split or dataset directory; '
                         'default is to load every manifest.json / '
                         'grid_manifest.json found at or below --data-root '
                         '(so towns from train and test both appear in the '
                         'picker)')
    p.add_argument('--work-dir', default=None,
                    help='training work_dir; enables overlaying predicted '
                         'polylines from any *result*.json found under it '
                         '(or under val/<work-dir>, where the training eval '
                         'hook writes them)')
    p.add_argument('--port', type=int, default=5001)
    p.add_argument('--max-points', type=int, default=150000,
                    help='cap on points drawn in scatter modes (density mode '
                         'always uses every point so the histogram stays exact)')
    # --- statistics tab ---
    p.add_argument('--stats-cache', default=None,
                    help='where the deep scan caches its per-tile results '
                         '(default ~/.cache/mapdiffusion_dataset_viewer). Never '
                         'written inside the dataset dir, which is read-only '
                         'in the container workflow')
    p.add_argument('--scan-grid', type=float, nargs='+', default=(0.1, 0.1, 0.4),
                    metavar='M',
                    help='cell size for the effective-point count, in metres, '
                         'per axis. Defaults to mapdiffusion_carla_lidar_25m '
                         '.py\'s own lidar_voxel_size, which is what '
                         'GridSamplePoints uses -- so "effective points" '
                         'means "points the model actually sees". Changing '
                         'it invalidates the cache')
    p.add_argument('--scan-workers', type=int, default=8,
                    help='threads for the deep scan; npz decompression is '
                         'zlib, which releases the GIL, so this is a real '
                         'speedup on the 12 GB read')
    p.add_argument('--scan-stride', type=int, default=1,
                    help='scan every Nth tile only -- a fast approximate '
                         'pass over a large split')
    p.add_argument('--pc-range-z', type=float, nargs=2, default=(-72.0, 96.0),
                    metavar=('ZMIN', 'ZMAX'),
                    help='the model\'s lidar_point_cloud_range z bounds, drawn '
                         'on the z-extent chart and used for the '
                         'z-out-of-range flag. Default matches '
                         'mapdiffusion_carla_lidar_25m.py\'s current, widened '
                         'range -- NOT the old [-8, 15], which predates the '
                         'town03 overpass fix')
    # --- training-results tab ---
    p.add_argument('--gt-json', '--gt', dest='gt_json', default=None,
                    metavar='PATH',
                    help='the eval ground truth to score predictions '
                         'against. Takes EITHER a converter annotation pkl '
                         '(carla_map_infos_<split>.pkl) or a _format_gt() '
                         'json (carla_map_gt.json) -- they yield identical '
                         'GT. Default: search data/carla_infos, data/carla and --work-dir, '
                         'preferring a pkl, since a freshly converted '
                         'dataset has no json yet and an existing json is '
                         'never regenerated')
    p.add_argument('--num-pts-per-vec', type=int, default=20,
                    help='the model\'s fixed polyline resampling length, '
                         'marked on the vertices-per-polyline chart')
    return p.parse_args()


def main():
    args = parse_args()
    STATE['data_root'] = args.data_root
    STATE['max_points'] = args.max_points
    STATE['work_dir'] = args.work_dir
    STATE['results_cache'] = {}
    STATE['gt_cache'] = {}
    STATE['gt_json'] = osp.abspath(args.gt_json) if args.gt_json else None
    if STATE['gt_json'] and not osp.isfile(STATE['gt_json']):
        raise SystemExit(f'--gt-json is not a file: {args.gt_json}')
    STATE['deep'] = {}
    # one value broadcasts to all three axes, like GridSamplePoints' own
    # scalar grid_size handling
    grid = list(args.scan_grid)
    STATE['scan_grid'] = tuple(grid * 3)[:3] if len(grid) == 1 else tuple(grid)
    if len(STATE['scan_grid']) != 3:
        raise SystemExit('--scan-grid takes 1 or 3 values, got '
                          f'{len(grid)}: {grid}')
    STATE['scan_workers'] = max(1, args.scan_workers)
    STATE['scan_stride'] = max(1, args.scan_stride)
    STATE['pc_range_z'] = tuple(args.pc_range_z)
    STATE['num_pts_per_vec'] = args.num_pts_per_vec
    STATE['cache_dir'] = args.stats_cache or osp.join(
        osp.expanduser('~'), '.cache', 'mapdiffusion_dataset_viewer')
    if args.work_dir is not None and not osp.isdir(args.work_dir):
        hint = ''
        if osp.isfile(args.work_dir):
            hint = (f'\n  That is a file, not a directory -- did you mean '
                    f'its parent?\n    --work-dir {osp.dirname(args.work_dir)}')
        elif osp.isdir(osp.dirname(args.work_dir) or '.'):
            siblings = [d for d in os.listdir(osp.dirname(args.work_dir) or '.')
                        if osp.isdir(osp.join(osp.dirname(args.work_dir) or '.', d))]
            if siblings:
                hint = ('\n  Directories that do exist there: '
                        + ', '.join(sorted(siblings)[:8]))
        raise SystemExit(f'--work-dir is not a directory: {args.work_dir}{hint}')
    if not osp.isdir(args.data_root):
        raise SystemExit(f'no such data root: {args.data_root}')

    datasets = discover_datasets(args.data_root, only=args.split)
    if not datasets:
        extra = f' matching --split {args.split}' if args.split else ''
        raise SystemExit(
            f'no {" or ".join(MANIFEST_NAMES)} found at or below '
            f'{args.data_root}{extra}')
    STATE['datasets'] = datasets
    tiles, groups = build_index(datasets)
    if not tiles:
        raise SystemExit(f'manifests found but they list no tiles: '
                          f'{", ".join(datasets)}')
    STATE['tiles'] = tiles
    STATE['tiles_by_uid'] = {t['_uid']: t for t in tiles}
    STATE['groups'] = groups
    # These lookups are dataset-wide; merge across manifests so a viewer
    # started on a root holding several exports decodes them all.
    STATE['lane_types'] = merged_lookup(datasets, 'lane_type_lookup')
    STATE['class_lookup'] = merged_lookup(datasets, 'class_lookup')
    # results json identifies a predicted class by name only, so the
    # reverse map is what lets predictions share the GT class palette
    STATE['class_ids'] = {v: int(k) for k, v in STATE['class_lookup'].items()}
    STATE['class_choices'] = class_choices(datasets)
    STATE['class_summary'] = class_summary(datasets)

    # Flask runs with debug=False (the reloader is unreliable when the
    # process is backgrounded), so edits to this file do NOT take effect
    # until it's restarted. Print the file's mtime so a stale process is
    # obvious rather than looking like "my changes did nothing".
    mtime = datetime.fromtimestamp(osp.getmtime(osp.abspath(__file__)))
    for key, man in datasets.items():
        classes = ', '.join(man['class_lookup'].values()) or 'unclassified'
        print(f"  {key}: {man['n_tiles']:,} tiles  "
              f"({', '.join(man['towns'])})  split={man['split']}  "
              f"classes: {classes}")
    print(f"Loaded {len(tiles):,} tiles across {len(groups)} town(s) "
          f"from {len(datasets)} dataset(s)")
    n_cached = load_deep_cache()
    print(f"Deep statistics: {n_cached:,}/{len(tiles):,} tiles cached "
          f"(grid={grid_str()} m) in {STATE['cache_dir']}")
    print(f'Code version: {osp.basename(__file__)} last modified '
          f'{mtime:%Y-%m-%d %H:%M:%S}')
    print(f'Viewer on http://127.0.0.1:{args.port}')
    app.run(host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
