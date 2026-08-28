"""Convert the CARLA road-polyline tile dataset into MapDiffusion's
map-annotation pkl format.

Ported from the sibling MapTRv2 tree's
``tools/maptrv2/custom_carla_map_converter.py`` (commit 5aa3b26), retargeted
at ``BaseMapDataset``'s sample schema and at MapDiffusion's tile-centred ROI
convention.

Unlike the nuScenes converter, CARLA tiles are static square patches (not
per-timestamp driving-log frames), so there is no ego pose/SE3 transform and
no need to clip polylines against a moving patch. Each tile becomes its own
one-frame "scene".

Nothing here assumes a particular tile size. The exporter has produced at
least 25m tiles (``tile_radius`` 12.5) and 60m ones (30.0), and both layouts
it writes are accepted::

    <data_root>/<split>/manifest.json            # split export
    <data_root>/<split>/blocks/<tile>.npz
    <data_root>/<split>/reference_lines/<tile>_reference_lines.json

    <data_root>/grid_manifest.json               # single-town grid export
    <data_root>/blocks/<tile>.npz                # (no <split> level)
    <data_root>/reference_lines/<tile>_...json

Where a directory holds both files, ``manifest.json`` wins and
``grid_manifest.json`` only backfills keys it lacks -- the same rule
``utils/dataset_viewer.py`` uses. Manifest-level *counts* are not trustworthy
(one file reports ``n_tiles: 4103`` while listing 30), so everything here is
derived from the tile entries.

--- The frame (this is the one real divergence from MapTRv2/GeMap) ---

Both sibling converters emit GT in the block's ``offset`` frame, i.e.
``pts - block['offset']``, matching ``features[:, 0:3]`` which is what their
point loader reads. That keeps polylines and points mutually aligned -- a
median 0.038m from real driving-surface returns, vs 0.388m if only the
polylines are shifted -- and nothing below changes that.

What it does *not* do is centre the tile. ``offset`` is not the tile centre,
so in that frame a tile sits in a box centred on ``tile_center - offset``:
1.6m off on average for the 25m export (max 12.1m), and up to ~17m on the
60m one, growing with tile size. MapTRv2 and GeMap absorb the overhang
downstream (``LiDARInstanceLines`` clamps xy to +/-tile_radius, and their
coder has a wider ``post_center_range``). MapDiffusion has no such clamp:
``VectorizeMap.normalize_line`` maps metric xy to (0, 1) about
``origin = -roi_size/2`` and the head's reference points are sigmoid-bounded,
so anything outside the ROI is both unlearnable and an out-of-range
regression target. Measured over 80 random train tiles (6262 GT vertices),
13.5% of them fall outside +/-12.5m in the offset frame (max 18.6m).

So this converter shifts by ``tile_center`` instead -- applied to *both*
modalities, which is what keeps the sibling repos' alignment result intact:

    GT      : pts_world      - tile_center
    points  : features[:,:3] - (tile_center - offset)     # == points_world - tile_center

i.e. a single rigid translation of the whole tile, not a change in their
relative geometry. The point-side half of it is applied by
``LoadCarlaPointsFromFile`` using the ``tile_shift`` recorded on each sample.
In this frame every GT vertex lands inside +/-tile_radius by construction.

--- Zero-voxel tiles ---

Each tile's LiDAR block is scanned for points that would actually survive
the training pipeline (``z <= --z-max``, then inside
``--lidar-point-cloud-range``, both in the recentred frame). A tile with
fewer than ``--min-lidar-points`` such points voxelizes to zero voxels and
takes the whole run down inside ``extract_lidar_feat``, so it is dropped here
and listed in a sidecar report. Every kept sample records its own count, so
``CarlaDataset`` can re-check an already-generated pkl cheaply.

Usage::

    python tools/data_converter/carla_converter.py \\
        --data-root ./data/carla --out-dir ./data/carla_infos --split train
"""

import argparse
import json
import os

import mmcv
import numpy as np

# z half of the range the training config's LiDAR branch uses
# (`lidar_point_cloud_range` in plugin/configs/mapdiffusion_carla_lidar.py)
# and its `z_max`. The xy half is NOT a constant: it follows the tile size
# read from the manifest (see default_pc_range), because a range narrower
# than the tile silently crops it and a wider one just wastes BEV cells.
# The whole range is recorded into the pkl so the dataset can warn if config
# and pkl ever drift apart.
DEFAULT_Z_RANGE = (-72.0, 96.0)
DEFAULT_Z_MAX = 96.0
# Only used when a manifest carries no tile geometry at all; matches the
# 25m export this converter was written against.
FALLBACK_TILE_RADIUS = 12.5
# Slack allowed before a polyline is reported as leaving its own tile. The
# exporter samples arcs at a fixed arc_gap, so a vertex can land slightly
# past the boundary without anything being wrong.
BOUNDS_MARGIN = 1.0

MANIFEST_NAMES = ('manifest.json', 'grid_manifest.json')


def parse_args():
    parser = argparse.ArgumentParser(
        description='CARLA map data converter arg parser')
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='root of the CARLA tile dataset (contains <split>/manifest.json, '
        'or a manifest.json/grid_manifest.json directly)')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./data/carla_infos',
        help='output directory for the generated pkl. Kept apart from '
        '--data-root by default so the tile export can stay read-only')
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help='split subdirectory under --data-root to convert; also used as '
        'the label in the output filename. If no such subdirectory exists, '
        '--data-root itself is read and this is only the output label')
    parser.add_argument(
        '--classes',
        type=str,
        nargs='+',
        default=None,
        metavar='CLASS',
        help='optional subset of polyline classes to keep as divider '
        'instances, as class_lookup ids or names (e.g. `0` or '
        '`driving_centerline`). Default: keep every polyline. Only exports '
        'whose polylines carry a class can be filtered')
    parser.add_argument(
        '--lidar-point-cloud-range',
        type=float,
        nargs=6,
        default=None,
        metavar=('X_MIN', 'Y_MIN', 'Z_MIN', 'X_MAX', 'Y_MAX', 'Z_MAX'),
        help='range the LiDAR voxelizer will use; points outside it are '
        'dropped before voxelization, so a tile with none inside produces '
        'zero voxels. Default: +/-tile_radius in xy (read from the manifest) '
        f'and {DEFAULT_Z_RANGE[0]}..{DEFAULT_Z_RANGE[1]} in z')
    parser.add_argument(
        '--z-max',
        type=float,
        default=DEFAULT_Z_MAX,
        help="LoadCarlaPointsFromFile's early z filter (default: matches "
        'the training config)')
    parser.add_argument(
        '--min-lidar-points',
        type=int,
        default=1,
        help='drop tiles with fewer than this many in-range LiDAR points; '
        '1 drops only genuinely zero-voxel tiles (default: 1)')
    parser.add_argument(
        '--no-lidar-check',
        action='store_true',
        help='skip the per-tile LiDAR scan entirely (faster, but no tile is '
        'dropped and no point counts are recorded)')
    return parser.parse_args()


def load_manifest(data_root, split):
    """Locate and load the manifest describing one set of tiles.

    Accepts either export layout: ``<data_root>/<split>/`` (the split export)
    or ``<data_root>`` itself (a grid export, which has no split level).
    Returns ``(tile_dir, manifest)``; `tile_dir` is where `blocks/` and
    `reference_lines/` live.
    """
    for candidate in ([os.path.join(data_root, split)] if split else
                      []) + [data_root]:
        present = [
            n for n in MANIFEST_NAMES
            if os.path.isfile(os.path.join(candidate, n))
        ]
        if not present:
            continue
        manifest = {}
        # reversed so the preferred name (first in MANIFEST_NAMES) is applied
        # last and its keys win, while the other only backfills. A null value
        # never overwrites a real one -- grid_manifest.json spells absent
        # options as `null` rather than omitting them.
        for name in reversed(present):
            with open(os.path.join(candidate, name), encoding='utf-8') as f:
                loaded = json.load(f)
            manifest.update({
                k: v
                for k, v in loaded.items()
                if v is not None or k not in manifest
            })
        if not manifest.get('tiles'):
            raise ValueError(f'{candidate}: manifest lists no tiles')
        return candidate, manifest
    raise FileNotFoundError(
        f'no {" or ".join(MANIFEST_NAMES)} under {data_root!r}'
        f'{f" or {os.path.join(data_root, split)!r}" if split else ""}')


def manifest_tile_radius(manifest):
    """Half a tile's side, in metres, from whichever key the export used.

    `tile_radius` (25m export) and `tile_side` (some grid manifests) are
    dataset-level; failing both, the widest per-tile `bounds` gives the same
    answer for a uniform grid and an upper bound otherwise, which is the safe
    direction for sizing a point-cloud range. Returns None only when an
    export states its geometry nowhere.
    """
    if manifest.get('tile_radius') is not None:
        return float(manifest['tile_radius'])
    if manifest.get('tile_side') is not None:
        return float(manifest['tile_side']) / 2.0
    radii = []
    for tile in manifest.get('tiles', []):
        bounds = tile.get('bounds')
        if bounds is None:
            continue
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.size == 4:
            radii.append(np.max(bounds[2:] - bounds[:2]) / 2.0)
        elif bounds.size == 6:
            radii.append(np.max(bounds[3:5] - bounds[:2]) / 2.0)
    return float(max(radii)) if radii else None


def tile_footprint(tile, ref, tile_radius):
    """World-frame xy footprint of one tile, as ``(lo, hi)`` arrays.

    Prefers the tile's own `bounds` (exact, and the only source that would
    describe a non-square tile), then its centre plus the export's tile
    radius. The reference-lines json repeats all three keys, so it backs up a
    manifest whose tile entries are sparse. Returns None when nothing
    supplies a footprint, which only disables the bounds warning.
    """
    bounds = tile.get('bounds') or ref.get('tile_bounds')
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.size == 4:  # [x_min, y_min, x_max, y_max]
            return bounds[:2], bounds[2:]
        if bounds.size == 6:  # [x_min, y_min, z_min, x_max, y_max, z_max]
            return bounds[:2], bounds[3:5]

    center = tile.get('center') or ref.get('tile_center')
    radius = ref.get('tile_radius')
    radius = float(radius) if radius is not None else tile_radius
    if center is None or radius is None:
        return None
    center = np.asarray(center, dtype=np.float64)[:2]
    return center - radius, center + radius


def default_pc_range(tile_radius):
    """The range a training config for this tile size would use: square in
    xy, matching the tile, with the config's z span.

    Because this converter recentres each tile on its own centre, this range
    is centred on the tile too -- no allowance for an offset-frame
    displacement is needed at any tile size.
    """
    radius = tile_radius if tile_radius is not None else FALLBACK_TILE_RADIUS
    z_min, z_max = DEFAULT_Z_RANGE
    return [-radius, -radius, z_min, radius, radius, z_max]


def resolve_classes(selected, manifest):
    """Map ``--classes`` entries (ids or names) onto class_lookup ids.

    Returns a set of int ids, or None for "keep everything". Raises if the
    export has no class taxonomy or a name/id is not in it -- silently
    keeping nothing is how MapTRv2's old --lane-types flag behaved (it
    compared lane_type ids against each polyline's `type`, which holds a
    geometry kind such as 'arc'/'straight'), and it produced an empty pkl
    with no indication why.
    """
    if not selected:
        return None
    lookup = manifest.get('class_lookup') or {}
    if not lookup:
        raise ValueError(
            'this export has no `class_lookup`, so its polylines carry no '
            'class and cannot be filtered; re-run without --classes')
    by_name = {name: int(cid) for cid, name in lookup.items()}
    out = set()
    for entry in selected:
        entry = str(entry)
        if entry in by_name:
            out.add(by_name[entry])
        elif entry.lstrip('-').isdigit() and str(int(entry)) in lookup:
            out.add(int(entry))
        else:
            known = ', '.join(
                f'{cid}={name}' for cid, name in sorted(lookup.items()))
            raise ValueError(
                f'unknown class {entry!r}; this export has: {known}')
    return out


def polyline_class_id(poly):
    """The polyline's class_lookup id, or None on a class-free export.

    Deliberately does not fall back to `type`: that key exists in every
    export and holds a geometry kind ('arc'/'straight'), not a class.
    """
    if poly.get('class_id') is not None:
        return int(poly['class_id'])
    return None


def count_points_in_range(lidar_path, tile_shift, pc_range, z_max):
    """Count a block's points that would survive the training pipeline.

    Mirrors what the model actually sees, in order: the recentring
    ``LoadCarlaPointsFromFile`` applies (``features[:, :3] - tile_shift``),
    then its ``z <= z_max`` filter, then the LiDAR ``Voxelization``'s own
    range filtering. ``GridSamplePoints`` sits between the last two but only
    *clamps* grid coordinates -- it never drops points and never moves them
    -- so it cannot change this count.

    Returns:
        tuple[int, int]: raw point count, and count surviving both filters.
    """
    with np.load(lidar_path) as block:
        xyz = np.asarray(block['features'][:, :3], dtype=np.float32)
    n_raw = int(xyz.shape[0])
    xyz = xyz - np.asarray(tile_shift, dtype=np.float32)
    if z_max is not None:
        xyz = xyz[xyz[:, 2] <= z_max]
    lo = np.asarray(pc_range[:3], dtype=np.float32)
    hi = np.asarray(pc_range[3:], dtype=np.float32)
    # `>= lo` / `< hi` mirrors hard_voxelize's own
    # `floor((p - lo) / voxel) in [0, grid)` convention rather than
    # PointsRangeFilter.in_range_3d's strict-both-sides test. The two differ
    # only for points exactly on a boundary plane, which can never flip a
    # zero/non-zero verdict in practice.
    n_in_range = int(np.all((xyz >= lo) & (xyz < hi), axis=1).sum())
    return n_raw, n_in_range


def convert_carla_tiles(data_root,
                        split,
                        classes=None,
                        pc_range=None,
                        z_max=DEFAULT_Z_MAX,
                        min_lidar_points=1,
                        lidar_check=True):
    tile_dir, manifest = load_manifest(data_root, split)
    tile_radius = manifest_tile_radius(manifest)
    keep_classes = resolve_classes(classes, manifest)
    if pc_range is None:
        pc_range = default_pc_range(tile_radius)
        if tile_radius is None:
            print(f'[warn] this export states no tile size; assuming '
                  f'tile_radius={FALLBACK_TILE_RADIUS}m for the point-cloud '
                  'range. Pass --lidar-point-cloud-range if that is wrong')
    print(f'{tile_dir}: {len(manifest["tiles"])} tiles, tile_radius='
          f'{tile_radius if tile_radius is not None else "unknown"}, '
          f'pc_range={[round(v, 2) for v in pc_range]}')

    samples = []
    dropped = []
    out_of_bounds = []
    coverage = []
    total_instances = 0
    prog_bar = mmcv.ProgressBar(len(manifest['tiles']))
    for idx, tile in enumerate(manifest['tiles']):
        prog_bar.update()
        name = tile['name']

        lidar_path = os.path.join(tile_dir, 'blocks', f'{name}.npz')
        if not os.path.isfile(lidar_path):
            raise FileNotFoundError(lidar_path)

        # np.load is lazy, so this reads only the two small arrays -- it does
        # not pull the point cloud into memory. (count_points_in_range()
        # below does read the full `features` array when --no-lidar-check is
        # off; that is the expensive part of this loop, ~14ms for a
        # 110K-point tile.)
        #
        # Both are needed because they answer different questions: `offset`
        # is the frame `features[:, 0:3]` is stored in, `tile_center` is the
        # frame everything is being moved to. `tile_shift` between them is
        # what the point loader subtracts. See the module docstring.
        with np.load(lidar_path) as block:
            offset = np.asarray(block['offset'], dtype=np.float32)
            center = np.asarray(
                block['tile_center'] if 'tile_center' in block else
                tile['center'],
                dtype=np.float32)
        tile_shift = center - offset

        n_raw = n_in_range = None
        if lidar_check:
            n_raw, n_in_range = count_points_in_range(lidar_path, tile_shift,
                                                      pc_range, z_max)
            if n_in_range < min_lidar_points:
                # This tile would voxelize to zero (or near-zero) voxels and
                # crash extract_lidar_feat mid-run. Drop it before its
                # polylines are counted, so the reported instance totals
                # describe what training will actually see.
                dropped.append(
                    dict(
                        name=name,
                        town=tile.get('town'),
                        reason='lidar',
                        n_points=n_raw,
                        n_points_in_range=n_in_range))
                continue
            if n_raw:
                coverage.append((n_in_range / n_raw, name))

        ref_path = os.path.join(tile_dir, 'reference_lines',
                                f'{name}_reference_lines.json')
        with open(ref_path, encoding='utf-8') as f:
            ref = json.load(f)

        divider = []
        for poly in ref['polylines']:
            if keep_classes is not None and \
                    polyline_class_id(poly) not in keep_classes:
                continue
            pts = np.array(poly['points'], dtype=np.float32)
            if pts.shape[0] < 2:
                continue
            divider.append(pts - center)

        if not divider:
            # MapDiffusion cannot represent a GT-free sample: batch_data()
            # asserts every sample in the batch has at least one line
            # (plugin/models/mapers/MapDiffusion.py), so one of these is a
            # hard crash mid-run rather than a skipped sample.
            dropped.append(
                dict(
                    name=name,
                    town=tile.get('town'),
                    reason='no_gt',
                    n_points=n_raw,
                    n_points_in_range=n_in_range))
            continue
        total_instances += len(divider)

        # Sanity check only -- collect, don't crash, if this fires. Tiles are
        # already the patch, so no clipping is applied.
        #
        # Compared against the tile's own footprint expressed in the frame
        # the GT is now in (world minus `center`), which for a square tile is
        # just +/-tile_radius. Written generally so it stays exact if the
        # exporter ever emits a non-square or off-centre tile.
        footprint = tile_footprint(tile, ref, tile_radius)
        if footprint is not None:
            lo, hi = (footprint[0] - center[:2] - BOUNDS_MARGIN,
                      footprint[1] - center[:2] + BOUNDS_MARGIN)
            pts = np.concatenate([p[:, :2] for p in divider])
            overshoot = float(
                np.max(np.maximum(lo - pts, pts - hi), initial=0.0))
            if overshoot > 0:
                out_of_bounds.append(dict(name=name, overshoot=overshoot))

        samples.append(
            dict(
                # Relative to --data-root, joined back by CarlaDataset. Not
                # absolute: the dataset is bind-mounted at a different path
                # inside the container.
                lidar_path=os.path.relpath(lidar_path, data_root),
                token=name,
                sample_idx=name,
                # Each tile is its own single-frame scene: they are static
                # patches with no temporal relation, so the streaming BEV
                # memory must treat every sample as a first frame, and
                # BaseMapDataset._set_sequence_group_flag must put each in
                # its own group.
                scene_name=name,
                prev=-1,
                next=-1,
                timestamp=idx,
                town=tile.get('town'),
                # A tile's own centre, in world coordinates, doubles as an
                # honest ego2global translation: the GT and points are
                # expressed relative to it, and the tiles share one world
                # frame. Rotation is identity (tiles are axis-aligned).
                e2g_translation=center.tolist(),
                e2g_rotation=[1.0, 0.0, 0.0, 0.0],  # quaternion, wxyz
                tile_center=center.tolist(),
                annotation_origin=center.tolist(),
                # What LoadCarlaPointsFromFile subtracts from features[:,:3]
                # to put the points in the same frame as `annotation`.
                tile_shift=tile_shift.tolist(),
                block_offset=offset.tolist(),
                tile_bounds=tile.get('bounds'),
                # None when --no-lidar-check was passed; the dataset treats a
                # missing count as "unknown" and keeps the sample.
                num_lidar_points=n_raw,
                num_lidar_points_in_range=n_in_range,
                annotation=dict(divider=divider),
            ))

    n = max(len(samples), 1)
    n_lidar_drops = sum(1 for d in dropped if d['reason'] == 'lidar')
    n_gt_drops = sum(1 for d in dropped if d['reason'] == 'no_gt')
    print()
    print(f'{split}: {len(samples)} tiles kept, {len(dropped)} dropped '
          f'({n_lidar_drops} with <{min_lidar_points} in-range LiDAR point'
          f'{"" if min_lidar_points == 1 else "s"}, {n_gt_drops} with no GT '
          f'polyline), {total_instances} divider instances '
          f'({total_instances / n:.1f} per tile)'
          f'{"" if lidar_check else " [LiDAR check skipped]"}')
    for entry in dropped[:20]:
        print(f'  [drop] {entry["name"]} ({entry["reason"]}): '
              f'{entry["n_points"]} raw points, '
              f'{entry["n_points_in_range"]} in range')
    if len(dropped) > 20:
        # A long list usually means the range is wrong, not the data -- the
        # sidecar json below has the rest.
        print(f'  [drop] ... and {len(dropped) - 20} more')
    report_coverage(coverage, pc_range)
    report_out_of_bounds(out_of_bounds)
    return samples, dropped, dict(
        tile_dir=tile_dir,
        tile_radius=tile_radius,
        tile_side=manifest.get('tile_side'),
        pc_range=list(pc_range),
        class_lookup=manifest.get('class_lookup') or {},
        classes_kept=sorted(keep_classes) if keep_classes else None,
        n_out_of_bounds=len(out_of_bounds))


def report_coverage(coverage, pc_range):
    """How much of each tile the configured range actually keeps.

    A range narrower than the tile crops it. Because this converter recentres
    each tile, a square range matching the tile size should keep ~everything;
    a low number here means the range and the export's tile size disagree.
    """
    if not coverage:
        return
    fracs = np.array([c for c, _ in coverage])
    worst_frac, worst_name = min(coverage)
    poor = int((fracs < 0.9).sum())
    print(f'  [range] median {np.median(fracs):.1%} of raw points fall inside '
          f'{[round(v, 2) for v in pc_range]}; worst tile {worst_name} '
          f'{worst_frac:.1%}')
    if poor:
        print(f'  [range] {poor}/{len(fracs)} tiles keep <90% of their points '
              '-- check the range matches the export\'s tile size')


def report_out_of_bounds(out_of_bounds):
    if not out_of_bounds:
        return
    print(f'  [warn] {len(out_of_bounds)} tile(s) have polyline points '
          f'outside their own tile bounds (+{BOUNDS_MARGIN}m margin):')
    for entry in sorted(out_of_bounds, key=lambda e: -e['overshoot'])[:10]:
        print(f'  [warn]   {entry["name"]}: {entry["overshoot"]:.2f}m past '
              'the boundary')
    if len(out_of_bounds) > 10:
        print(f'  [warn]   ... and {len(out_of_bounds) - 10} more')


def main():
    args = parse_args()
    lidar_check = not args.no_lidar_check
    samples, dropped, meta = convert_carla_tiles(
        args.data_root,
        args.split,
        classes=args.classes,
        pc_range=args.lidar_point_cloud_range,
        z_max=args.z_max,
        min_lidar_points=args.min_lidar_points,
        lidar_check=lidar_check)
    mmcv.mkdir_or_exist(args.out_dir)
    out_path = os.path.join(args.out_dir, f'carla_map_infos_{args.split}.pkl')
    mmcv.dump(
        dict(
            samples=samples,
            split=args.split,
            # Absolute on purpose: every sample's lidar_path is relative to
            # it, and CarlaDataset uses it as the join-base fallback when no
            # raw_data_root is configured (matching the sibling repos).
            data_root=os.path.abspath(args.data_root),
            # Geometry of the export these samples came from. CarlaDataset
            # asserts tile_radius against the config's roi_size, which is
            # what stops a 25m config being pointed at the 60m export.
            tile_radius=meta['tile_radius'],
            tile_side=meta['tile_side'],
            class_lookup=meta['class_lookup'],
            classes_kept=meta['classes_kept'],
            # Records the geometry the per-sample counts were measured
            # against, so the dataset can warn if the config it runs under no
            # longer matches.
            lidar_check=dict(
                enabled=lidar_check,
                point_cloud_range=meta['pc_range'],
                z_max=args.z_max,
                min_lidar_points=args.min_lidar_points),
            dropped_tiles=dropped),
        out_path)
    print(f'Saved {out_path}')

    # Sidecar report, so a cluster run can be inspected without unpickling
    # the (large) infos file.
    report_path = os.path.join(args.out_dir,
                               f'carla_map_infos_{args.split}_dropped.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(
            dict(
                split=args.split,
                data_root=args.data_root,
                tile_radius=meta['tile_radius'],
                point_cloud_range=meta['pc_range'],
                z_max=args.z_max,
                min_lidar_points=args.min_lidar_points,
                n_kept=len(samples),
                n_dropped=len(dropped),
                n_out_of_bounds=meta['n_out_of_bounds'],
                dropped=dropped),
            f,
            indent=2)
    print(f'Saved {report_path}')


if __name__ == '__main__':
    main()
