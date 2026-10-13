"""CARLA 50 m tiles -> rotated 30 m crops, augmented at load time.

Port of the sibling MapTRv2 tree's ``CustomCarla50mCropDataset`` /
``LoadCarla50mCrop`` (``projects/mmdet3d_plugin/datasets/
carla50m_crop_dataset.py`` on its ``maptrv2`` branch) onto this repo's
``BaseMapDataset`` contract. The crop semantics are deliberately identical so
the two repos score on the SAME crops of the SAME tiles:

  1. random yaw about the tile centre + random centre shift, then a
     ``crop_size`` (30 m) square CROP. A 30 m square rotated arbitrarily needs
     30*sqrt(2) = 42.43 m, which fits inside the 50 m tile, so every crop is
     fully populated. The leftover slack (25 - 21.21 = 3.79 m) is the maximum
     centre shift.
  2. additive Gaussian noise on the point coordinates (train only);
  3. GT polylines rotated and clipped to the SAME box with the tiler's own
     ``clip_polyline_to_box`` (vendored below, verbatim).

FRAMES (verified against the npz/json on disk):
  * ``points``                world-absolute (N,3)
  * ``features[:, :3]``       points - offset (block-local)   -- unused here
  * GT ``polylines[].points`` WORLD-absolute (N,3)
  * ``tile_center``           world centre of the 50 m tile
The augmentation is done in world frame about ``tile_center`` and the
crop-local frame is what the network sees: the crop centre IS the origin, so
there is no ``tile_shift`` for ``LoadCarlaPointsFromFile`` to subtract --
``LoadCarla50mCrop`` reframes the world points itself with the pose the
dataset drew, so points and GT cannot decouple.

EVAL DETERMINISM. ``VectorEvaluate.gts`` builds the GT by iterating the
eval dataset once and caches it to ``tmp_gts_*.pkl``; the val/test crops must
therefore be identical on every call and every run. With ``crop_seed`` set,
the pose of sample ``i`` is drawn from ``default_rng(crop_seed + i)`` where
``i`` is the tile's position in the *sorted* manifest (before ``interval``
subsampling, so a subset sees the same crops as the full split). The eval
crop is *not* centred: 24.4% of the test tiles have no GT in their centred
30 m crop (the manifest was filtered for a *reachable* non-empty crop); the
seeded aiming drops that to ~0.4%. No yaw at eval, so map quality is not
conflated with rotation robustness.

This is a manifest-driven dataset: ``ann_file`` is the export's
``<split>/manifest.json``, not a converter pkl, so none of the pkl-side
bookkeeping in ``CarlaDataset`` (schema adaptation, zero-voxel filter,
``_check_tile_size``) applies. ``data_root`` is the export root itself::

    <data_root>/<split>/manifest.json               tiles: [{name, center}]
    <data_root>/<split>/blocks/<name>.npz           points (world), tile_center,
                                                    tile_radius
    <data_root>/<split>/<reference_dir>/<name>_reference_lines.json
"""
import json
import os
import os.path as osp
from time import time
from typing import List

import numpy as np
from mmdet.datasets import DATASETS
from shapely.geometry import LineString

from .base_dataset import BaseMapDataset


# ---------------------------------------------------------------------------
# Vendored from Pointcept/data/grid_tile_export.py (via the MapTRv2 port) so
# the crop GT is clipped by EXACTLY the tiler's logic (boundary vertex placed
# on the edge, runs of <2 vertices dropped).
# ---------------------------------------------------------------------------
def _clip_segment_to_box(p0, p1, box):
    """Liang-Barsky. Returns (q0, q1, t0, t1) clipped to box, or None."""
    xmin, ymin, xmax, ymax = box
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin),
                 (dy, ymax - y0)):
        if abs(p) < 1e-12:
            if q < 0:
                return None
        else:
            r = q / p
            if p < 0:
                if r > t1:
                    return None
                if r > t0:
                    t0 = r
            else:
                if r < t0:
                    return None
                if r < t1:
                    t1 = r
    q0 = p0 + t0 * (p1 - p0)
    q1 = p0 + t1 * (p1 - p0)
    return q0, q1, t0, t1


def clip_polyline_to_box(points: np.ndarray, curved: np.ndarray,
                         box) -> List[dict]:
    """Clip a polyline to a square -> continuous sub-polylines reaching the edge."""
    out: List[dict] = []
    cur_pts: List[np.ndarray] = []
    cur_cur: List[bool] = []

    def flush():
        if len(cur_pts) >= 2:
            out.append({
                'points': np.asarray(cur_pts, dtype=np.float32),
                'curved': np.asarray(cur_cur, dtype=bool)
            })
        cur_pts.clear()
        cur_cur.clear()

    for i in range(len(points) - 1):
        res = _clip_segment_to_box(points[i], points[i + 1], box)
        if res is None:
            flush()
            continue
        q0, q1, t0, t1 = res
        c_i = bool(curved[i])
        if not cur_pts:
            cur_pts.append(q0)
            cur_cur.append(c_i)
        elif np.linalg.norm(cur_pts[-1] - q0) > 1e-4:
            flush()
            cur_pts.append(q0)
            cur_cur.append(c_i)
        cur_pts.append(q1)
        cur_cur.append(bool(curved[i + 1]) if t1 >= 1.0 - 1e-9 else c_i)
        if t1 < 1.0 - 1e-9:
            flush()
    flush()
    return out


@DATASETS.register_module()
class Carla50mCropDataset(BaseMapDataset):
    """50 m CARLA tiles -> augmented ``crop_size`` crops, as map_geoms.

    Args (beyond ``BaseMapDataset``'s ``cat2id``/``roi_size``/``meta``/
    ``pipeline``/``interval``/``eval_config``/``test_mode``):
        data_root (str): the export root (see the module docstring).
        ann_file (str): ``<data_root>/<split>/manifest.json``.
        split (str): ``<data_root>/<split>/{blocks,<reference_dir>}`` layout.
        crop_size (float): side of the square crop, metres. Must equal
            ``roi_size`` on both axes (asserted) -- the GT is normalised
            against ``roi_size`` downstream.
        rotate / translate / noise_std: the augmentation. Train: all on.
            Val/test: ``rotate=False``, ``noise_std=0.0`` and a ``crop_seed``.
        require_gt / gt_tries: aim each crop at a random GT vertex so it is
            non-empty (see ``_draw_crop``). ``MapDiffusion.batch_data``
            asserts every training sample has GT, and ``BaseMapDataset``
            has no resample-on-empty path, so this must stay on for train.
        min_polyline_len (float): drop clipped stubs shorter than this.
        gt_classes (tuple[str] | None): the export's class names, in the
            order they map onto ``cat2id``'s labels (positional over
            ``sorted(cat2id, key=cat2id.get)``). ``None`` means the export
            uses the config's own names, which is the case for the
            `driving`/`curb` exports. Polylines carrying no class at all are
            treated as ``gt_classes[0]``.
        reference_dir (str): polyline directory under ``<split>/``.
        z_ground_percentile (float): points' z is re-zeroed on this
            percentile of the crop's own z, per crop (in the loader).
        crop_seed (int | None): freeze the crop pose per index (val/test).
    """

    def __init__(self,
                 data_root,
                 ann_file,
                 split='train',
                 crop_size=30.0,
                 rotate=True,
                 translate=True,
                 noise_std=0.02,
                 noise_xy_only=False,
                 require_gt=True,
                 gt_tries=32,
                 min_polyline_len=2.0,
                 gt_classes=None,
                 reference_dir='reference_lines',
                 z_ground_percentile=2.0,
                 crop_seed=None,
                 cat2id=None,
                 roi_size=None,
                 **kwargs):
        # --- read by load_annotations(), which the base __init__ calls ---
        self.data_root = data_root
        self.split = split
        self.blocks_dir = osp.join(data_root, split, 'blocks')
        self.gt_dir = osp.join(data_root, split, reference_dir)
        self.reference_dir = reference_dir
        self.half = float(crop_size) / 2.0
        self.rotate = bool(rotate)
        self.translate = bool(translate)
        self.noise_std = float(noise_std)
        self.noise_xy_only = bool(noise_xy_only)
        self.require_gt = bool(require_gt)
        self.gt_tries = int(gt_tries)
        self.min_polyline_len = float(min_polyline_len)
        self.z_ground_percentile = float(z_ground_percentile)
        self.crop_seed = crop_seed
        if not np.allclose(np.asarray(roi_size, dtype=np.float64),
                           (float(crop_size), float(crop_size))):
            raise ValueError(
                f'roi_size={tuple(roi_size)} must equal the crop, '
                f'({crop_size}, {crop_size}): VectorizeMap normalises the '
                'crop-local GT against roi_size')
        # Export class name -> label. Labels are cat2id's values; gt_classes
        # pairs positionally with cat2id's names in label order.
        names = [n for n, _ in sorted(cat2id.items(), key=lambda kv: kv[1])]
        self.gt_classes = tuple(gt_classes) if gt_classes else tuple(names)
        if len(self.gt_classes) != len(names):
            raise ValueError(
                f'gt_classes {self.gt_classes} must pair positionally with '
                f'cat2id {names}')
        self._src2label = {
            src: cat2id[name] for src, name in zip(self.gt_classes, names)
        }
        self._tc_z = {}   # tile centre elevation, filled by _load_polylines
        self._polys = {}
        # a 30 m square rotated arbitrarily needs half*sqrt(2)
        self._need = self.half * (np.sqrt(2.0) if self.rotate else 1.0)
        self.max_shift = 0.0  # set in load_annotations once tile_radius is known
        self.tile_radius = None
        super().__init__(ann_file=ann_file, cat2id=cat2id, roi_size=roi_size,
                         **kwargs)

    # ------------------------------------------------------------------
    @property
    def gt_cache_stem(self):
        """Distinct key for ``VectorEvaluate``'s ``tmp_gts_*.pkl`` cache.

        The evaluator keys its cache on the ann_file's basename, which for a
        manifest-driven dataset is always ``manifest`` -- two exports, two
        seeds or two crop aimings would otherwise share one cache file and
        the second silently scores against the first's GT. Name the export
        directory, split, reference dir, seed and tile count instead.
        """
        export = osp.basename(osp.normpath(self.data_root))
        return (f'carla50m_crop_{export}_{self.split}_{self.reference_dir}'
                f'_seed{self.crop_seed}_n{len(self.samples)}')

    def load_annotations(self, ann_file):
        start = time()
        with open(ann_file) as f:
            man = json.load(f)
        tiles = man.get('tiles', [])
        entries = [t if isinstance(t, dict) else {'name': t} for t in tiles]
        if not entries:
            raise RuntimeError(f'no tiles in manifest {ann_file}')
        # Sorted so a sample's position (and therefore its frozen crop pose)
        # is a property of the data, not of the manifest's ordering -- same
        # convention as CarlaDataset; format_results pairs positionally.
        entries.sort(key=lambda t: t['name'])

        # tile radius -> how much centre shift the crop can afford
        with np.load(osp.join(self.blocks_dir,
                              f"{entries[0]['name']}.npz")) as d:
            radius = float(d['tile_radius'])
        self.tile_radius = radius
        if self._need > radius + 1e-6:
            raise ValueError(
                f'crop_size={2*self.half} with rotate={self.rotate} needs '
                f'tile_radius >= {self._need:.2f} m, but tiles have radius '
                f'{radius:.2f} m')
        self.max_shift = max(0.0, radius - self._need) if self.translate else 0.0

        samples = []
        for i, t in enumerate(entries):
            name = t['name']
            centre = (np.asarray(t['center'], dtype=np.float64)[:2]
                      if 'center' in t else None)
            if centre is None:
                with np.load(osp.join(self.blocks_dir, f'{name}.npz')) as d:
                    centre = np.asarray(d['tile_center'], dtype=np.float64)[:2]
            # Polylines are needed on EVERY __getitem__ (a fresh crop each
            # time), so read the json files once here.
            self._polys[name] = self._load_polylines(name)
            samples.append(dict(
                token=name,
                sample_idx=name,
                scene_name=name,
                town=t.get('town'),
                prev=-1,
                next=-1,
                crop_index=i,
                lidar_path=osp.join(self.blocks_dir, f'{name}.npz'),
                tile_center=centre,
            ))

        n_loaded = len(samples)
        # A tile with no polylines at all can never yield a non-empty crop,
        # and MapDiffusion.batch_data asserts every train sample has GT.
        dropped = [s['token'] for s in samples if not self._polys[s['token']]]
        if dropped:
            print(f'[warn] dropped {len(dropped)} tile(s) with no '
                  f'{self.reference_dir} polylines: '
                  f'{dropped[:5]}{" ..." if len(dropped) > 5 else ""}')
            samples = [s for s in samples if self._polys[s['token']]]

        # DETERMINISM FOR EVAL: with crop_seed set, draw the pose ONCE from a
        # per-index seeded rng and freeze it (see the module docstring).
        if self.crop_seed is not None:
            for s in samples:
                rng = np.random.default_rng(self.crop_seed + s['crop_index'])
                th, R, sh = self._draw_crop(self._polys[s['token']],
                                            s['tile_center'], rng)
                s['crop_theta'] = th
                s['crop_R'] = R
                s['crop_shift'] = sh
        samples = samples[::self.interval]
        self.samples = samples
        print(f'collected {len(samples)} samples (of {n_loaded}) from '
              f'{ann_file} in {time() - start:.2f}s; tile_radius={radius} '
              f'crop={2*self.half} max_shift={self.max_shift:.2f} '
              f'rotate={self.rotate} seed={self.crop_seed}')

    # ------------------------------------------------------------------
    def _load_polylines(self, name):
        f = osp.join(self.gt_dir, f'{name}_reference_lines.json')
        if not osp.exists(f):
            self._tc_z[name] = 0.0
            return []
        with open(f) as fh:
            gt = json.load(fh)
        lookup = gt.get('classes') or {}
        tc = gt.get('tile_center')
        self._tc_z[name] = float(tc[2]) if tc is not None and len(tc) > 2 else 0.0
        out = []
        for pl in gt.get('polylines', []):
            cname = pl.get('class')
            if cname is None and 'class_id' in pl:
                cname = lookup.get(str(pl['class_id']))
            if cname is None:
                cname = self.gt_classes[0]
            if cname not in self._src2label:
                continue
            p = np.asarray(pl['points'], dtype=np.float32)
            if p.ndim != 2 or len(p) < 2:
                continue
            out.append({'points': p, 'label': self._src2label[cname],
                        'is_arc': bool(pl.get('is_arc', False))})
        return out

    # ------------------------------------------------------------------
    # Augmentation geometry -- mirrors the MapTRv2 port exactly
    # ------------------------------------------------------------------
    @staticmethod
    def _transform(xy, centre, R, shift=None):
        out = (xy - centre) @ R.T
        return out if shift is None else out - shift

    def _crop_polylines(self, polys, centre, R, shift=None):
        """Rotate + shift into crop frame, then clip to the +-half box."""
        box = (-self.half, -self.half, self.half, self.half)
        pts_out, lbl_out = [], []
        for pl in polys:
            p = pl['points']
            q = p.copy().astype(np.float64)
            q[:, :2] = self._transform(p[:, :2].astype(np.float64), centre, R,
                                       shift)
            curved = np.full(len(q), pl['is_arc'], dtype=bool)
            for sub in clip_polyline_to_box(q.astype(np.float32), curved, box):
                s = sub['points']
                if len(s) < 2:
                    continue
                # drop degenerate stubs the clip leaves at corners
                if float(np.linalg.norm(np.diff(s[:, :2], axis=0),
                                        axis=1).sum()) < self.min_polyline_len:
                    continue
                pts_out.append(np.asarray(s, dtype=np.float32))
                lbl_out.append(pl['label'])
        return pts_out, lbl_out

    def _rand_pose(self, rng):
        th = float(rng.uniform(0.0, 2.0 * np.pi)) if self.rotate else 0.0
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        if self.max_shift > 0 and self.translate:
            rad = self.max_shift * np.sqrt(rng.random())
            ang = rng.uniform(0.0, 2.0 * np.pi)
            sh = np.array([rad * np.cos(ang), rad * np.sin(ang)],
                          dtype=np.float64)
        else:
            sh = np.zeros(2, dtype=np.float64)
        return th, R, sh

    def _draw_crop(self, polys, centre, rng):
        """Pick (yaw, shift), AIMING the crop at a random GT vertex.

        Rotate freely, then pull the crop centre along the chosen vertex
        direction (bounded by max_shift) so the vertex lands inside the box.
        Each try is verified; falls back to the GT vertex closest to the
        tile centre, then to a plain random pose.
        """
        if not (self.require_gt and polys):
            return self._rand_pose(rng)
        for _ in range(max(1, self.gt_tries)):
            pl = polys[int(rng.integers(len(polys)))]
            q = pl['points']
            p = q[int(rng.integers(len(q)))][:2].astype(np.float64)
            th, R, _ = self._rand_pose(rng)
            v = (p - centre) @ R.T
            n = float(np.hypot(v[0], v[1]))
            if n > 1e-9:
                pull = min(self.max_shift, max(0.0, n - 0.6 * self.half))
                sh = v * (pull / n)
            else:
                sh = np.zeros(2, dtype=np.float64)
            if self._crop_polylines(polys, centre, R, sh)[0]:
                return th, R, sh
        best = None
        for pl in polys:
            q = np.asarray(pl['points'], dtype=np.float64)[:, :2]
            d = np.hypot(q[:, 0] - centre[0], q[:, 1] - centre[1])
            j = int(np.argmin(d))
            if best is None or d[j] < best[0]:
                best = (float(d[j]), q[j])
        if best is not None:
            p = best[1]
            for th in (0.0,) + tuple(
                    np.linspace(0, 2 * np.pi, 8, endpoint=False)[1:]):
                c, s_ = np.cos(th), np.sin(th)
                R = np.array([[c, -s_], [s_, c]], dtype=np.float64)
                v = (p - centre) @ R.T
                n = float(np.hypot(v[0], v[1]))
                sh = v * (min(self.max_shift, max(0.0, n - 0.6 * self.half)) / n) \
                    if n > 1e-9 else np.zeros(2, dtype=np.float64)
                if self._crop_polylines(polys, centre, R, sh)[0]:
                    return float(th), R, sh
        return self._rand_pose(rng)

    # ------------------------------------------------------------------
    def _map_geoms(self, name, R, shift, centre):
        """Crop GT -> ``{label: [LineString, ...]}`` in crop-local metres.

        z is re-zeroed on the tile centre elevation so it is a sane number
        for anything that reads it; ``VectorizeMap`` slices to coords_dim=2
        so it never reaches the model. It is NOT the same reference as the
        points' z, which the loader re-zeroes on the crop's own ground.
        """
        pts, lbls = self._crop_polylines(self._polys.get(name, []), centre, R,
                                         shift)
        z0 = self._tc_z.get(name, 0.0)
        geoms = {label: [] for label in self.cat2id.values()}
        for p, lbl in zip(pts, lbls):
            q = np.zeros((len(p), 3), dtype=np.float64)
            q[:, :2] = p[:, :2]
            if p.shape[1] > 2:
                q[:, 2] = p[:, 2] - z0
            geoms[lbl].append(LineString(q))
        return geoms

    def get_sample(self, idx):
        sample = self.samples[idx]
        name = sample['token']
        centre = sample['tile_center']
        if 'crop_R' in sample:          # frozen (seeded) split
            R, shift = sample['crop_R'], sample['crop_shift']
        else:                           # train: fresh crop every epoch
            rng = np.random.default_rng(
                None if self.crop_seed is None
                else self.crop_seed + sample['crop_index'])
            _, R, shift = self._draw_crop(self._polys.get(name, []), centre,
                                          rng)
        return {
            'token': name,
            'sample_idx': name,
            'scene_name': name,
            'town': sample.get('town'),
            'pts_filename': sample['lidar_path'],
            'map_geoms': self._map_geoms(name, R, shift, centre),
            # The crop's own world-frame centre, identity rotation (every
            # crop is its own single-frame scene; see CarlaDataset).
            'ego2global_translation': [
                float(centre[0] + (R.T @ shift)[0]),
                float(centre[1] + (R.T @ shift)[1]), 0.0],
            'ego2global_rotation': np.eye(3).tolist(),
            # consumed by LoadCarla50mCrop so points get the SAME pose as GT
            'crop_centre': np.asarray(centre, dtype=np.float64),
            'crop_R': np.asarray(R, dtype=np.float64),
            'crop_shift': np.asarray(shift, dtype=np.float64),
            'crop_half': self.half,
            'crop_noise_std': self.noise_std,
            'crop_noise_xy_only': self.noise_xy_only,
            'crop_z_pct': self.z_ground_percentile,
        }
