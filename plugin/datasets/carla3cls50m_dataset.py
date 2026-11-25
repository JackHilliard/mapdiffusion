"""CARLA 3-class 50 m tiles (point2vector_data, Town12/13 + held-out Town01-05) as WHOLE tiles,
loaded exactly like the HRDX benchmark (hrdx50m_dataset.py) wherever the data allows --
MapDiffusion side.

Port of the sibling MapTR tree's ``CustomCarla3cls50mDataset`` / ``LoadCarla3cls50mTile``
(``projects/mmdet3d_plugin/datasets/carla3cls50m_dataset.py``) onto this repo's
``BaseMapDataset`` contract, the same way ``Hrdx50mDataset`` ports ``CustomHrdx50mDataset``.

Same as HRDX, value for value: the whole 50 m tile is the sample (+-25 m, no crop, no
translation); xy tile-local (world - tile centre); z re-zeroed on the tile's own 2nd-percentile
ground; the PTv3-nuScenes train augmentation (rotate / scale / flip as one 2x2 map on points and
GT, jitter on points only; same RNG order); GT clipped to the box, pieces < 2 m dropped; val and
test frozen at the identity pose; the evaluator's GT cache content-checked and written
atomically (``Hrdx50mDataset.check_gt_cache``, inherited).

What differs, because the data differs:

* Layout. ``data_root`` is the root of ``carla_3cls_50m`` (the point2vector_data ``outputs/``
  tree, cut down to the split tiles). Each manifest tile names its ``tiles_dir`` (e.g.
  ``Town12/sweep/tiles_50m``), holding ``blocks/<name>.npz`` and
  ``reference_lines/<name>_reference_lines.json``. Split manifests
  ``carla_splits_v1/{train,val,test,test_towns}.json`` (18,522 / 2,236 / 2,309 / 688 tiles);
  ``test_towns`` = Town01-05 held out whole. Tile names repeat across towns, so the sample key
  (``token`` / ``sample_idx`` / ``scene_name``, and the GT cache key) is ``<town>_<name>``.
* Classes. Three: ``centerline``, ``curb_model``, ``road_edge`` (the export's own names). The
  config's ``cat2id`` names them; its label order is the class order. Pieces of one class along
  one road are joined at the annotation's straight/arc cuts (carla_chain_gt.py, shared verbatim
  by all sister repos), the counterpart of HRDX's per-alignment chaining of design elements.
* Strength. The CARLA aerial LiDAR has no intensity. The 4th input channel is kept so the model
  is the HRDX one unchanged, and filled with ZEROS (``strength='zero'``): a constant zero input
  channel contributes nothing to the first (bias-free sparse) convolution, so this is the same
  model as a 3-channel one. ``strength='luma'`` uses the BT.709 luma of the camera colour instead
  (not the benchmark).
"""
import json
import os.path as osp
from time import time

import numpy as np
from mmdet.datasets import DATASETS
from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.points import get_points_type

from .carla_chain_gt import CARLA_3CLS_CLASSES, chain_pieces
from .hrdx50m_dataset import _FIXED_KWARGS, PTV3_NUSCENES_AUG, Hrdx50mDataset, LoadHrdx50mTile
from .pipelines.loading import EmptyLidarTileError
from .sanjose50m_crop_dataset import SanJose50mCropDataset


@DATASETS.register_module()
class Carla3cls50mDataset(Hrdx50mDataset):
    """Whole CARLA 50 m tiles, 3 classes, HRDX's augmentation / eval / GT-cache behaviour.

    Args (beyond ``SanJose50mCropDataset``'s):
        data_root (str): the ``carla_3cls_50m`` root (tile paths = data_root/tiles_dir/...).
        ann_file (str): a split manifest whose tiles carry ``name``, ``city``, ``tiles_dir``,
            ``center``.
        cat2id (dict): export class names -> labels (the benchmark: all three, in the order
            centerline, curb_model, road_edge).
        aug, aug_tries, min_crop_points, map_ann_file, crop_seed: as ``Hrdx50mDataset``.
        gap_tol (float): end-to-end tolerance for joining pieces (carla_chain_gt.chain_pieces).
    """

    def __init__(self, data_root, ann_file, aug=None, aug_tries=8, min_polyline_len=2.0,
                 min_crop_points=0, map_ann_file=None, gap_tol=0.15, **kwargs):
        for k in _FIXED_KWARGS + ('gt_source',):
            if k in kwargs:
                raise ValueError(f'{k} is fixed for whole CARLA tiles')
        cat2id = kwargs.get('cat2id') or {}
        names = [n for n, _ in sorted(cat2id.items(), key=lambda kv: kv[1])]
        unknown = [c for c in names if c not in CARLA_3CLS_CLASSES]
        if not names or unknown:
            raise ValueError(f'cat2id must name CARLA export classes {CARLA_3CLS_CLASSES}, '
                             f'got {cat2id}')
        if sorted(cat2id.values()) != list(range(len(cat2id))):
            raise ValueError(f'cat2id labels must be 0..{len(cat2id) - 1}, got {cat2id}')
        self.gt_source = 'carla'
        self.aug = None if aug is None else {**PTV3_NUSCENES_AUG, **dict(aug)}
        self.aug_tries = int(aug_tries)
        self.hrdx_root = data_root
        self.map_ann_file = map_ann_file
        self.gap_tol = float(gap_tol)
        self._gt_path = {}
        # skip Hrdx50mDataset.__init__ (one class, HRDX cities); everything after it is shared.
        # gt_classes = the export names in label order (Carla50mCropDataset pairs them
        # positionally with cat2id -> self._src2label)
        SanJose50mCropDataset.__init__(
            self, data_root=data_root, ann_file=ann_file, crop_size=50.0, rotate=False,
            translate=False, noise_std=0.0, min_polyline_len=min_polyline_len,
            gt_classes=tuple(names), check_finalize=False, min_crop_points=min_crop_points,
            **kwargs)

    # ------------------------------------------------------------------
    @property
    def gt_cache_stem(self):
        """Key of the evaluator's default ``./tmp_gts_*.pkl`` (when ``map_ann_file`` is None)."""
        split = osp.splitext(osp.basename(self.ann_file))[0]
        return f'carla3cls50m_{split}_n{len(self.samples)}'

    # ------------------------------------------------------------------
    def load_annotations(self, ann_file):
        start = time()
        self.ann_file = ann_file
        with open(ann_file) as f:
            man = json.load(f)
        self.min_crop_points = self._resolve_min_points(self.min_crop_points, man, ann_file)
        self.max_shift = 0.0            # whole tiles: nothing to shift
        self.tile_radius = self.half
        self.finalize = {}
        tiles = man.get('tiles', [])
        if not tiles:
            raise RuntimeError(f'no tiles in manifest {ann_file}')
        samples = []
        for i, t in enumerate(tiles):       # MANIFEST order, every tile kept (GT-less ones too)
            name, tdir = t['name'], t['tiles_dir']
            town = t.get('city') or tdir.split('/')[0]
            d = osp.join(self.hrdx_root, tdir)
            # tile names repeat across towns (tile_00000): the sample key is <town>_<name>
            key = f'{town}_{name}'
            samples.append(dict(
                token=key, sample_idx=key, scene_name=key, town=town, city=town,
                prev=-1, next=-1, crop_index=i,
                lidar_path=osp.join(d, 'blocks', f'{name}.npz'),
                tile_center=np.asarray(t['center'], dtype=np.float64)[:2]))
            self._gt_path[key] = osp.join(d, 'reference_lines', f'{name}_reference_lines.json')
        self._polys = {s['token']: self._load_polylines(s['token']) for s in samples}
        if self.crop_seed is not None:          # val/test: identity pose, frozen
            for s in samples:
                s['crop_theta'], s['crop_R'], s['crop_shift'] = 0.0, np.eye(2), np.zeros(2)
        n_loaded = len(samples)
        self.samples = samples[::self.interval]
        n_gt = sum(1 for s in self.samples if self._polys[s['token']])
        print(f'collected {len(self.samples)} samples (of {n_loaded}) from {ann_file} in '
              f'{time() - start:.2f}s; whole {2 * self.half:g} m CARLA tiles, classes '
              f'{list(self.gt_classes)}, {n_gt} with GT, aug={"on" if self.aug else "off"}, '
              f'frozen={self.crop_seed is not None}')

    def _load_polylines(self, name):
        f = self._gt_path.get(name)
        if f is None or not osp.exists(f):
            self._tc_z[name] = 0.0
            return []
        with open(f) as fh:
            gt = json.load(fh)
        tc = gt.get('tile_center')
        self._tc_z[name] = float(tc[2]) if tc is not None and len(tc) > 2 else 0.0
        return [{'points': c['points'].astype(np.float32), 'label': self._src2label[c['cls']],
                 'is_arc': bool(c['is_arc'])}
                for c in chain_pieces(gt.get('polylines', []), classes=self.gt_classes,
                                      gap_tol=self.gap_tol)]


@PIPELINES.register_module(force=True)
class LoadCarla3cls50mTile(LoadHrdx50mTile):
    """Whole CARLA tile -> ``(x, y, z, strength)``, exactly ``LoadHrdx50mTile`` except for the
    strength column: ``'zero'`` (benchmark default; CARLA has no intensity) or ``'luma'``.
    Line for line the MapTR port's loader."""

    def __init__(self, coord_type='LIDAR', load_dim=4, use_dim=4, strength='zero'):
        if strength not in ('zero', 'luma'):
            raise ValueError(f"strength must be 'zero' or 'luma', got {strength!r}")
        self.strength = strength
        super().__init__(coord_type=coord_type, load_dim=load_dim, use_dim=use_dim)

    def __call__(self, results):
        with np.load(results['pts_filename']) as d:
            world = d['points'].astype(np.float64)
            rgb = d['features'][:, 3:6] if (self._need_strength and self.strength == 'luma') else None
        A = results['crop_R']
        half = results['crop_half']
        local = (world[:, :2] - results['crop_centre']) @ A.T - results['crop_shift']
        keep = (np.abs(local[:, 0]) <= half) & (np.abs(local[:, 1]) <= half)
        n = int(keep.sum())
        need = int(results.get('crop_min_points', 0) or 0)
        if n < need:
            raise EmptyLidarTileError(
                f'tile {results.get("sample_idx")}: {n} raw point(s) in the '
                f'{2 * half:g} m tile (need >= {need})')
        xyz = np.empty((n, 3), dtype=np.float32)
        xyz[:, :2] = local[keep].astype(np.float32)
        xyz[:, 2] = world[keep, 2].astype(np.float32)
        if n:
            xyz[:, 2] -= np.percentile(xyz[:, 2], float(results.get('crop_z_pct', 2.0)))
        zs = float(results.get('aug_z_scale', 1.0))
        if zs != 1.0:
            xyz[:, 2] *= np.float32(zs)
        sig = float(results.get('aug_jitter_sigma', 0.0))
        if sig > 0 and n:
            clip = float(results.get('aug_jitter_clip', 0.0)) or np.inf
            xyz += np.clip(sig * np.random.randn(n, 3), -clip, clip).astype(np.float32)
        if self._need_strength:
            if rgb is None:
                strength = np.zeros((n, 1), dtype=np.float32)
            else:
                strength = (rgb[keep].astype(np.float32) @ self._rgb2strength).reshape(-1, 1)
            points = np.concatenate([xyz, strength], axis=1)
        else:
            points = xyz
        points = points[:, self.use_dim]
        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(points, points_dim=points.shape[-1], attribute_dims=None)
        return results
