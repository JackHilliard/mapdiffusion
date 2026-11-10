"""HRDX 50 m tiles (San Jose, San Francisco, San Mateo) as WHOLE tiles, with ground truth
sampled on the parametric design equations and the point-cloud augmentation of Pointcept's
configs/nuscenes/semseg-pt-v3m1-0-base.py -- MapDiffusion side.

Port of the sibling MapTR tree's ``CustomHrdx50mDataset`` / ``LoadHrdx50mTile``
(``projects/mmdet3d_plugin/datasets/hrdx50m_dataset.py``) onto this repo's ``BaseMapDataset``
contract, so that every sister baseline (MapTR, GeMap, PseudoMapTrainer, mapdiffusion,
SuperMapNet) trains and evaluates on exactly the same samples. The numbers below are the
benchmark's definition; keep them equal in all five repos.

* Layout. ``data_root`` holds one directory per city (``<city>_tiles_50m_lines/{blocks,
  reference_lines}``); each manifest tile names its ``city`` (split manifests
  ``hrdx_splits_v1/{train,val,test}.json``: 20,499 / 2,227 / 3,392 tiles). Every city has its
  OWN local frame, so a tile is only ever read in its city's frame (tile-local = world - tile
  centre).
* Ground truth. ``gt_source='parametric'`` (default): one instance per continuous in-tile
  stretch of a design alignment, its vertices the equation samples (hrdx_param_gt.py, verbatim
  from the reference repo); ``gt_source='hand'``: the hand-drawn 'curb' polylines.
* No crops. The whole 50 m tile is the sample (crop_size=50, no translation), so the model
  ranges are +-25 m.
* Input. ``(x, y, z, strength)``: xy tile-local, z re-zeroed on the tile's own 2nd-percentile
  ground, strength = the uint8 LiDAR ``intensity`` / 255.
* Augmentation (train only, ``aug`` dict; val/test are never augmented). Pointcept's active
  transforms, in its order:
      RandomRotate(angle=[-1, 1] * pi, axis z, about the tile centre, p=0.5)
      RandomScale([0.9, 1.1])          isotropic, x, y and z
      RandomFlip(p=0.5)                x and y flipped independently, each with p=0.5
      RandomJitter(sigma=0.005, clip=0.02)   points only
  The rotation, scale and flips form ONE 2x2 map A applied to the points (in the loader) and to
  the GT polylines (here) through the inherited pose (crop_R = A), then both are clipped to the
  +-25 m box. z is scaled by the same factor; jitter is applied to the points only. If an
  augmented tile would hold no GT, the draw is repeated (up to ``aug_tries``), then the tile is
  used unaugmented. The draws use the global ``np.random`` stream in the SAME order as the
  MapTR port (``_draw_aug`` per try, then the loader's ``randn(n, 3)``), so two repos seeded
  alike produce the same sample.
* Evaluation. val/test tiles are frozen at load time (identity pose), so the GT is a pure
  function of the split manifest and ``gt_source``. ``map_ann_file`` names the GT cache that
  ``VectorEvaluate`` reads and writes (a pickle keyed by tile name -- this repo's evaluator
  format, NOT the mmdet siblings' json). An existing cache is compared with this dataset's own
  GT before it is used and refused when it differs; it is written atomically.

Differences from the MapTR port that the ``BaseMapDataset`` contract forces: samples are
``map_geoms`` (shapely LineStrings) instead of ``annotation`` arrays; the GT cache is checked
when the evaluator loads it rather than at construction; a train tile that still has no GT
(or too few points) is resampled by the inherited ``__getitem__`` instead of mmdet's
``filter_empty_gt``.
"""
import json
import os
import os.path as osp
from time import time

import numpy as np
from mmdet.datasets import DATASETS
from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.points import get_points_type

from .hrdx_param_gt import chain_parametric
from .pipelines.loading import EmptyLidarTileError, LoadCarla50mCrop
from .sanjose50m_crop_dataset import SanJose50mCropDataset

CITY_DIRS = {'san_jose': 'san_jose_tiles_50m_lines',
             'san_francisco': 'san_francisco_tiles_50m_lines',
             'san_mateo': 'san_mateo_tiles_50m_lines'}

PTV3_NUSCENES_AUG = dict(rotate_p=0.5, rotate_angle=(-1.0, 1.0), scale=(0.9, 1.1),
                         flip_p=0.5, jitter_sigma=0.005, jitter_clip=0.02)

# fixed by the benchmark (whole tiles); a config that passes one of these is a mistake
_FIXED_KWARGS = ('crop_size', 'rotate', 'translate', 'noise_std', 'noise_xy_only', 'gt_classes',
                 'check_finalize', 'reference_dir')


@DATASETS.register_module()
class Hrdx50mDataset(SanJose50mCropDataset):
    """Whole HRDX tiles, parametric (or hand-drawn) curb GT, PTv3-nuScenes augmentation.

    Args (beyond ``SanJose50mCropDataset``'s):
        data_root (str): directory holding the ``<city>_tiles_50m_lines`` directories.
        ann_file (str): a split manifest whose tiles carry ``name``, ``city`` and ``center``.
        gt_source (str): ``'parametric'`` (default) or ``'hand'``.
        aug (dict | None): train augmentation, merged over ``PTV3_NUSCENES_AUG``; ``None``
            (val/test) disables it.
        aug_tries (int): redraws when an augmented tile holds no GT.
        min_crop_points (int | 'manifest'): train-only floor on the raw points inside the
            +-25 m box (0 = off, the benchmark default; ``GridSamplePoints(min_points=...)``
            in the train pipeline is the benchmark's sparse-tile guard).
        map_ann_file (str | None): GT cache of the evaluator (set it on the dataset that
            ``eval_config`` builds). ``None`` keeps the evaluator's ``./tmp_gts_*.pkl``.
        cat2id (dict): exactly one class (the benchmark names it ``boundary``).
        crop_seed (int | None): not None freezes the tiles (val/test): identity pose, no
            augmentation, never skipped.
    """

    def __init__(self, data_root, ann_file, gt_source='parametric', aug=None, aug_tries=8,
                 min_polyline_len=2.0, min_crop_points=0, map_ann_file=None, **kwargs):
        if gt_source not in ('parametric', 'hand'):
            raise ValueError(f"gt_source must be 'parametric' or 'hand', got {gt_source!r}")
        for k in _FIXED_KWARGS:
            if k in kwargs:
                raise ValueError(f'{k} is fixed for whole HRDX tiles; use aug=... for augmentation')
        cat2id = kwargs.get('cat2id') or {}
        if len(cat2id) != 1:
            raise ValueError(f'the HRDX benchmark has exactly one class, got cat2id={cat2id}')
        # read by load_annotations(), which the base __init__ calls
        self.gt_source = gt_source
        self.aug = None if aug is None else {**PTV3_NUSCENES_AUG, **dict(aug)}
        self.aug_tries = int(aug_tries)
        self.hrdx_root = data_root
        self.map_ann_file = map_ann_file
        self._gt_path = {}
        self._label = list(cat2id.values())[0]
        super().__init__(data_root=data_root, ann_file=ann_file, crop_size=50.0, rotate=False,
                         translate=False, noise_std=0.0, min_polyline_len=min_polyline_len,
                         gt_classes=('curb_parametric',) if gt_source == 'parametric' else ('curb',),
                         check_finalize=False, min_crop_points=min_crop_points, **kwargs)

    # ------------------------------------------------------------------
    @property
    def gt_cache_stem(self):
        """Key of the evaluator's default ``./tmp_gts_*.pkl`` (when ``map_ann_file`` is None)."""
        split = osp.splitext(osp.basename(self.ann_file))[0]
        return f'hrdx50m_{split}_{self.gt_source}_n{len(self.samples)}'

    @property
    def gt_cache_file(self):
        """Explicit GT cache path for ``VectorEvaluate`` (None = its default name)."""
        return self.map_ann_file

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
            name, city = t['name'], t['city']
            if city not in CITY_DIRS:
                raise KeyError(f'{ann_file}: tile {name} has unknown city {city!r} '
                               f'(known: {sorted(CITY_DIRS)})')
            d = osp.join(self.hrdx_root, CITY_DIRS[city])
            samples.append(dict(
                token=name, sample_idx=name, scene_name=name, town=city, city=city,
                prev=-1, next=-1, crop_index=i,
                lidar_path=osp.join(d, 'blocks', f'{name}.npz'),
                tile_center=np.asarray(t['center'], dtype=np.float64)[:2]))
            self._gt_path[name] = osp.join(d, 'reference_lines', f'{name}_reference_lines.json')
        self._polys = {s['token']: self._load_polylines(s['token']) for s in samples}
        if self.crop_seed is not None:          # val/test: identity pose, frozen
            for s in samples:
                s['crop_theta'], s['crop_R'], s['crop_shift'] = 0.0, np.eye(2), np.zeros(2)
        n_loaded = len(samples)
        self.samples = samples[::self.interval]
        n_gt = sum(1 for s in self.samples if self._polys[s['token']])
        print(f'collected {len(self.samples)} samples (of {n_loaded}) from {ann_file} in '
              f'{time() - start:.2f}s; whole {2 * self.half:g} m tiles, gt_source={self.gt_source}, '
              f'{n_gt} with GT, aug={"on" if self.aug else "off"}, '
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
        if self.gt_source == 'parametric':
            return [{'points': c['points'].astype(np.float32), 'label': self._label,
                     'is_arc': bool(c['is_arc'])}
                    for c in chain_parametric(gt.get('polylines', []))]
        out = []
        for pl in gt.get('polylines', []):
            if pl.get('class') != 'curb':
                continue
            p = np.asarray(pl['points'], dtype=np.float32)
            if p.ndim == 2 and len(p) >= 2:
                out.append({'points': p, 'label': self._label,
                            'is_arc': bool(pl.get('is_arc', False))})
        return out

    # ------------------------------------------------------------------
    def _draw_aug(self):
        """One draw of Pointcept's rotate / scale / flip, as the 2x2 map A = F * s * R(theta).

        Global ``np.random``, in the MapTR port's order: rotate coin, (angle), scale, flip x,
        flip y.
        """
        a = self.aug
        rng = np.random
        th = float(rng.uniform(a['rotate_angle'][0], a['rotate_angle'][1]) * np.pi) \
            if rng.random() < a['rotate_p'] else 0.0
        s = float(rng.uniform(a['scale'][0], a['scale'][1]))
        fx = -1.0 if rng.random() < a['flip_p'] else 1.0
        fy = -1.0 if rng.random() < a['flip_p'] else 1.0
        c, sn = np.cos(th), np.sin(th)
        A = np.diag([fx, fy]) @ (s * np.array([[c, -sn], [sn, c]], dtype=np.float64))
        return A, s, th

    def get_sample(self, idx):
        sample = self.samples[idx]
        name, centre = sample['token'], sample['tile_center']
        zero = np.zeros(2, dtype=np.float64)
        frozen = 'crop_R' in sample
        A, zs, th = np.eye(2), 1.0, 0.0
        if frozen or self.aug is None:
            geoms = self._map_geoms(name, A, zero, centre)
        else:
            geoms = None
            for _ in range(max(1, self.aug_tries)):
                A, zs, th = self._draw_aug()
                geoms = self._map_geoms(name, A, zero, centre)
                if any(geoms.values()):
                    break
            else:
                A, zs, th = np.eye(2), 1.0, 0.0
                geoms = self._map_geoms(name, A, zero, centre)
        a = self.aug if (self.aug is not None and not frozen) else {}
        return {
            'token': name,
            'sample_idx': name,
            'scene_name': name,
            'town': sample.get('town'),
            'pts_filename': sample['lidar_path'],
            'map_geoms': geoms,
            # every tile is its own single-frame scene, in its own tile-local frame
            'ego2global_translation': [float(centre[0]), float(centre[1]), 0.0],
            'ego2global_rotation': np.eye(3).tolist(),
            # consumed by LoadHrdx50mTile so the points get the SAME map as the GT
            'crop_centre': np.asarray(centre, dtype=np.float64),
            'crop_R': np.asarray(A, dtype=np.float64),
            'crop_shift': zero,
            'crop_half': self.half,
            'crop_z_pct': self.z_ground_percentile,
            'aug_z_scale': float(zs),
            'aug_theta': float(th),
            'aug_jitter_sigma': float(a.get('jitter_sigma', 0.0)),
            'aug_jitter_clip': float(a.get('jitter_clip', 0.0)),
            # train-only raw-point floor (0 = off); frozen tiles are never skipped
            'crop_min_points': 0 if frozen else self.min_crop_points,
        }

    # ------------------------------------------------------------------
    # GT cache of VectorEvaluate: content check (the write is atomic there)
    # ------------------------------------------------------------------
    def gt_vectors(self, idx):
        """This dataset's GT for sample idx, exactly as ``VectorEvaluate.gts`` collects it."""
        return self.pipeline(self.get_sample(idx))['vectors'].data

    def check_gt_cache(self, gts, path='<gt cache>'):
        """Refuse a GT cache whose CONTENT differs from this dataset's own GT.

        ``gts``: ``{token: {label: [(n, 2) array, ...]}}`` as ``VectorEvaluate`` caches it.
        """
        problem = None
        mine_tokens = [s['token'] for s in self.samples]
        if not isinstance(gts, dict):
            problem = f'not a dict ({type(gts).__name__})'
        elif len(gts) != len(mine_tokens) or set(gts) != set(mine_tokens):
            problem = (f'{len(gts)} cached tiles vs {len(mine_tokens)} in this dataset, '
                       f'{len(set(gts) ^ set(mine_tokens))} tile name(s) differ')
        else:
            for i, name in enumerate(mine_tokens):
                mine, theirs = self.gt_vectors(i), gts[name]
                if not isinstance(theirs, dict) or set(theirs) != set(mine):
                    problem = f'sample {i} ({name}): cached labels differ'
                    break
                for lab, vecs in mine.items():
                    cv = theirs[lab]
                    if len(cv) != len(vecs):
                        problem = f'sample {i} ({name}): {len(cv)} cached GT vs {len(vecs)} here'
                        break
                    for a, b in zip(vecs, cv):
                        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
                        if a.shape != b.shape or not np.allclose(a, b, atol=1e-4):
                            problem = f'sample {i} ({name}): GT geometry differs'
                            break
                    if problem:
                        break
                if problem:
                    break
        if problem:
            raise RuntimeError(
                f'stale GT cache {path}: {problem}. The evaluation would score against the '
                'wrong ground truth. Delete the file (it is rebuilt at the next evaluation) '
                'or point map_ann_file at a new path.')


@PIPELINES.register_module(force=True)
class LoadHrdx50mTile(LoadCarla50mCrop):
    """Whole HRDX tile -> ``(x, y, z, strength)`` under the dataset's 2x2 map A.

    xy: ``A (world_xy - tile centre)``, kept inside the +-25 m box; z: re-zeroed on the kept
    points' 2nd percentile, then scaled by the augmentation's scale factor; clipped Gaussian
    jitter on xyz (train only); strength = uint8 ``intensity`` / 255 (never augmented).
    ``use_dim=3`` drops the strength column. Line for line the MapTR port's loader.
    """

    def __init__(self, coord_type='LIDAR', load_dim=4, use_dim=4):
        super().__init__(coord_type=coord_type, load_dim=load_dim, use_dim=use_dim)

    def __call__(self, results):
        with np.load(results['pts_filename']) as d:
            world = d['points'].astype(np.float64)
            inten = d['intensity'] if self._need_strength else None
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
            strength = (inten[keep].astype(np.float32) / np.float32(255.0)).reshape(-1, 1)
            points = np.concatenate([xyz, strength], axis=1)
        else:
            points = xyz
        points = points[:, self.use_dim]
        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(points, points_dim=points.shape[-1], attribute_dims=None)
        return results
