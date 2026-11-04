"""HRDX San Jose 50 m tiles -> rotated 30 m crops (curb only), MapDiffusion side.

The San Jose export (``san_jose_tiles_50m``) carries the same per-tile files
as the CARLA 50 m export -- ``blocks/<name>.npz`` with world-frame ``points``,
``tile_center`` and ``tile_radius`` 25, and ``reference_lines/<name>_
reference_lines.json`` with world-frame polylines -- so the whole crop
machinery of :class:`Carla50mCropDataset` applies unchanged. What differs:

1. **Flat layout, external split manifests.** There is no ``<split>/``
   level: ``blocks/`` and ``reference_lines/`` sit directly under the export
   root, and the train/val/test manifests live in a separate directory
   (``san_jose_splits_v1/v1/{train,val,test}.json``). ``ann_file`` is any
   of those; ``data_root`` is the export root.
2. **One class, ``curb``.** The export's ``class_lookup`` is
   ``{1: curb, 2: curb_parametric}``; the parametric class re-traces the
   same curbs and must NOT be trained alongside. ``gt_classes`` defaults to
   ``('curb',)``.
3. **Manifest order, not sorted order, for the frozen pose.** The splits
   were finalised with the MapTRv2 loader (``san_jose_splits_v1/tools/
   sj_finalize_splits.py``): every val/test tile is guaranteed a non-empty,
   >= 1000-point crop at the pose ``default_rng(crop_seed + p)`` where
   ``p`` is the tile's POSITION IN THE MANIFEST. ``Carla50mCropDataset``
   sorts its tiles by name first, which would re-index the seeds and void
   that guarantee (and score different crops from the sibling repos), so
   this class keeps manifest order. The manifests' ``finalize`` block is
   checked against the dataset's own kwargs and a mismatch warns loudly.
4. **Own GT cache key.** ``gt_cache_stem`` names the split json, so val
   and test (which differ here, unlike the CARLA export) never share a
   ``tmp_gts_*.pkl``.
"""
import json
import os
import os.path as osp
import warnings
from time import time

import numpy as np
from mmdet.datasets import DATASETS

from .carla50m_crop_dataset import Carla50mCropDataset

SANJOSE_GT_CLASSES = ('curb',)


@DATASETS.register_module()
class SanJose50mCropDataset(Carla50mCropDataset):
    """San Jose 50 m tiles -> 30 m crops, in ``Carla50mCropDataset``'s contract.

    Args (beyond the parent's): ``data_root`` is the flat export root
    (``split`` must be empty / None); ``ann_file`` a split json from
    ``san_jose_splits_v1``; ``gt_classes`` defaults to ``('curb',)``;
    ``check_finalize`` (default True) compares the manifest's ``finalize``
    block with the crop kwargs.
    """

    def __init__(self, data_root, ann_file, split=None, gt_classes=None,
                 check_finalize=True, **kwargs):
        if split:
            raise ValueError(
                f'{self.__class__.__name__}: the San Jose export is flat '
                f'(<data_root>/blocks, <data_root>/<reference_dir>); got '
                f'split={split!r}')
        self.check_finalize = bool(check_finalize)
        self.finalize = {}
        super().__init__(data_root=data_root, ann_file=ann_file, split='',
                         gt_classes=gt_classes or SANJOSE_GT_CLASSES, **kwargs)

    @property
    def gt_cache_stem(self):
        export = osp.basename(osp.normpath(self.data_root))
        split = osp.splitext(osp.basename(self.ann_file))[0]
        return (f'sanjose50m_crop_{export}_{split}_{self.reference_dir}'
                f'_seed{self.crop_seed}_n{len(self.samples)}')

    def load_annotations(self, ann_file):
        start = time()
        self.ann_file = ann_file
        with open(ann_file) as f:
            man = json.load(f)
        self.finalize = dict(man.get('finalize') or {})
        lookup = man.get('class_lookup') or {}
        if lookup:
            missing = [c for c in self.gt_classes if c not in lookup.values()]
            if missing:
                raise ValueError(f'{ann_file}: gt_classes {missing} not in '
                                 f'the export class_lookup {lookup}')
        tiles = man.get('tiles', [])
        entries = [t if isinstance(t, dict) else {'name': t} for t in tiles]
        if not entries:
            raise RuntimeError(f'no tiles in manifest {ann_file}')
        # MANIFEST order on purpose -- see the module docstring (3).

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
            self._polys[name] = self._load_polylines(name)
            samples.append(dict(
                token=name, sample_idx=name, scene_name=name,
                town=t.get('town'), prev=-1, next=-1, crop_index=i,
                lidar_path=osp.join(self.blocks_dir, f'{name}.npz'),
                tile_center=centre))

        n_loaded = len(samples)
        dropped = [s['token'] for s in samples if not self._polys[s['token']]]
        if dropped:
            # Cannot happen on a finalised split; kept as the parent's guard.
            warnings.warn(f'dropped {len(dropped)} tile(s) with no '
                          f'{self.reference_dir} polylines: {dropped[:5]} -- '
                          f'their positions still consume a frozen-pose index')
            samples = [s for s in samples if self._polys[s['token']]]

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
        if self.check_finalize:
            self._check_finalize(ann_file)

    def _check_finalize(self, ann_file):
        fz = self.finalize
        if not fz:
            warnings.warn(
                f'{ann_file} has no "finalize" block: it was not filtered '
                f'with the loader, so frozen crops may be GT-empty or '
                f'point-free (see sj_finalize_splits.py)')
            return
        problems = []
        if tuple(fz.get('gt_classes', ())) != tuple(self.gt_classes):
            problems.append(f'gt_classes {tuple(self.gt_classes)} != '
                            f'finalised {tuple(fz.get("gt_classes", ()))}')
        ms = fz.get('max_shift_m')
        if ms is not None and abs(float(ms) - self.max_shift) > 1e-3:
            problems.append(f'max_shift {self.max_shift:.3f} m != finalised '
                            f'{float(ms):.3f} m')
        if 'crop_seed' in fz:
            if self.crop_seed != fz['crop_seed']:
                problems.append(f'crop_seed {self.crop_seed} != finalised '
                                f'{fz["crop_seed"]}')
            if self.rotate:
                problems.append('rotate=True on a split finalised unrotated')
        elif self.crop_seed is not None:
            warnings.warn(f'{ann_file} is a train split but crop_seed='
                          f'{self.crop_seed} freezes its crops')
        if problems:
            warnings.warn(
                f'{self.__class__.__name__}: {ann_file} was finalised with '
                f'different crop settings -- its non-empty-crop guarantee '
                f'does not hold: ' + '; '.join(problems))
        else:
            print(f'[{self.__class__.__name__}] {os.path.basename(ann_file)}: '
                  f'finalize block matches (gt_classes={tuple(self.gt_classes)}'
                  f', max_shift={self.max_shift:.2f} m, crop_seed={self.crop_seed})')
