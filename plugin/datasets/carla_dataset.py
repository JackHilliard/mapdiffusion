import os.path as osp
from time import time

import mmcv
import numpy as np
from mmdet.datasets import DATASETS
from shapely.geometry import LineString

from .base_dataset import BaseMapDataset


@DATASETS.register_module()
class CarlaDataset(BaseMapDataset):
    """CARLA static-tile map dataset.

    Unlike :class:`NuscDataset` and :class:`AV2Dataset`, nothing here is
    extracted from a map API at load time: the GT polylines are already in
    the annotation pkl, tile-local and tile-centred, written by
    ``tools/data_converter/carla_converter.py``. ``get_sample`` only has to
    wrap them as shapely geometries and point the pipeline at the tile's
    ``.npz`` block.

    Each tile is a static 25m (or 60m) patch with no temporal relation to any
    other, so every sample is its own single-frame scene: ``prev`` is always
    -1, ``scene_name`` is the tile name, and the streaming BEV memory
    therefore treats every sample as a first frame.

    Args:
        data_root (str): root of the CARLA tile export, i.e. what each
            sample's relative ``lidar_path`` is joined onto. This is *not*
            where the pkl lives.
        ann_file (str): annotation pkl path
        cat2id (dict): category to class id
        roi_size (tuple): bev range
        eval_config (Config): evaluation config
        meta (dict): meta information
        pipeline (Config): data processing pipeline config
        interval (int): annotation load interval
        work_dir (str): path to work dir
        test_mode (bool): whether in test mode
    """

    def __init__(self, data_root, **kwargs):
        self.data_root = data_root
        # Set by load_annotations(), which the base __init__ calls.
        self.ann_meta = {}
        super().__init__(**kwargs)
        self._check_tile_size()

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        start_time = time()
        ann = mmcv.load(ann_file)
        self.ann_meta = {k: v for k, v in ann.items() if k != 'samples'}
        # Sorted so a sample's position is a property of the data rather than
        # of the manifest's ordering: format_results() and the evaluator pair
        # predictions to samples positionally.
        samples = sorted(ann['samples'], key=lambda s: s['sample_idx'])
        n_loaded = len(samples)
        samples = self._filter_empty_lidar_tiles(samples)
        samples = samples[::self.interval]
        self.samples = samples

        print(f'collected {len(samples)} samples (of {n_loaded}) in '
              f'{(time() - start_time):.2f}s')

    def _filter_empty_lidar_tiles(self, samples):
        """Drop tiles that would voxelize to zero voxels.

        A tile with no point inside the LiDAR range crashes
        ``extract_lidar_feat`` mid-run. The converter already drops these and
        records ``num_lidar_points_in_range`` on every kept sample, so this is
        a cheap re-check that also protects a pkl generated before that
        existed.

        This has to happen here rather than in ``__getitem__``: the base
        ``__init__`` builds ``idx2token``, ``self.flag`` and the sequence
        groups straight after ``load_annotations``, and ``format_results``
        indexes results positionally against ``self.samples``, so a later skip
        would desynchronise evaluation. ``BaseMapDataset.__getitem__`` also
        has no resample-on-None path to fall back to.
        """
        check = self.ann_meta.get('lidar_check') or {}
        counted = [
            s for s in samples if s.get('num_lidar_points_in_range') is not None
        ]
        if not counted:
            print('[warn] this annotation file records no '
                  'num_lidar_points_in_range, so empty (zero-voxel) tiles '
                  'cannot be filtered here; regenerate it with '
                  'tools/data_converter/carla_converter.py if training dies '
                  'inside extract_lidar_feat')
            return samples

        min_points = max(int(check.get('min_points', 1) or 1), 1)
        kept, dropped = [], []
        for s in samples:
            n = s.get('num_lidar_points_in_range')
            if n is None or n >= min_points:
                kept.append(s)
            else:
                dropped.append(s['sample_idx'])
        if dropped:
            print(f'[warn] dropped {len(dropped)} tile(s) with <{min_points} '
                  f'in-range LiDAR point(s): '
                  f'{dropped[:5]}{" ..." if len(dropped) > 5 else ""}')
        return kept

    def _check_tile_size(self):
        """Fail loudly when the config's ROI does not match the export.

        The exporter has produced 25m and 60m tiles. Pointing a config at the
        wrong one is silent otherwise -- the GT still loads, it is just
        normalized against the wrong extent -- so it is worth an assert.
        """
        tile_radius = self.ann_meta.get('tile_radius')
        if tile_radius is None:
            return
        expected = (2 * float(tile_radius), 2 * float(tile_radius))
        if not np.allclose(np.asarray(self.roi_size, dtype=np.float64),
                           expected):
            raise ValueError(
                f'roi_size={tuple(self.roi_size)} does not match this '
                f'export\'s tiles (tile_radius={tile_radius}, so '
                f'roi_size should be {expected}). The GT is tile-centred and '
                'normalized against roi_size, so a mismatch silently '
                'rescales every map element.')

        check = self.ann_meta.get('lidar_check') or {}
        pc_range = check.get('point_cloud_range')
        if pc_range is not None and not np.allclose(
                pc_range[:2] + pc_range[3:5],
                [-tile_radius, -tile_radius, tile_radius, tile_radius]):
            print(f'[warn] this pkl\'s LiDAR point counts were measured '
                  f'against xy range {pc_range[:2] + pc_range[3:5]}, which is '
                  f'not +/-{tile_radius}; the zero-voxel filter above may not '
                  'describe what training will see')

    def get_sample(self, idx):
        """Get data sample.

        Args:
            idx (int): data index

        Returns:
            result (dict): dict of input
        """
        sample = self.samples[idx]

        map_label2geom = {}
        for name, lines in sample['annotation'].items():
            if name not in self.cat2id:
                continue
            # Lines are (N, 3) in the tile-centred frame already; VectorizeMap
            # slices them to coords_dim and normalizes against roi_size.
            map_label2geom[self.cat2id[name]] = [
                LineString(np.asarray(line)) for line in lines
            ]

        input_dict = {
            'token': sample['token'],
            'sample_idx': sample['sample_idx'],
            'scene_name': sample['scene_name'],
            'town': sample.get('town'),
            'pts_filename': osp.join(self.data_root, sample['lidar_path']),
            # LoadCarlaPointsFromFile subtracts this from features[:, 0:3] to
            # land in the same tile-centred frame as map_geoms above.
            'tile_shift': sample['tile_shift'],
            'map_geoms': map_label2geom,  # {0: List[LineString], ...}
            # A tile's own world-frame centre, with identity rotation. The
            # streaming BEV memory never warps between tiles (each is its own
            # scene) but the head and memory read these unconditionally.
            'ego2global_translation': sample['e2g_translation'],
            'ego2global_rotation': np.eye(3).tolist(),
        }

        return input_dict
