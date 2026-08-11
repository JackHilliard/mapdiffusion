import mmcv
import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.points import get_points_type

@PIPELINES.register_module(force=True)
class LoadMultiViewImagesFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filenames']
        img = [mmcv.imread(name, self.color_type) for name in filename]
        if self.to_float32:
            img = [i.astype(np.float32) for i in img]
        results['img'] = img
        results['img_shape'] = [i.shape for i in img]
        results['ori_shape'] = [i.shape for i in img]
        # Set initial values for default meta_keys
        results['pad_shape'] = [i.shape for i in img]
        # results['scale_factor'] = 1.0
        num_channels = 1 if len(img[0].shape) < 3 else img[0].shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__} (to_float32={self.to_float32}, '\
            f"color_type='{self.color_type}')"


class EmptyLidarTileError(RuntimeError):
    """A tile has no LiDAR point inside the voxelizer's range.

    Such a tile voxelizes to zero voxels, which crashes
    ``BEVFormerBackbone.extract_lidar_feat``. Raised by ``GridSamplePoints``
    so the failure names the offending tile instead of surfacing as a bare
    IndexError deep in the encoder. Tiles like this are dropped at conversion
    time and again in ``CarlaDataset.load_annotations``; this is the last
    line of defence, for an annotation pkl that predates both.

    Note this is fatal here, unlike in the sibling MapTRv2 tree where
    ``prepare_train_data`` catches it and resamples: ``BaseMapDataset``'s
    ``__getitem__`` is a bare ``self.pipeline(self.get_sample(idx))`` with no
    ``_rand_another`` retry, and skipping at test time would desynchronise
    ``format_results``' positional indexing.
    """


@PIPELINES.register_module(force=True)
class LoadCarlaPointsFromFile(object):
    """Load a CARLA-simulator LiDAR point cloud from an ``.npz`` tile block.

    Each block stores a ``features`` array of shape ``(N, 6)`` (xyz + rgb).
    This builds the LiDAR point cloud the model sees: xyz plus a scalar
    "strength" derived from the RGB channels via ITU-R BT.709 luma, matching
    the ``strength = rgb @ [0.2126, 0.7152, 0.0722]`` formula the sibling
    MapTRv2/GeMap loaders use.

    Two frames are in play and it matters which one wins. ``features[:, 0:3]``
    is stored relative to the block's ``offset``, in which the tile is *not*
    centred -- it sits around ``tile_center - offset``, 1.6m off on average
    for the 25m export (max 12.1m) and up to ~17m on the 60m one. MapDiffusion
    normalizes map coordinates about the ROI centre and cannot represent
    anything outside it, so this subtracts ``results['tile_shift']``
    (``tile_center - offset``, recorded per sample by the converter) to put
    the points in the tile-centred frame the GT is already in. The two are
    shifted by the same vector, so their relative alignment -- a median
    0.038m from GT vertices to real driving-surface returns -- is unchanged.

    Args:
        coord_type (str): Coordinate frame of the points. One of ``'LIDAR'``,
            ``'DEPTH'``, ``'CAMERA'``. Defaults to ``'LIDAR'``.
        load_dim (int): Number of columns produced before selection
            (``x, y, z, strength``). Defaults to 4.
        use_dim (int | list[int]): Which of those columns to keep. Defaults
            to 4 (all of them).
        z_max (float | None): Drop points with ``z`` greater than this, after
            recentring. Must stay >= the z upper bound of the voxelizer's
            ``point_cloud_range``. ``None`` disables it. Defaults to 96.0,
            which covers the town03 overpass tiles (z up to ~90m).
    """

    def __init__(self,
                 coord_type='LIDAR',
                 load_dim=4,
                 use_dim=4,
                 z_max=96.0):
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']
        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.z_max = z_max
        self._rgb2strength = np.array([0.2126, 0.7152, 0.0722],
                                      dtype=np.float32)

    def _load_points(self, pts_filename, tile_shift):
        mmcv.check_file_exist(pts_filename)
        with np.load(pts_filename) as block:
            features = np.asarray(block['features'], dtype=np.float32)
        coord = features[:, 0:3]
        if tile_shift is not None:
            coord = coord - np.asarray(tile_shift, dtype=np.float32)
        strength = (features[:, 3:6] @ self._rgb2strength).reshape([-1, 1])
        points = np.concatenate([coord, strength], axis=1)
        if self.z_max is not None:
            points = points[points[:, 2] <= self.z_max]
        return points

    def __call__(self, results):
        points = self._load_points(results['pts_filename'],
                                   results.get('tile_shift'))
        points = points[:, self.use_dim]

        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(
            points, points_dim=points.shape[-1], attribute_dims=None)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'coord_type={self.coord_type}, '
                f'load_dim={self.load_dim}, use_dim={self.use_dim}, '
                f'z_max={self.z_max})')


@PIPELINES.register_module(force=True)
class GridSamplePoints(object):
    """Keep one representative point per occupied ``grid_size`` cell, via
    integer coordinate packing plus a single 1D ``torch.unique`` (vectorized,
    no Python loop over points).

    Ported from the sibling MapTRv2 tree. Some CARLA tiles hold up to
    5,000,000 raw points -- an exporter artifact, an unbounded number of scan
    passes merged into one static block, unlike nuScenes/AV2's
    hardware-bounded ~10-sweep aggregation. Feeding that straight into the
    LiDAR voxelizer is both very slow and, confirmed there against a
    known-ground-truth synthetic cloud, wrong: the legacy ``Voxelization``
    CUDA kernel silently under-reports occupied voxels by ~36% at that scale
    (2000 known distinct cells, 5,000,000 points -> 1,280 reported). Grid
    sampling first collapses the redundancy *before* voxelization: ~26x
    faster, and it recovers 100% of the occupied voxels on the same
    worst-case tile, vs 8.6%-22.4% for random subsampling to a similar
    budget. Grid sampling is density-uniform rather than
    density-proportional, so it does not disproportionately thin the sparse
    regions -- divider lines -- that this task depends on.

    Args:
        grid_size (float | tuple[float, float, float]): cell size in metres.
            Defaults to (0.1, 0.1, 0.4), exactly the LiDAR ``voxel_size``, so
            this costs no spatial precision beyond what the model's own
            voxelizer already imposes.
        point_cloud_range (list[float]): must match the range passed to the
            LiDAR voxelizer downstream (``lidar_point_cloud_range`` in the
            config). Used to offset/bound the integer grid coordinates and to
            detect tiles with nothing in range -- not to filter points, which
            the voxelizer does itself.
        min_points (int): raise ``EmptyLidarTileError`` when fewer than this
            many points fall inside ``point_cloud_range``.
    """

    def __init__(self,
                 grid_size=(0.1, 0.1, 0.4),
                 point_cloud_range=None,
                 min_points=1):
        if point_cloud_range is None:
            raise ValueError(
                'GridSamplePoints needs point_cloud_range; pass the same '
                'lidar_point_cloud_range the voxelizer uses')
        if isinstance(grid_size, (int, float)):
            grid_size = (grid_size, grid_size, grid_size)
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.min_points = min_points
        self.dims = [
            int(round((point_cloud_range[3 + i] - point_cloud_range[i])
                      / grid_size[i])) + 1
            for i in range(3)
        ]

    def __call__(self, results):
        points = results['points']
        tensor = points.tensor
        xyz = tensor[:, :3]
        lo = xyz.new_tensor(self.point_cloud_range[:3])
        hi = xyz.new_tensor(self.point_cloud_range[3:])
        if tensor.shape[0] == 0:
            # torch.unique(...).max() below is undefined on an empty tensor,
            # so bail out before it: nothing to downsample either way.
            self._check_not_empty(results, 0, 0)
            return results
        # Counted before the clamp below, which would otherwise pull
        # out-of-range points into edge cells and hide the fact that the
        # voxelizer is about to drop every one of them.
        n_in_range = int(((xyz >= lo) & (xyz < hi)).all(1).sum())
        self._check_not_empty(results, tensor.shape[0], n_in_range)
        gsize = xyz.new_tensor(self.grid_size)
        gcoord = torch.floor((xyz - lo) / gsize).long()
        for i in range(3):
            gcoord[:, i].clamp_(0, self.dims[i] - 1)
        key = (gcoord[:, 0] * self.dims[1] + gcoord[:, 1]) * self.dims[2] \
            + gcoord[:, 2]

        _, inverse = torch.unique(key, return_inverse=True)
        order = torch.arange(tensor.shape[0], device=tensor.device)
        rep_idx = order.new_full((int(inverse.max()) + 1,), tensor.shape[0])
        rep_idx.scatter_reduce_(0, inverse, order, reduce='amin',
                                include_self=True)

        results['points'] = points[rep_idx]
        return results

    def _check_not_empty(self, results, n_raw, n_in_range):
        if self.min_points <= 0 or n_in_range >= self.min_points:
            return
        raise EmptyLidarTileError(
            f'tile {results.get("sample_idx")} has {n_in_range} of {n_raw} '
            f'point(s) inside point_cloud_range={self.point_cloud_range} '
            f'(need >= {self.min_points}); it would voxelize to zero voxels. '
            'Regenerate the annotation pkl with '
            'tools/data_converter/carla_converter.py to drop tiles like this '
            'up front, or widen the range.')

    def __repr__(self):
        return (f'{self.__class__.__name__}(grid_size={self.grid_size}, '
                f'point_cloud_range={self.point_cloud_range}, '
                f'min_points={self.min_points})')
