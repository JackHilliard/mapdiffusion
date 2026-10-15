# MapDiffusion LiDAR-only on 50 m CARLA tiles -> rotated 30 m crops, two
# classes (driving / curb).
#
# A thin overlay on mapdiffusion_carla_lidar.py, mirroring the sibling
# MapTRv2 tree's maptrv2_carla_r50_24ep_lidar_50m_crop.py so the two repos
# benchmark on the SAME crops of the SAME export. The MODEL is the 30 m one
# unchanged -- same roi_size (30, 30), bev 100x100, LiDAR range +-15,
# sparse_shape, colour-free SparseEncoder(in_channels=3) -- because every
# crop the network sees IS a 30 m tile. What changes is where those tiles
# come from:
#
# * The data are 50 m tiles on disk (tile_radius 25 m), read straight from
#   the export's <split>/manifest.json by Carla50mCropDataset -- there is no
#   converter pkl. Each __getitem__ draws a uniform yaw about the tile centre
#   plus a centre shift of up to 3.79 m (25 - 15*sqrt2, so every crop is
#   fully populated), crops a 30 m square, and adds sigma = 0.02 m Gaussian
#   noise to the points. GT polylines get the SAME pose and are clipped to
#   the same box with the tiler's own clip_polyline_to_box.
# * Val/test crops are frozen by crop_seed, not centred: VectorEvaluate
#   caches the eval GT once (tmp_gts_<gt_cache_stem>.pkl), so the crop each
#   tile is scored on must be identical on every run. A centred crop would
#   leave ~24% of the test tiles with no GT (the manifest was filtered for a
#   *reachable* non-empty crop); seeded aiming drops that to <1%. No yaw at
#   eval, so map quality is not conflated with rotation robustness.
# * The crop frame replaces the tile_center frame: the crop centre is the
#   origin by construction, so LoadCarla50mCrop does the reframing itself
#   and there is no tile_shift in these pipelines. Points' z is re-zeroed
#   per crop on its own 2nd-percentile ground.
# * Two classes, driving and curb, in that (label) order -- the export's own
#   names. num_class follows cat2id, so the head grows to 2 classes.
#
# Data layout expected under data_root (the export root, NOT ./data/carla):
#
#     <data_root>/<split>/manifest.json               tiles: [{name, center}]
#     <data_root>/<split>/blocks/<name>.npz           points (world),
#                                                     tile_center, tile_radius
#     <data_root>/<split>/reference_lines/<name>_reference_lines.json
#
# Verify the crop invariants (GT on LiDAR returns, full quadrant occupancy,
# uniform yaw, frozen val / fresh train) with
#     python tools/misc/verify_crop_dataset.py plugin/configs/mapdiffusion_carla_lidar_50m_crop.py --data-root <export>
#
# mmcv evaluates each config file in isolation before _base_ merging, so
# every base variable the dicts below need by name is restated here and must
# be kept equal to the base's.
_base_ = ['./mapdiffusion_carla_lidar.py']

# ---- data source -----------------------------------------------------------
data_root = './data/carla50m'
ann_file_train = f'{data_root}/train/manifest.json'
ann_file_test = f'{data_root}/test/manifest.json'

# ---- taxonomy --------------------------------------------------------------
cat2id = {
    'driving': 0,
    'curb': 1,
}
num_class = max(list(cat2id.values())) + 1

# ---- geometry (== base) ----------------------------------------------------
tile_radius = 15.0            # the CROP's half-size; the tiles on disk are 25
roi_size = (2 * tile_radius, 2 * tile_radius)
crop_size = roi_size[0]
lidar_point_cloud_range = [-tile_radius, -tile_radius, -72.0, tile_radius, tile_radius, 96.0]
lidar_voxel_size = [0.1, 0.1, 0.4]
coords_dim = 2
num_points = 20
permute = True
batch_size = 16
meta = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False,
    output_format='vector')

# ---- crop kwargs -----------------------------------------------------------
crop_common = dict(
    crop_size=crop_size,
    require_gt=True,
    gt_tries=32,
    min_polyline_len=2.0,
    # export class names, positionally paired with cat2id's label order
    gt_classes=('driving', 'curb'),
    reference_dir='reference_lines',
)
crop_train = dict(split='train', rotate=True, translate=True, noise_std=0.02,
                  **crop_common)
# Frozen by seed, no yaw, no noise -- see the header.
crop_eval = dict(split='test', rotate=False, translate=True, noise_std=0.0,
                 crop_seed=0, **crop_common)

# ---- pipelines -------------------------------------------------------------
# LoadCarla50mCrop replaces LoadCarlaPointsFromFile: it reads the world-frame
# points, applies the pose drawn by the dataset and crops. load_dim/use_dim=3
# is the colour-free convention and must match SparseEncoder.in_channels=3
# in the base.
lidar_pipeline = [
    dict(type='LoadCarla50mCrop', coord_type='LIDAR', load_dim=3, use_dim=3),
    dict(type='GridSamplePoints',
         grid_size=lidar_voxel_size,
         point_cloud_range=lidar_point_cloud_range),
]

train_pipeline = [
    dict(
        type='VectorizeMap',
        coords_dim=coords_dim,
        roi_size=roi_size,
        sample_num=num_points,
        normalize=True,
        permute=permute,
    ),
    *lidar_pipeline,
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['points', 'vectors', 'gts'], meta_keys=(
        'token', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'scene_name'))
]

test_pipeline = [
    *lidar_pipeline,
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['points'], meta_keys=(
        'token', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'scene_name'))
]

# ---- evaluation GT ---------------------------------------------------------
# Same frozen crops as data.val/test (same split, seed and aiming), so the
# GT the evaluator collects is the GT the model was shown.
eval_config = dict(
    type='Carla50mCropDataset',
    data_root=data_root,
    ann_file=ann_file_test,
    meta=meta,
    roi_size=roi_size,
    cat2id=cat2id,
    pipeline=[
        dict(
            type='VectorizeMap',
            coords_dim=coords_dim,
            simplify=True,
            normalize=False,
            roi_size=roi_size
        ),
        dict(type='FormatBundleMap'),
        dict(type='Collect3D', keys=['vectors'], meta_keys=['token'])
    ],
    interval=1,
    **crop_eval,
)

model = dict(
    head_cfg=dict(num_classes=num_class),
)

data = dict(
    samples_per_gpu=batch_size,
    train=dict(
        _delete_=True,
        type='Carla50mCropDataset',
        data_root=data_root,
        ann_file=ann_file_train,
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=train_pipeline,
        seq_split_num=1,
        **crop_train,
    ),
    val=dict(
        _delete_=True,
        type='Carla50mCropDataset',
        data_root=data_root,
        ann_file=ann_file_test,
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config,
        test_mode=True,
        seq_split_num=1,
        **crop_eval,
    ),
    test=dict(
        _delete_=True,
        type='Carla50mCropDataset',
        data_root=data_root,
        ann_file=ann_file_test,
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config,
        test_mode=True,
        seq_split_num=1,
        **crop_eval,
    ),
)
