# MapDiffusion LiDAR-only on HRDX San Jose 50 m tiles -> rotated 30 m crops,
# one class (curb).
#
# A thin overlay on mapdiffusion_carla_lidar_50m_crop.py, mirroring the
# sibling MapTRv2 tree's maptrv2_sanjose_r50_24ep_lidar_50m_crop.py so the
# repos benchmark on the SAME frozen crops of the SAME tiles. The MODEL and
# the crop recipe are the CARLA 50 m ones unchanged (roi_size (30, 30), bev
# 100x100, LiDAR range +-15, colour-free SparseEncoder(in_channels=3); 30 m
# crops with uniform yaw + <= 3.79 m shift + sigma 0.02 m noise at train,
# frozen unrotated aimed crops at eval). What changes is the data:
#
# * The export is san_jose_tiles_50m (HRDX San Jose, 17723 tiles of real
#   LiDAR, tile_radius 25, world-frame points). Its layout is FLAT -- blocks/
#   and reference_lines/ sit directly under the root, no <split>/ level --
#   and the split manifests live in a separate directory,
#   san_jose_splits_v1/v1/{train,val,test}.json (12073 / 1312 / 1978 tiles;
#   a leakage-free 750 m-block spatial split). SanJose50mCropDataset reads
#   both, in MANIFEST order (the CARLA class sorts by name, which would
#   re-index the frozen seeds -- see its docstring).
# * One class, curb. The export's taxonomy is {1: curb, 2: curb_parametric};
#   the parametric class re-traces the same curbs, so only curb is trained.
#   num_class follows cat2id, so the head is 1 class.
# * val != test here, so each frozen split gets its own eval_config (same
#   split, seed and aiming as the dataset it scores) and its own
#   tmp_gts_<gt_cache_stem>.pkl.
#
# Data layout (relative to the repo root):
#     data/san_jose_50m/blocks/<name>.npz
#     data/san_jose_50m/reference_lines/<name>_reference_lines.json
#     data/san_jose_splits_v1/v1/{train,val,test}.json
#
# Verify the crop invariants with
#     python tools/misc/verify_crop_dataset.py plugin/configs/mapdiffusion_sanjose_lidar_50m_crop.py
#
# mmcv evaluates each config file in isolation before _base_ merging, so
# every base variable the dicts below need by name is restated here and must
# be kept equal to the base's.
_base_ = ['./mapdiffusion_carla_lidar_50m_crop.py']

# ---- data source -----------------------------------------------------------
data_root = './data/san_jose_50m'
splits_root = './data/san_jose_splits_v1/v1'
ann_file_train = f'{splits_root}/train.json'
ann_file_val = f'{splits_root}/val.json'
ann_file_test = f'{splits_root}/test.json'

# ---- taxonomy --------------------------------------------------------------
cat2id = {
    'curb': 0,
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
    split=None,               # flat export: no <split>/ level
    crop_size=crop_size,
    require_gt=True,
    gt_tries=32,
    min_polyline_len=2.0,
    gt_classes=('curb',),
    reference_dir='reference_lines',
)
crop_train = dict(rotate=True, translate=True, noise_std=0.02, **crop_common)
# Frozen by seed, no yaw, no noise; crop_seed=0 is what the splits were
# finalised with.
crop_eval = dict(rotate=False, translate=True, noise_std=0.0, crop_seed=0,
                 **crop_common)

# ---- pipelines (== base) ---------------------------------------------------
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

# ---- evaluation GT: one per frozen split -----------------------------------
eval_pipeline = [
    dict(
        type='VectorizeMap',
        coords_dim=coords_dim,
        simplify=True,
        normalize=False,
        roi_size=roi_size
    ),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['vectors'], meta_keys=['token'])
]


def _eval_config(ann_file):
    return dict(
        type='SanJose50mCropDataset',
        data_root=data_root,
        ann_file=ann_file,
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=eval_pipeline,
        interval=1,
        **crop_eval,
    )


eval_config_val = _eval_config(ann_file_val)
eval_config_test = _eval_config(ann_file_test)

model = dict(
    head_cfg=dict(num_classes=num_class),
)

data = dict(
    samples_per_gpu=batch_size,
    train=dict(
        _delete_=True,
        type='SanJose50mCropDataset',
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
        type='SanJose50mCropDataset',
        data_root=data_root,
        ann_file=ann_file_val,
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config_val,
        test_mode=True,
        seq_split_num=1,
        **crop_eval,
    ),
    test=dict(
        _delete_=True,
        type='SanJose50mCropDataset',
        data_root=data_root,
        ann_file=ann_file_test,
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config_test,
        test_mode=True,
        seq_split_num=1,
        **crop_eval,
    ),
)
