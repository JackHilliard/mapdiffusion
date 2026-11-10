"""MapDiffusion (LiDAR only) on the HRDX whole-50 m-tile benchmark (San Jose + San Francisco +
San Mateo).

An overlay on mapdiffusion_sanjose_lidar_50m_crop.py: the model, losses, matching, diffusion
schedule, DDIM evaluation sampling, optimizer type and learning-rate rule are that config's;
the data, the ranges and the training schedule are the benchmark's.

THE BENCHMARK (identical in every sister baseline: MapTR, GeMap, PseudoMapTrainer,
mapdiffusion, SuperMapNet -- change it in all five or in none):

  data      data/hrdx_tiles_50m/<city>_tiles_50m_lines/{blocks,reference_lines}
            splits data/hrdx_splits_v1/{train,val,test}.json (20,499 / 2,227 / 3,392 tiles),
            smoke subsets data/hrdx_splits_v1/smoke/{train200,val30}.json
  sample    the WHOLE 50 m tile, +-25 m, no crop, no translation
  GT        one class ('boundary'); gt_source='parametric' (design-equation samples chained
            per alignment, hrdx_param_gt.py), clipped to the +-25 m box, pieces shorter than
            2 m dropped
  input     (x, y, z, strength): z re-zeroed on the tile's 2nd-percentile ground,
            strength = LiDAR intensity / 255; grid-sampled at 0.1 x 0.1 x 0.4 m; shuffled
  train aug Pointcept nuScenes PTv3: rotate z U(-pi, pi) p 0.5, scale U(0.9, 1.1), flip x and
            flip y each p 0.5 (one 2x2 map on points AND GT), jitter sigma 0.005 clip 0.02
            (points only); tiles with < 5000 in-range points or no GT are resampled
  val/test  never augmented, frozen; evaluated every 2 epochs on val, chamfer AP
  schedule  16 epochs, effective batch 16, cosine annealing with linear warmup,
            checkpoint every epoch (keep 4)

MAPDIFFUSION SPECIFICS (not benchmark):

  model     MapDiffusion / BEVFormerBackbone(modality='lidar') / MapDetectorHeadDiffuse,
            100 queries x 20 points, FocalLoss 5 + LinesL1Loss 50, HungarianLinesAssigner,
            1000-step cosine diffusion, ConvGRU streaming fusion -- all unchanged.
            Only what the data forces differs from the San Jose crop config:
            roi_size (50, 50); BEV 168 x 168 (the repo's native 0.3 m cell, 100 cells per 30 m,
            scaled to 50 m: 166.7 -> 168); sparse_shape [420, 500, 500]; SparseEncoder
            in_channels 4; voxelizer max_voxels [700000, 1000000] (a whole grid-sampled tile
            holds several 1e5 voxels, the 30 m value of 90000 would truncate it).
  optimizer AdamW, weight decay 1e-2, grad clip 35, lr 4e-4 = the repo's rule
            1e-4 * (batch / 4) at the effective batch 16. No fp16.
  batch     samples_per_gpu x GPUs x cumulative_iters = 16. tools/train.py derives
            cumulative_iters (GradientCumulativeOptimizerHook) and every iteration count from
            `epoch_schedule` below once the train set is built, because this repo's
            IterBasedRunner + infinite batch sampler count iterations, not epochs:
                iters/epoch = ceil(n_train / 16) * cumulative_iters
            Override from the command line with
                --cfg-options epoch_schedule.num_epochs=E data.samples_per_gpu=B \
                              data.workers_per_gpu=W
            (B x GPUs must divide 16; tools/train.py also sizes the streaming buffer to B).
            Default B = 4, accumulated 4 times. The runner / evaluation / checkpoint numbers
            written in this file are the values for the full train split at that default,
            used only by tools that do not go through tools/train.py. A run must be resumed
            with the batch size it was started with (the checkpoint counts iterations).
  warmup    8000 samples (the reference's 4000 iterations x batch 2), ratio 1/3,
            min_lr_ratio 1e-3 (the reference's schedule shape; the lr itself is this repo's).
  eval      this repo's VectorEvaluate: chamfer AP at 0.5 / 1.0 / 1.5 m on 200-point
            resamplings, predictions from 5 DDIM steps (eta 0.5, query threshold 0.5). The GT
            it scores against is the benchmark GT with every vertex kept
            (simplify_tolerance=0). The GT cache is a pickle keyed by tile name (not the
            mmdet siblings' json), hence its own file name; it is content-checked whenever it
            is loaded and written atomically.

Data layout (relative to the repo root):
    data/hrdx_tiles_50m/<city>_tiles_50m_lines/blocks/<name>.npz
    data/hrdx_tiles_50m/<city>_tiles_50m_lines/reference_lines/<name>_reference_lines.json
    data/hrdx_splits_v1/{train,val,test}.json            (directory must be writable: GT caches)

mmcv evaluates each config file in isolation before _base_ merging, so every model field
that depends on the geometry is restated in the `model` overlay below.
"""
_base_ = ['./mapdiffusion_sanjose_lidar_50m_crop.py']

# ---- data source (benchmark) -----------------------------------------------
dataset_type = 'Hrdx50mDataset'
data_root = 'data/hrdx_tiles_50m/'
splits_root = 'data/hrdx_splits_v1/'
ann_file_train = splits_root + 'train.json'
ann_file_val = splits_root + 'val.json'
ann_file_test = splits_root + 'test.json'
map_ann_file_val = splits_root + 'val_gt_param_mapdiffusion.pkl'
map_ann_file_test = splits_root + 'test_gt_param_mapdiffusion.pkl'

# ---- taxonomy (benchmark) --------------------------------------------------
cat2id = {
    'boundary': 0,
}
num_class = max(list(cat2id.values())) + 1
gt_source = 'parametric'

# ---- geometry (benchmark ranges; BEV grid is model specific) ---------------
tile_radius = 25.0
roi_size = (2 * tile_radius, 2 * tile_radius)
bev_h = 168
bev_w = 168
pc_range = [-tile_radius, -tile_radius, -30.0, tile_radius, tile_radius, 20.0]              # BEV / head range
lidar_point_cloud_range = [-tile_radius, -tile_radius, -72.0, tile_radius, tile_radius, 96.0]  # voxelisation range
lidar_voxel_size = [0.1, 0.1, 0.4]
lidar_grid = [
    int(round((lidar_point_cloud_range[3 + i] - lidar_point_cloud_range[i])
              / lidar_voxel_size[i])) for i in range(3)
]
sparse_shape = lidar_grid[::-1]          # (nz, ny, nx) = [420, 500, 500]
load_dim = 4
use_dim = 4                              # (x, y, z, strength)

coords_dim = 2
num_points = 20
permute = True
meta = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False,
    output_format='vector')

# ---- batch / schedule ------------------------------------------------------
# benchmark: 16 epochs, effective batch 16, eval every 2 epochs, checkpoint every epoch.
# samples_per_gpu x GPUs must divide effective_batch; the rest is accumulated.
# 4 per GPU x 4 accumulated iterations by default: a whole 50 m tile is 2.8x a 30 m crop
# (one tile alone needs > 2.8 GB of GPU memory in training), so the San Jose crop config's
# 16 per GPU is left to GPUs that can hold it (--cfg-options data.samples_per_gpu=16).
batch_size = 4
num_gpus = 1
num_train_tiles = 20499
epoch_schedule = dict(
    num_epochs=16,
    eval_every=2,
    ckpt_every=1,
    effective_batch=16,
    warmup_samples=8000)
# static values for the full split at the default batch (tools/train.py recomputes them
# from epoch_schedule, the real train-set size and the real batch)
cumulative_iters = epoch_schedule['effective_batch'] // (num_gpus * batch_size)
num_iters_per_epoch = -(-num_train_tiles // epoch_schedule['effective_batch']) * cumulative_iters

# ---- augmentation (benchmark, train only) ----------------------------------
hrdx_aug = dict(rotate_p=0.5, rotate_angle=(-1.0, 1.0), scale=(0.9, 1.1), flip_p=0.5,
                jitter_sigma=0.005, jitter_clip=0.02)

# ---- pipelines -------------------------------------------------------------
train_pipeline = [
    dict(
        type='VectorizeMap',
        coords_dim=coords_dim,
        roi_size=roi_size,
        sample_num=num_points,
        normalize=True,
        permute=permute,
    ),
    dict(type='LoadHrdx50mTile', coord_type='LIDAR', load_dim=load_dim, use_dim=use_dim),
    dict(type='GridSamplePoints',
         grid_size=lidar_voxel_size,
         point_cloud_range=lidar_point_cloud_range,
         min_points=5000),
    dict(type='PointShuffle'),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['points', 'vectors', 'gts'], meta_keys=(
        'token', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'scene_name'))
]

test_pipeline = [
    dict(type='LoadHrdx50mTile', coord_type='LIDAR', load_dim=load_dim, use_dim=use_dim),
    dict(type='GridSamplePoints',
         grid_size=lidar_voxel_size,
         point_cloud_range=lidar_point_cloud_range),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['points'], meta_keys=(
        'token', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'scene_name'))
]

# evaluation GT: the benchmark GT in metres, every vertex kept
eval_pipeline = [
    dict(
        type='VectorizeMap',
        coords_dim=coords_dim,
        simplify=True,
        simplify_tolerance=0.0,
        normalize=False,
        roi_size=roi_size
    ),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['vectors'], meta_keys=['token'])
]

hrdx_common = dict(
    type=dataset_type,
    data_root=data_root,
    meta=meta,
    roi_size=roi_size,
    cat2id=cat2id,
    gt_source=gt_source,
    min_polyline_len=2.0,
    seq_split_num=1,
)

# one evaluation GT per frozen split (same manifest, same gt_source as the dataset it scores)
eval_config_val = dict(
    ann_file=ann_file_val,
    map_ann_file=map_ann_file_val,
    pipeline=eval_pipeline,
    interval=1,
    crop_seed=0,
    **hrdx_common)
eval_config_test = dict(
    ann_file=ann_file_test,
    map_ann_file=map_ann_file_test,
    pipeline=eval_pipeline,
    interval=1,
    crop_seed=0,
    **hrdx_common)

# ---- model: the San Jose crop model, re-ranged to +-25 m, 4 input channels --
model = dict(
    roi_size=roi_size,
    bev_h=bev_h,
    bev_w=bev_w,
    backbone_cfg=dict(
        roi_size=roi_size,
        bev_h=bev_h,
        bev_w=bev_w,
        lidar_encoder=dict(
            voxelize=dict(
                point_cloud_range=lidar_point_cloud_range,
                voxel_size=lidar_voxel_size,
                max_voxels=[700000, 1000000]),  # [train, test]
            backbone=dict(
                in_channels=use_dim,
                sparse_shape=sparse_shape)),
        transformer=dict(encoder=dict(pc_range=pc_range)),
        positional_encoding=dict(row_num_embed=bev_h, col_num_embed=bev_w),
    ),
    head_cfg=dict(
        num_classes=num_class,
        roi_size=roi_size),
    streaming_cfg=dict(batch_size=batch_size),
)

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=8,
    train=dict(
        _delete_=True,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        aug=hrdx_aug,
        aug_tries=8,
        **hrdx_common),
    val=dict(
        _delete_=True,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        eval_config=eval_config_val,
        test_mode=True,
        crop_seed=0,
        **hrdx_common),
    test=dict(
        _delete_=True,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        eval_config=eval_config_test,
        test_mode=True,
        crop_seed=0,
        **hrdx_common),
    # every tile is its own one-sample scene: the sampler is a shuffled infinite stream of
    # tiles, ceil(n / batch) iterations of it are one epoch
    shuffler_sampler=dict(
        type='InfiniteGroupEachSampleInBatchSampler',
        seq_split_num=1,
        num_iters_to_seq=0,
        random_drop=0.0),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

# ---- optimisation / schedule -----------------------------------------------
optimizer = dict(
    type='AdamW',
    lr=1e-4 * (epoch_schedule['effective_batch'] / 4),      # 4e-4, tied to the EFFECTIVE batch
    weight_decay=1e-2)
# gradients are accumulated over cumulative_iters iterations (loss / cumulative_iters), so
# one optimizer step sees effective_batch tiles
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=cumulative_iters,
    grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=epoch_schedule['warmup_samples'] // (num_gpus * batch_size),
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

runner = dict(
    type='IterBasedRunner',
    max_iters=epoch_schedule['num_epochs'] * num_iters_per_epoch)

evaluation = dict(
    interval=epoch_schedule['eval_every'] * num_iters_per_epoch,
    eval_diffusion_eta=0.5,
    eval_diffusion_sampling_timesteps=5,
    eval_diffusion_query_threshold=0.5)

# IterBasedRunner writes iter_<n>.pth.  create_symlink=True (the repo's other configs use False,
# which leaves NO latest.pth at all) adds latest.pth, a relative symlink to the newest
# iter_<n>.pth -- the file the submit script resumes from and evaluates
checkpoint_config = dict(
    create_symlink=True,
    interval=epoch_schedule['ckpt_every'] * num_iters_per_epoch,
    max_keep_ckpts=4)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

work_dir = './work_dirs/mapdiffusion_hrdx_lidar_50m_tile'
