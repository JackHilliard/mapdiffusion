"""Smoke subset of mapdiffusion_hrdx_lidar_50m_tile.py: 200 train / 30 val tiles
(hrdx_splits_v1/smoke). Same benchmark, same model; own GT cache and work dir."""
_base_ = ['./mapdiffusion_hrdx_lidar_50m_tile.py']

splits_root = 'data/hrdx_splits_v1/'
ann_file_train = splits_root + 'smoke/train200.json'
ann_file_val = splits_root + 'smoke/val30.json'
map_ann_file_val = splits_root + 'smoke/val30_gt_param_mapdiffusion.pkl'

data = dict(
    train=dict(ann_file=ann_file_train),
    val=dict(ann_file=ann_file_val,
             eval_config=dict(ann_file=ann_file_val, map_ann_file=map_ann_file_val)),
    test=dict(ann_file=ann_file_val,
              eval_config=dict(ann_file=ann_file_val, map_ann_file=map_ann_file_val)))

# 200 tiles = 13 optimizer steps (52 iterations at the default 4 x 4) per epoch: keep the
# warmup inside the first epoch (the reference smoke config's 100 iterations x batch 2)
epoch_schedule = dict(warmup_samples=200)
lr_config = dict(warmup_iters=50)

# static values for 200 tiles at the default batch (recomputed by tools/train.py)
runner = dict(max_iters=16 * 52)
evaluation = dict(interval=2 * 52)
checkpoint_config = dict(interval=52)

work_dir = './work_dirs/mapdiffusion_hrdx_lidar_50m_tile_smoke'
