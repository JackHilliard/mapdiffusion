"""Smoke subset of mapdiffusion_carla3cls_lidar_50m_tile.py: 200 train / 30 val / 30 test tiles
(carla_splits_v1/smoke). Same benchmark, same model; own GT caches and work dir."""
_base_ = ['./mapdiffusion_carla3cls_lidar_50m_tile.py']

splits_root = 'data/carla_splits_v1/'
ann_file_train = splits_root + 'smoke/train200.json'
ann_file_val = splits_root + 'smoke/val30.json'
ann_file_test = splits_root + 'smoke/test30.json'
map_ann_file_val = splits_root + 'smoke/val30_gt_3cls_mapdiffusion.pkl'
map_ann_file_test = splits_root + 'smoke/test30_gt_3cls_mapdiffusion.pkl'

data = dict(
    train=dict(ann_file=ann_file_train),
    val=dict(ann_file=ann_file_val,
             eval_config=dict(ann_file=ann_file_val, map_ann_file=map_ann_file_val)),
    test=dict(ann_file=ann_file_test,
              eval_config=dict(ann_file=ann_file_test, map_ann_file=map_ann_file_test)))

# 200 tiles = 13 optimizer steps (52 iterations at the default 4 x 4) per epoch: keep the
# warmup inside the first epoch (as the HRDX smoke config)
num_train_tiles = 200
epoch_schedule = dict(warmup_samples=200)
lr_config = dict(warmup_iters=50)

# static values for 200 tiles at the default batch (recomputed by tools/train.py)
runner = dict(max_iters=16 * 52)
evaluation = dict(interval=2 * 52)
checkpoint_config = dict(interval=52)

work_dir = './work_dirs/mapdiffusion_carla3cls_lidar_50m_tile_smoke'
