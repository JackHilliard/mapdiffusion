# Smoke-test overlay: the v1_smoke split subset (400 train / 60 val tiles).
# Same model and crop kwargs as the parent; only the manifests change.
# val_60.json is a prefix of val.json, so its frozen crops are the first 60
# of the full split (the GT cache key includes the json name, so it never
# collides with the full split's). Val doubles as test here.
_base_ = ['./mapdiffusion_sanjose_lidar_50m_crop.py']

splits_root = './data/san_jose_splits_v1/v1_smoke'
ann_file_train = f'{splits_root}/train_400.json'
ann_file_val = f'{splits_root}/val_60.json'

data = dict(
    train=dict(ann_file=ann_file_train),
    val=dict(ann_file=ann_file_val, eval_config=dict(ann_file=ann_file_val)),
    test=dict(ann_file=ann_file_val, eval_config=dict(ann_file=ann_file_val)),
)
