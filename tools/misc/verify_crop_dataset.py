"""Verify the 50 m -> 30 m crop dataset: alignment, coverage, determinism.

Port of the sibling MapTRv2 tree's tools/carla50m/verify_crop_dataset.py.
Builds the train and val datasets straight from a config (so it checks the
crop kwargs the training job will actually use), pulls crops through
`LoadCarla50mCrop`, and reports the invariants that matter:

  * GT vertices sit ON LiDAR returns (points and GT received the same pose);
  * every crop is fully populated and everything is inside the +-half box;
  * the train yaw is uniform;
  * the val crop is identical across calls (VectorEvaluate's cached GT
    depends on it) while the train crop is not;
  * how many frozen eval crops carry no GT at all.

Run inside the container (CPU is enough)::

    PYTHONPATH=/workspace/mapdiffusion python3 tools/misc/verify_crop_dataset.py \
        plugin/configs/mapdiffusion_carla_lidar_50m_crop.py \
        [--data-root <export root>] [--n 40] [--train-split test]
"""
import argparse
import warnings

import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset

import plugin  # noqa: F401  (registers the dataset and pipelines)
from plugin.datasets.pipelines.loading import LoadCarla50mCrop


def build(cfg, key, data_root, split=None):
    d = dict(cfg.data[key])
    d['pipeline'] = []
    d.pop('eval_config', None)
    d['test_mode'] = False
    if split:
        d['split'] = split
    if data_root or split:
        data_root = data_root or d['data_root']
        d['data_root'] = data_root
        d['ann_file'] = f"{data_root.rstrip('/')}/{d['split']}/manifest.json"
    return build_dataset(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--data-root', default=None,
                    help='override the config\'s data_root (export root)')
    ap.add_argument('--n', type=int, default=40, help='crops to sample')
    ap.add_argument('--train-split', default=None,
                    help='split to draw the TRAIN-mode crops from (default: '
                    "the config's; use `test` on a test-only export)")
    args = ap.parse_args()
    warnings.filterwarnings('ignore')
    cfg = Config.fromfile(args.config)
    cat2id = dict(cfg.data.train.cat2id)
    labels = sorted(cat2id.values())

    print('=' * 78 + '\n1. BUILD\n' + '=' * 78)
    tr = build(cfg, 'train', args.data_root, args.train_split)
    te = build(cfg, 'val', args.data_root)
    print(f'  train: {len(tr)} samples   val: {len(te)} samples   '
          f'classes: {cat2id}')
    print(f'  train max_shift={tr.max_shift:.3f} m  need={tr._need:.3f} m')
    print(f'  val   max_shift={te.max_shift:.3f} m  need={te._need:.3f} m')
    print(f'  val gt cache stem: {te.gt_cache_stem}')
    assert 'crop_R' in te.samples[0], 'val crops must be frozen'
    assert 'crop_R' not in tr.samples[0], 'train crops must be fresh'

    load_cfg = dict(cfg.lidar_pipeline[0])
    load_cfg.pop('type')
    loader = LoadCarla50mCrop(**load_cfg)
    half = tr.half
    # the point noise is added AFTER the crop, so allow a few sigma of slack
    slack = 4.0 * float(cfg.data.train.noise_std) + 1e-3
    n = min(args.n, len(tr))
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(tr), n, replace=False)

    print('\n' + '=' * 78 + '\n2. POINT/GT ALIGNMENT\n' + '=' * 78)
    d_all, occ, npts, ngt, bad_p, bad_g, dims = [], [], [], [], 0, 0, set()
    for i in idxs:
        res = tr.get_sample(int(i))
        loader(res)
        P = res['points'].tensor.numpy()
        dims.add(P.shape[1])
        npts.append(len(P))
        G = [np.asarray(g.coords) for lbl in labels
             for g in res['map_geoms'][lbl]]
        ngt.append(len(G))
        if len(P) and np.abs(P[:, :2]).max() > half + slack:
            bad_p += 1
        bad_g += sum(np.abs(g[:, :2]).max() > half + 1e-3 for g in G)
        if len(P) == 0 or not G:
            continue
        V = np.concatenate(G, 0)
        d_all.append(np.linalg.norm(V[:, None, :2] - P[None, :, :2],
                                    axis=-1).min(1))
        q = [((P[:, 0] > 0) & (P[:, 1] > 0)), ((P[:, 0] <= 0) & (P[:, 1] > 0)),
             ((P[:, 0] > 0) & (P[:, 1] <= 0)),
             ((P[:, 0] <= 0) & (P[:, 1] <= 0))]
        occ.append(sum(1 for m in q if m.sum() > 0))
    d = np.concatenate(d_all)
    print(f'  point dims: {sorted(dims)} (3 = colour-free convention)')
    print(f'  GT vertex -> nearest LiDAR point over {n} train crops: '
          f'median {np.median(d):.3f} m | p90 {np.percentile(d, 90):.3f} | '
          f'max {d.max():.3f}')
    print(f'  VERDICT: {"ALIGNED" if np.median(d) < 0.5 else "MISALIGNED"}')
    print(f'  all 4 quadrants populated: {sum(o == 4 for o in occ)}/{len(occ)}')
    print(f'  points/crop: median {int(np.median(npts)):,} '
          f'min {min(npts):,} max {max(npts):,}')
    print(f'  GT lines/crop: median {np.median(ngt):.1f} min {min(ngt)} '
          f'max {max(ngt)} empty: {sum(g == 0 for g in ngt)}')
    print(f'  crops with points outside +-{half} m (+noise slack): {bad_p}/{n}; '
          f'GT polylines outside: {bad_g}')

    print('\n' + '=' * 78 + '\n3. ROTATION\n' + '=' * 78)
    m = min(300, len(tr))
    Rs = [tr.get_sample(int(i))['crop_R']
          for i in rng.choice(len(tr), m, replace=True)]
    ang = np.array([np.arctan2(R[1, 0], R[0, 0]) for R in Rs]) % (2 * np.pi)
    h, _ = np.histogram(ang, bins=8, range=(0, 2 * np.pi))
    print(f'  yaw histogram over {m} draws (8 bins): {h.tolist()}')

    print('\n' + '=' * 78 + '\n4. EVAL DETERMINISM\n' + '=' * 78)
    k = min(5, len(te))
    a = [te.get_sample(i) for i in range(k)]
    b = [te.get_sample(i) for i in range(k)]
    same = all(
        np.array_equal(x['crop_R'], y['crop_R']) and all(
            len(x['map_geoms'][c]) == len(y['map_geoms'][c]) and all(
                np.array_equal(np.asarray(u.coords), np.asarray(v.coords))
                for u, v in zip(x['map_geoms'][c], y['map_geoms'][c]))
            for c in labels) for x, y in zip(a, b))
    print(f'  val get_sample identical across calls: {same}')
    print(f'  val yaw is identity: {np.allclose(a[0]["crop_R"], np.eye(2))}')
    ta, tb = tr.get_sample(0)['crop_R'], tr.get_sample(0)['crop_R']
    print(f'  train crop varies across calls: {not np.array_equal(ta, tb)}')
    empty = sum(
        1 for i in range(len(te))
        if not any(te.get_sample(i)['map_geoms'][c] for c in labels))
    print(f'  frozen val crops with no GT: {empty}/{len(te)}')
    assert same and not np.array_equal(ta, tb)
    print('\nOK')


if __name__ == '__main__':
    main()
