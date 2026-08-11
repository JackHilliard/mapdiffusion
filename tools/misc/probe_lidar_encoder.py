"""Probe the LiDAR encoder's geometry before wiring it into the model.

Answers the three questions the config cannot be written without, and that
must not be hand-derived:

  1. Which order does this install's `Voxelization` emit `coors` in, and
     which axis of `SparseEncoder`'s dense output is y vs x? This repo uses
     *stock* mmdet3d 1.0.0rc6 + stock mmcv ops, whose coors are
     `(batch, z, y, x)` -- the opposite of the vendored fork in the sibling
     GeMap/MapTR trees, whose CUDA kernel was patched to emit `(x, y, z)`
     and whose configs therefore carry `sparse_shape=[x, y, z]` plus a
     `permute(0, 1, 3, 2)`. CARLA's square tiles make a swapped x/y
     invisible in the shapes alone, so this is checked with an asymmetric
     synthetic blob rather than assumed.
  2. What `sparse_shape` and `encoder_paddings` actually produce, given
     that a stride-2 sparse conv's output size depends on the padding per
     axis and the odd padding sits on a different axis here than in the
     fork's configs.
  3. The real `C * D` channel count of the flattened dense output, which is
     `lidar_bev_proj.in_channels`.

Run inside the container, with the CARLA dataset mounted:

    docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \\
      -e PYTHONPATH=/workspace/mapdiffusion -w /workspace/mapdiffusion \\
      -v /path/to/carla:/workspace/mapdiffusion/datasets/carla \\
      mapdiffusion:latest python tools/misc/probe_lidar_encoder.py
"""
import argparse
import os.path as osp

import numpy as np
import torch
from mmdet3d.models.builder import build_middle_encoder
from mmdet3d.ops import Voxelization

# Must match plugin/configs/mapdiffusion_carla_lidar*.py. The xy half-extent
# is the export's tile_radius and is set from --tile-radius; z is fixed.
LIDAR_Z_RANGE = (-72.0, 96.0)
LIDAR_VOXEL_SIZE = [0.1, 0.1, 0.4]
Z_MAX = 96.0
LIDAR_POINT_CLOUD_RANGE = None  # set in main() from --tile-radius

# ITU-R BT.709 luma, matching LoadCarlaPointsFromFile's rgb->strength.
RGB2GRAY = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def grid_size(pc_range, voxel_size):
    """(nx, ny, nz), the same rounding Voxelization itself does."""
    lo = np.asarray(pc_range[:3], dtype=np.float64)
    hi = np.asarray(pc_range[3:], dtype=np.float64)
    return np.round((hi - lo) / np.asarray(voxel_size)).astype(int).tolist()


def build(sparse_shape, in_channels=4):
    voxelize = Voxelization(
        voxel_size=LIDAR_VOXEL_SIZE,
        point_cloud_range=LIDAR_POINT_CLOUD_RANGE,
        max_num_points=10,
        max_voxels=(90000, 120000))
    backbone = build_middle_encoder(
        dict(
            type='SparseEncoder',
            in_channels=in_channels,
            sparse_shape=sparse_shape,
            output_channels=128,
            order=('conv', 'norm', 'act'),
            encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128),
                              (128, 128)),
            # Odd padding on the z axis. Stock coors are (z, y, x), so z is
            # dim 0 here -- the sibling forks put it last because their
            # kernel emits (x, y, z).
            encoder_paddings=([0, 0, 1], [0, 0, 1], [0, 0, [0, 1, 1]], [0, 0]),
            block_type='basicblock'))
    return voxelize.cuda(), backbone.cuda().eval()


def run(voxelize, backbone, points):
    """points: list of (N, 4) cuda tensors. Returns the dense BEV feature."""
    feats, coords, sizes = [], [], []
    for k, res in enumerate(points):
        f, c, n = voxelize(res)
        feats.append(f)
        coords.append(torch.nn.functional.pad(c, (1, 0), value=k))
        sizes.append(n)
    feats = torch.cat(feats, dim=0)
    coords = torch.cat(coords, dim=0)
    sizes = torch.cat(sizes, dim=0)
    # Mean-reduce each voxel's points, as MapTRv2's voxelize() does.
    feats = (feats.sum(dim=1) / sizes.type_as(feats).view(-1, 1)).contiguous()
    with torch.no_grad():
        return backbone(feats, coords, len(points)), coords


def check_coord_order(voxelize, backbone, sparse_shape):
    """Put a dense blob at a known asymmetric (x, y) and see where it lands.

    x=+10, y=-5 is deliberately away from both the origin and the diagonal,
    so an x/y swap, a sign flip, or a transposed dense output each move the
    hit to a different cell.
    """
    print('\n--- coordinate order ---')
    n = 20000
    rng = np.random.default_rng(0)
    blob = np.zeros((n, 4), dtype=np.float32)
    blob[:, 0] = 10.0 + rng.normal(0, 0.2, n)   # x = +10  (near x_max)
    blob[:, 1] = -5.0 + rng.normal(0, 0.2, n)   # y = -5
    blob[:, 2] = rng.normal(0, 0.5, n)
    blob[:, 3] = 1.0
    pts = [torch.from_numpy(blob).cuda()]

    _, coords = run(voxelize, backbone, pts)
    # coords columns are (batch, a, b, c); report each axis' occupied span
    # in voxel units, which identifies which column is which axis.
    lo = np.asarray(LIDAR_POINT_CLOUD_RANGE[:3])
    vs = np.asarray(LIDAR_VOXEL_SIZE)
    expect = np.floor((np.array([10.0, -5.0, 0.0]) - lo) / vs).astype(int)
    got = coords[:, 1:].float().mean(0).cpu().numpy()
    print(f'  expected voxel idx (x, y, z) = {expect.tolist()}')
    print(f'  coors[:, 1:] column means    = {np.round(got, 1).tolist()}')
    order = 'UNKNOWN'
    if abs(got[0] - expect[2]) < 3 and abs(got[2] - expect[0]) < 3:
        order = '(batch, z, y, x)  <- stock mmdet3d/mmcv'
    elif abs(got[0] - expect[0]) < 3 and abs(got[2] - expect[2]) < 3:
        order = '(batch, x, y, z)  <- GeMap/MapTR vendored fork'
    print(f'  => coors order: {order}')

    feat, _ = run(voxelize, backbone, pts)
    energy = feat.abs().sum(1)[0]                      # (H, W)
    h, w = np.unravel_index(int(energy.argmax()), energy.shape)
    H, W = energy.shape
    # Metric centre of the winning cell, if H indexes y and W indexes x.
    y = LIDAR_POINT_CLOUD_RANGE[1] + (h + 0.5) / H * (
        LIDAR_POINT_CLOUD_RANGE[4] - LIDAR_POINT_CLOUD_RANGE[1])
    x = LIDAR_POINT_CLOUD_RANGE[0] + (w + 0.5) / W * (
        LIDAR_POINT_CLOUD_RANGE[3] - LIDAR_POINT_CLOUD_RANGE[0])
    print(f'  dense BEV {tuple(energy.shape)}, peak at (row={h}, col={w})')
    print(f'  reading rows as y and cols as x  -> (x={x:+.1f}, y={y:+.1f})')
    ok = abs(x - 10.0) < 2.0 and abs(y + 5.0) < 2.0
    print(f'  => dense output is (H=y, W=x): {ok}')
    if not ok:
        print('  !! rows/cols are NOT (y, x) -- do not trust anything below')

    # MapDiffusion's BEV convention is row 0 = y_max (StreamMapNet.py:63-72,
    # `y = linspace(ymax, ymin, bev_h)`), while SparseEncoder emits rows in
    # increasing y. Confirm the flip puts the blob where the mapper expects.
    flipped = torch.flip(energy, dims=[0])
    fh, _ = np.unravel_index(int(flipped.argmax()), flipped.shape)
    y_flipped = LIDAR_POINT_CLOUD_RANGE[4] - (fh + 0.5) / H * (
        LIDAR_POINT_CLOUD_RANGE[4] - LIDAR_POINT_CLOUD_RANGE[1])
    print(f'  after flip(dims=[H]): row={fh} -> y={y_flipped:+.1f} '
          f'(row 0 = y_max convention): {abs(y_flipped + 5.0) < 2.0}')
    return ok


def load_tile(path):
    """Same transform LoadCarlaPointsFromFile applies, minus the recentring."""
    with np.load(path) as block:
        features = np.asarray(block['features'], dtype=np.float32)
        offset = np.asarray(block['offset'], dtype=np.float32)
        tile_center = np.asarray(block['tile_center'], dtype=np.float32)
    xyz = features[:, :3] - (tile_center - offset)          # recentre, fact 1
    strength = (features[:, 3:6] @ RGB2GRAY).reshape(-1, 1)
    pts = np.concatenate([xyz, strength], axis=1)
    return pts[pts[:, 2] <= Z_MAX]


def grid_sample(points, pc_range, cell):
    """Cheap stand-in for GridSamplePoints: one point per occupied cell."""
    lo = np.asarray(pc_range[:3], dtype=np.float32)
    g = np.floor((points[:, :3] - lo) / np.asarray(cell, np.float32))
    _, keep = np.unique(g, axis=0, return_index=True)
    return points[np.sort(keep)]


def main():
    global LIDAR_POINT_CLOUD_RANGE
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='./datasets/carla')
    parser.add_argument('--split', default='train')
    parser.add_argument(
        '--tile-radius',
        type=float,
        default=12.5,
        help="half the export's tile side, i.e. the config's tile_radius "
        '(12.5 for the 25m export, 15.0 for a 30m one)')
    parser.add_argument(
        '--tiles',
        nargs='+',
        default=['town01_tile_00000', 'town03_tile_00100'],
        help='tile names to probe with real data')
    args = parser.parse_args()

    r = args.tile_radius
    LIDAR_POINT_CLOUD_RANGE = [-r, -r, LIDAR_Z_RANGE[0], r, r, LIDAR_Z_RANGE[1]]

    nx, ny, nz = grid_size(LIDAR_POINT_CLOUD_RANGE, LIDAR_VOXEL_SIZE)
    sparse_shape = [nz, ny, nx]        # stock order: (z, y, x)
    print(f'range {LIDAR_POINT_CLOUD_RANGE} @ voxel {LIDAR_VOXEL_SIZE}')
    print(f'grid (nx, ny, nz) = ({nx}, {ny}, {nz})')
    print(f'sparse_shape (stock [nz, ny, nx]) = {sparse_shape}')

    voxelize, backbone = build(sparse_shape)
    check_coord_order(voxelize, backbone, sparse_shape)

    print('\n--- real tiles ---')
    for name in args.tiles:
        path = osp.join(args.data_root, args.split, 'blocks', f'{name}.npz')
        if not osp.isfile(path):
            print(f'  {name}: missing ({path})')
            continue
        raw = load_tile(path)
        pts = grid_sample(raw, LIDAR_POINT_CLOUD_RANGE, LIDAR_VOXEL_SIZE)
        t = torch.from_numpy(pts).cuda()
        feat, coords = run(voxelize, backbone, [t])
        B, C, H, W = feat.shape
        print(f'  {name}: {raw.shape[0]:,} pts -> {pts.shape[0]:,} sampled '
              f'-> {coords.shape[0]:,} voxels -> {tuple(feat.shape)}')
        print(f'    => lidar_bev_proj in_channels=[{C}], BEV grid {H}x{W}')

    print('\nSet sparse_shape and lidar_bev_proj.in_channels from the above.')


if __name__ == '__main__':
    main()
