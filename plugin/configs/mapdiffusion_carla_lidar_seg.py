# MapDiffusion, LiDAR-only, CARLA 30m tiles, PLUS an auxiliary BEV
# segmentation loss.
#
# Identical to mapdiffusion_carla_lidar.py except for the aux head: the same
# tile_radius, the same LiDAR encoder, the same detection head and diffusion
# schedule. Diff the two files to see exactly what the aux loss costs.
#
# Why: LiDAR-only MapDiffusion trains the SparseEncoder and the 7.4M-param
# ConvFuser from scratch, and its only gradient into them arrives through
# sparse polyline matching, through a denoising decoder. Measured here, that
# is thin enough that 6000 iterations over 9 tiles reach a training reg loss
# of 1.19 (~0.6m mean error) while AP@1.5 on those same tiles stays at
# 0.0152 -- the head fits the noised-GT training task without the BEV ever
# becoming informative. Every sibling CARLA LiDAR config (MapTRv2, GeMap,
# PseudoMapTrainer) enables aux_seg with bev_seg=True; this is the
# equivalent, supervising the BEV densely and directly.
#
# Derived from plugin/configs/mapdiffusion.py. Everything downstream of the
# BEV feature -- the MapDiffusion mapper, MapDetectorHeadDiffuse, the
# diffusion schedule and DDIM eval, streaming fusion, losses, assigner -- is
# unchanged; only the encoder, the dataset and the geometry the data forces
# differ. The LiDAR branch itself (voxelize -> SparseEncoder -> ConvFuser)
# matches the sibling GeMap config
# projects/configs/carla/gemap_carla_r50_24ep_lidar.py.
_base_ = [
    './_base_/default_runtime.py'
]

# model type
type = 'Mapper'
plugin = True

# plugin code dir
plugin_dir = 'plugin/'

num_gpus = 1
batch_size = 16
# 4103 CARLA train tiles (was 27846 nuScenes samples)
num_iters_per_epoch = 26000 // (num_gpus * batch_size)
num_epochs = 24
num_epochs_single_frame = num_epochs // 24
total_iters = num_epochs * num_iters_per_epoch
num_queries = 100

# diffusion
scheduler = 'cosine'
total_steps = 1000

# category configs
# CARLA's reference lines are a single class; the exporter's `gt_source` is
# driving_lanes, with no ped_crossing/boundary equivalent.
cat2id = {
    'divider': 0,
}
num_class = max(list(cat2id.values())) + 1

# bev configs
# Square, following the export's own tile size, and the ONLY line that needs
# changing to retarget this config at a differently-sized export --
# everything geometric below is derived from it, including the LiDAR range
# and the encoder's sparse_shape. CarlaDataset asserts it against the
# `tile_radius` recorded in the annotation pkl, so a mismatched pkl fails
# loudly rather than silently rescaling every map element.
# See mapdiffusion_carla_lidar_25m.py for the 25m export (tile_radius 12.5).
tile_radius = 15.0
roi_size = (2 * tile_radius, 2 * tile_radius) # bev range, one square tile
bev_h = 100
bev_w = 100
pc_range = [-roi_size[0]/2, -roi_size[1]/2, -30.0, roi_size[0]/2, roi_size[1]/2, 20.0]

# LiDAR branch geometry, kept separate from the map `pc_range` above: this is
# where the *points* live, which is a much taller volume than where map
# elements live. The z span is verbatim from the GeMap/MapTRv2 CARLA configs
# and is not arbitrary -- town03 has overpass tiles with returns spanning
# z in [-66.9, 90.5] within a single tile, and a narrower range drops every
# point in them, which voxelizes to zero voxels and crashes the encoder.
# `z_max` below must stay >= this range's z upper bound.
lidar_point_cloud_range = [-tile_radius, -tile_radius, -72.0, tile_radius, tile_radius, 96.0]
lidar_voxel_size = [0.1, 0.1, 0.4]
lidar_z_max = 96.0

# Voxel grid, reversed to the (nz, ny, nx) that SparseEncoder wants because
# stock mmdet3d/mmcv emit voxel coords as (batch, z, y, x). Derived rather
# than written out: it changes with tile_radius, and hardcoding it is silent
# -- the voxelizer happily produces x/y indices past a too-small
# sparse_shape. tile_radius 12.5 -> [420, 250, 250], 15.0 -> [420, 300, 300],
# both confirmed with tools/misc/probe_lidar_encoder.py --tile-radius.
lidar_grid = [
    int(round((lidar_point_cloud_range[3 + i] - lidar_point_cloud_range[i])
              / lidar_voxel_size[i])) for i in range(3)
]
sparse_shape = lidar_grid[::-1]

# vectorize params
coords_dim = 2
sample_dist = -1
sample_num = -1
simplify = True

# meta info for submission pkl
meta = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False,
    output_format='vector')

# model configs
bev_embed_dims = 256
embed_dims = 512
num_feat_levels = 3
norm_cfg = dict(type='BN2d')
num_class = max(list(cat2id.values()))+1
num_points = 20
permute = True

model = dict(
    type='MapDiffusion',
    roi_size=roi_size,
    bev_h=bev_h,
    bev_w=bev_w,
    backbone_cfg=dict(
        type='BEVFormerBackbone',
        modality='lidar',
        roi_size=roi_size,
        bev_h=bev_h,
        bev_w=bev_w,
        use_grid_mask=True,  # inert: no images
        # No img_backbone / img_neck at all. The LiDAR BEV replaces the
        # camera BEV rather than fusing with it.
        lidar_encoder=dict(
            voxelize=dict(
                max_num_points=10,
                point_cloud_range=lidar_point_cloud_range,
                voxel_size=lidar_voxel_size,
                max_voxels=[90000, 120000]),  # [train, test]
            backbone=dict(
                type='SparseEncoder',
                # xyz only. CARLA points also carry a "strength" channel
                # (BT.709 luma of the per-point RGB), but it is dropped via
                # use_dim=3 in lidar_pipeline below to match the MapTRv2 30m
                # HM benchmark convention, which trains colour-free. This
                # value and that use_dim MUST move together -- a mismatch
                # fails at the first sparse conv. sparse_shape and
                # lidar_bev_proj do not depend on the input channel width.
                in_channels=3,
                # (nz, ny, nx), derived above. This order, and the
                # encoder_paddings below putting the odd padding on the z
                # axis, follow from stock mmdet3d/mmcv emitting voxel coords
                # as (batch, z, y, x) -- the vendored fork in the sibling
                # GeMap/MapTRv2 trees is transposed and their configs read
                # [nx, ny, nz].
                sparse_shape=sparse_shape,
                output_channels=128,
                order=('conv', 'norm', 'act'),
                encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128),
                                  (128, 128)),
                encoder_paddings=([0, 0, 1], [0, 0, 1], [0, 0, [0, 1, 1]],
                                  [0, 0]),
                block_type='basicblock')),
        # SparseEncoder emits a dense (B, C*D, H, W) BEV tensor -- z folded
        # into the channel axis -- so this only channel-projects it to
        # embed_dims. 3200 = output_channels 128 * the 25 z-slices left after
        # the encoder's downsampling. Unlike sparse_shape this does NOT move
        # with tile_radius: only H/W do (32x32 at 12.5, 38x38 at 15.0), since
        # D follows the z range alone. Re-measure with
        # tools/misc/probe_lidar_encoder.py if the z range or voxel size
        # changes; do not hand-derive it.
        lidar_bev_proj=dict(
            type='ConvFuser',
            in_channels=[3200],
            out_channels=bev_embed_dims),
        # Structurally required (BEVFormerBackbone builds it unconditionally)
        # but never invoked on the LiDAR path: forward() returns before the
        # encoder, bev_embedding and positional_encoding are reached. Kept
        # identical to mapdiffusion.py so the two configs stay comparable;
        # find_unused_parameters=True below covers the gradient-free params.
        transformer=dict(
            type='PerceptionTransformer',
            embed_dims=bev_embed_dims,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=1,
                pc_range=pc_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=bev_embed_dims,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=bev_embed_dims,
                                num_points=8,
                                num_levels=num_feat_levels),
                            embed_dims=bev_embed_dims,
                        )
                    ],
                    feedforward_channels=bev_embed_dims*2,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                    'ffn', 'norm')
                )
            ),
        ),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=bev_embed_dims//2,
            row_num_embed=bev_h,
            col_num_embed=bev_w,
            ),
    ),
    head_cfg=dict(
        type='MapDetectorHeadDiffuse',
        num_queries=num_queries,
        embed_dims=embed_dims,
        num_classes=num_class,
        in_channels=embed_dims//2,
        num_points=num_points,
        roi_size=roi_size,
        coord_dim=2,
        different_heads=False,
        predict_refine=False,
        sync_cls_avg_factor=True,
        streaming_cfg=None,
        transformer=dict(
            type='MapTransformer',
            num_feature_levels=1,
            num_points=num_points,
            coord_dim=2,
            encoder=dict(
                type='PlaceHolderEncoder',
                embed_dims=embed_dims,
            ),
            decoder=dict(
                type='MapTransformerDecoderDiffuse',
                num_layers=6,
                timestep_embed = 128*4,
                prop_add_stage=1,
                return_intermediate=True,
                transformerlayers=dict(
                    type='MapTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=embed_dims,
                            num_heads=8,
                            attn_drop=0.1,
                            proj_drop=0.1,
                        ),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=embed_dims,
                            num_heads=8,
                            num_levels=1,
                            num_points=num_points,
                            dropout=0.1,
                        ),
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=embed_dims,
                        feedforward_channels=embed_dims*2,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True),        
                    ),
                    feedforward_channels=embed_dims*2,
                    ffn_dropout=0.1,
                    # operation_order=('norm', 'self_attn', 'norm', 'cross_attn',
                    #                 'norm', 'ffn',)
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                    'ffn', 'norm')
                )
            )
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=5.0
        ),
        loss_reg=dict(
            type='LinesL1Loss',
            loss_weight=50.0,
            beta=0.01,
        ),
        assigner=dict(
            type='HungarianLinesAssigner',
                cost=dict(
                    type='MapQueriesCost',
                    cls_cost=dict(type='FocalLossCost', weight=5.0),
                    reg_cost=dict(type='LinesL1Cost', weight=50.0, beta=0.01, permute=permute),
                    ),
                ),
        ),
    # Auxiliary BEV segmentation. Applied to the same tensor the detection
    # head reads, so it supervises the BEV after streaming fusion.
    # pos_weight counters how few BEV cells a 3px-thick polyline covers --
    # at 100x100 with ~4 lines per tile the positives are well under 5% of
    # the map. 4.0 is MapTRv2's value for bev_seg.
    aux_seg_cfg=dict(
        type='BEVSegHead',
        in_channels=bev_embed_dims,
        num_classes=num_class,
        pos_weight=4.0,
        loss_weight=1.0),
    streaming_cfg=dict(
        streaming_bev=True,
        batch_size=batch_size,
        fusion_cfg=dict(
            type='ConvGRU',
            out_channels=bev_embed_dims,
        )
    ),
    model_name='SingleStage'
)

# data processing pipelines
# No image transforms, and no ego2img/img_shape meta keys -- those are read
# only by the camera BEV encoder's point_sampling, which never runs here.
lidar_pipeline = [
    dict(type='LoadCarlaPointsFromFile',
         coord_type='LIDAR',
         # load_dim stays 4: the loader builds the strength column before
         # selecting; use_dim=3 keeps only [x, y, z] -- see the
         # in_channels=3 note on the SparseEncoder above.
         load_dim=4,
         use_dim=3,
         z_max=lidar_z_max,
         ),
    # Collapses each tile's raw points to one per voxel cell before the
    # voxelizer sees them. Not an optimization only: some tiles hold
    # 5,000,000 points, at which scale the voxelizer silently under-reports
    # occupied voxels by ~36%. Must use the same range as the voxelizer.
    dict(type='GridSamplePoints',
         grid_size=lidar_voxel_size,
         point_cloud_range=lidar_point_cloud_range,
         ),
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
    # The aux head's target. canvas_size is (w, h), so it is (bev_w, bev_h)
    # -- BEVSegHead asserts the mask and the BEV agree, which is what catches
    # this being given the other way round. thickness is in BEV cells: at
    # 30m/100 cells one cell is 0.3m, so 3 cells is a ~0.9m wide line.
    dict(
        type='RasterizeMap',
        roi_size=roi_size,
        canvas_size=(bev_w, bev_h),
        thickness=3,
        coords_dim=coords_dim,
    ),
    *lidar_pipeline,
    dict(type='FormatBundleMap'),
    # gts are added to train diffusion model
    dict(type='Collect3D', keys=['points', 'vectors', 'gts', 'semantic_mask'],
         meta_keys=(
        'token', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'scene_name'))
]

# data processing pipelines
test_pipeline = [
    *lidar_pipeline,
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['points'], meta_keys=(
        'token', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'scene_name'))
]

# Where the CARLA tiles live (blocks/ + reference_lines/ under <split>/),
# and where carla_converter.py wrote the annotation pkls. Kept apart so the
# tile export can stay read-only.
data_root = './data/carla'
ann_root = './data/carla_infos'

# configs for evaluation code
# DO NOT CHANGE
eval_config = dict(
    type='CarlaDataset',
    data_root=data_root,
    ann_file=f'{ann_root}/carla_map_infos_test.pkl',
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
)

# dataset configs
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type='CarlaDataset',
        data_root=data_root,
        ann_file=f'{ann_root}/carla_map_infos_train.pkl',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=train_pipeline,
        seq_split_num=1,
    ),
    val=dict(
        type='CarlaDataset',
        data_root=data_root,
        ann_file=f'{ann_root}/carla_map_infos_test.pkl',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config,
        test_mode=True,
        seq_split_num=1,
    ),
    test=dict(
        type='CarlaDataset',
        data_root=data_root,
        ann_file=f'{ann_root}/carla_map_infos_test.pkl',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config,
        test_mode=True,
        seq_split_num=1,
    ),
    shuffler_sampler=dict(
        type='InfiniteGroupEachSampleInBatchSampler',
        # Every CARLA tile is its own scene, so every group holds exactly one
        # sample and there is nothing to split: seq_split_num=2 would compute
        # a sub-sequence length of round(1/2)=0 and raise from range().
        seq_split_num=1,
        num_iters_to_seq=num_epochs_single_frame*num_iters_per_epoch,
        random_drop=0.0
    ),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

# optimizer
# No `img_backbone` paramwise key: there is no image backbone to hold at a
# lower LR, and the LiDAR encoder is trained from scratch like the rest.
optimizer = dict(
    type='AdamW',
    lr=1e-4 * num_gpus * (batch_size / 4),
    weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy & schedule
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=3e-3)

evaluation = dict(
    interval=num_epochs*num_iters_per_epoch,
    eval_diffusion_eta = 0.5,
    eval_diffusion_sampling_timesteps = 5,
    eval_diffusion_query_threshold = 0.5)

find_unused_parameters = True #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(create_symlink=False, interval=num_iters_per_epoch)

runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

SyncBN = True
