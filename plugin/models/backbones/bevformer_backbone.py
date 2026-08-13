import copy
import imp
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import BACKBONES
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import mmcv
import cv2 as cv
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from .bevformer.grid_mask import GridMask
from mmdet3d.models import builder
from mmdet3d.ops import Voxelization, DynamicScatter


class UpsampleBlock(nn.Module):
    def __init__(self, ins, outs):
        super(UpsampleBlock, self).__init__()
        self.gn = nn.GroupNorm(32, outs)
        self.conv = nn.Conv2d(ins, outs, kernel_size=3,
                              stride=1, padding=1)  # same
        self.relu = nn.ReLU(inplace=True)
    
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):

        x = self.conv(x)
        x = self.relu(self.gn(x))
        x = self.upsample2x(x)

        return x

    def upsample2x(self, x):
        _, _, h, w = x.shape
        x = F.interpolate(x, size=(h*2, w*2),
                          mode='bilinear', align_corners=True)
        return x

@BACKBONES.register_module()
class BEVFormerBackbone(nn.Module):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 roi_size,
                 bev_h,
                 bev_w,
                 img_backbone=None,
                 img_neck=None,
                 transformer=None,
                 positional_encoding=None,
                 use_grid_mask=True,
                 upsample=False,
                 up_outdim=128,
                 modality='camera',
                 lidar_encoder=None,
                 lidar_bev_proj=None,
                 **kwargs):
        super(BEVFormerBackbone, self).__init__()

        assert modality in ('camera', 'lidar'), \
            f"modality must be 'camera' or 'lidar', got {modality!r}"
        self.modality = modality

        # image feature
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask

        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
            self.with_img_neck = True
        else:
            self.with_img_neck = False

        # LiDAR branch. Voxelization + SparseEncoder alone: no VFE, no
        # pts_backbone, no pts_neck -- SparseEncoder already emits a dense
        # BEV-shaped tensor, so mean-pooled raw point coordinates go straight
        # into it. Same structure as the sibling GeMap/MapTRv2 LiDAR path.
        if self.modality == 'lidar':
            assert lidar_encoder is not None and lidar_bev_proj is not None, \
                "modality='lidar' needs both lidar_encoder and lidar_bev_proj"
            voxelize_cfg = lidar_encoder['voxelize']
            if voxelize_cfg.get('max_num_points', -1) > 0:
                voxelize_module = Voxelization(**voxelize_cfg)
            else:
                voxelize_module = DynamicScatter(**voxelize_cfg)
            self.lidar_modal_extractor = nn.ModuleDict({
                'voxelize': voxelize_module,
                'backbone': builder.build_middle_encoder(
                    lidar_encoder['backbone']),
            })
            self.voxelize_reduce = lidar_encoder.get('voxelize_reduce', True)
            self.lidar_bev_proj = builder.build_neck(lidar_bev_proj)

        self.bev_h = bev_h
        self.bev_w = bev_w

        self.real_w = roi_size[0]
        self.real_h = roi_size[1]

        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims

        self.upsample = upsample
        if self.upsample:
            self.up = UpsampleBlock(self.transformer.embed_dims, up_outdim)

        self._init_layers()
        self.init_weights()


    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        self.bev_embedding = nn.Embedding(
            self.bev_h * self.bev_w, self.embed_dims)


    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        # Absent on the LiDAR-only path. The transformer above is still built
        # and initialized there, unused, to keep this class and the config
        # shape identical across modalities -- find_unused_parameters=True
        # covers the resulting gradient-free parameters.
        if getattr(self, 'img_backbone', None) is not None:
            self.img_backbone.init_weights()
        if self.with_img_neck:
            self.img_neck.init_weights()

        if self.upsample:
            self.up.init_weights()

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Hard-voxelize a batch of point clouds.

        Args:
            points (list[Tensor]): per-sample points, each (N_i, C).

        Returns:
            tuple: ``feats`` (M, C) mean-pooled per voxel, ``coords`` (M, 4)
            as ``(batch_idx, z, y, x)``, and ``sizes`` (M,) points per voxel.
        """
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.lidar_modal_extractor['voxelize'](res)
            if len(ret) == 3:  # hard voxelize
                f, c, n = ret
            else:
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(
                    feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=('points'), out_fp32=True)
    def extract_lidar_feat(self, points, img_metas=None):
        """Voxelize a batch and run the sparse encoder over it.

        Returns a dense ``(B, C*D, H, W)`` BEV tensor, where the sparse
        encoder has collapsed z into the channel axis. With this repo's stock
        mmdet3d/mmcv, ``H`` indexes y and ``W`` indexes x (the vendored fork
        in the sibling repos is transposed; see ``forward``).
        """
        feats, coords, sizes = self.voxelize(points)
        if coords.numel() == 0:
            # Otherwise this surfaces as a bare IndexError inside the sparse
            # encoder, with nothing naming the tile that caused it.
            sample_idxs = ([m.get('sample_idx') for m in img_metas]
                           if img_metas else None)
            raise RuntimeError(
                'extract_lidar_feat: voxelization produced zero voxels for '
                f'this batch. points per sample: {[p.shape[0] for p in points]}, '
                f'sample_idx: {sample_idxs}. Check the raw point count and '
                'coordinate range for these tiles against '
                'lidar_point_cloud_range and z_max.')
        batch_size = coords[-1, 0] + 1
        return self.lidar_modal_extractor['backbone'](feats, coords,
                                                      batch_size)
    
    # @auto_fp16(apply_to=('img'))
    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img = img.squeeze(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        
        return img_feats_reshaped

    def forward_lidar(self, points, img_metas=None):
        """LiDAR-only BEV, replacing (not fusing with) the camera BEV.

        Returns ``[B, embed_dims, bev_h, bev_w]``, the same contract
        ``forward`` has on the camera path, so everything downstream --
        streaming fusion, the head, the losses -- is unchanged.

        The camera path never runs here: no image backbone/neck, no
        ``SpatialCrossAttention``, no ``BEVFormerEncoder``, and neither
        ``bev_embedding`` nor ``positional_encoding``. Those modules still
        exist so the class and config keep one shape across modalities.

        Two geometry notes, both measured rather than assumed:

        * No x/y transpose. This repo's stock mmdet3d 1.0.0rc6 + mmcv ops emit
          voxel coords as ``(batch, z, y, x)``, so ``SparseEncoder``'s dense
          output already has ``H = y`` and ``W = x``. The sibling GeMap/MapTRv2
          configs carry a ``permute(0, 1, 3, 2)`` because their vendored fork
          patched the CUDA kernel to emit ``(x, y, z)`` instead. CARLA's square
          tiles would hide a mistake here.

        * No y flip either, and this one is subtle enough to have been got
          wrong once. There are two BEV row conventions in this codebase and
          they do NOT agree:

            - the camera encoder writes row 0 = y_max
              (``BEVFormerEncoder.get_reference_points`` uses
              ``ys = linspace(H-0.5, 0.5, H)``, and the mappers' ``plane``
              buffer matches it);
            - the head READS row 0 = y_min. Its reference points are
              ``VectorizeMap.normalize_line`` output, ``y_n = (y + roi/2) /
              roi``, and ``CustomMSDeformableAttention`` feeds that straight
              to ``grid_sample``, whose y axis indexes H ascending. Measured:
              GT at y=-12 is read at row 1.5, y=+12 at row 97.5.

          The camera path absorbs the mismatch because its BEV is a learned
          rearrangement of image features -- the encoder simply learns to
          write rows in whatever order the head reads them. A LiDAR BEV
          cannot: it is a geometric projection, and a convolution is
          translation-equivariant, so it cannot represent a global flip.
          Matching the camera encoder here (which this code did at first)
          therefore mirrors every prediction about the x axis.

          ``SparseEncoder`` emits rows in ascending y, which is already what
          the head reads, so the right thing is to leave it alone. Verified
          on 60 test tiles by correlating the BEV rows holding road returns
          against the rows holding GT, as the head indexes them: +0.31
          unflipped vs -0.04 flipped, with unflipped better on 90% of tiles.
        """
        lidar_feat = self.extract_lidar_feat(points, img_metas=img_metas)
        bev = F.interpolate(
            lidar_feat,
            size=(self.bev_h, self.bev_w),
            mode='bicubic',
            align_corners=False)
        return self.lidar_bev_proj([bev]).contiguous()

    def forward(self, img=None, img_metas=None, *args, points=None,
                prev_bev=None, only_bev=False, **kwargs):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            points (list[Tensor]): per-sample LiDAR points, used only when
                ``modality == 'lidar'``.
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        if self.modality == 'lidar':
            assert points is not None, \
                "modality='lidar' but no points reached the backbone; check " \
                "the pipeline's Collect3D keys include 'points'"
            return self.forward_lidar(points, img_metas=img_metas)

        mlvl_feats = self.extract_img_feat(img=img, img_metas=img_metas)

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                            device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        outs =  self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                            self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        
        outs = outs.unflatten(1,(self.bev_h,self.bev_w)).permute(0,3,1,2).contiguous()
        
        if self.upsample:
            outs = self.up(outs)
        
        return outs
