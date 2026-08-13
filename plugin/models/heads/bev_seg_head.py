import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS


@HEADS.register_module()
class BEVSegHead(nn.Module):
    """Auxiliary BEV segmentation head.

    Rasterizes nothing itself -- it consumes the ``semantic_mask`` that
    ``RasterizeMap`` already puts in the sample -- and just predicts a
    per-class occupancy map straight off the BEV feature, supervising it
    with a pos-weighted BCE.

    Why this exists: the LiDAR-only model's *only* gradient into the
    SparseEncoder and the 7.4M-parameter ConvFuser otherwise arrives through
    sparse polyline matching, through a denoising decoder. That is a very
    thin signal for an encoder trained from scratch. Every sibling CARLA
    LiDAR config -- MapTRv2, GeMap and PseudoMapTrainer -- turns on
    ``aux_seg`` with ``bev_seg=True`` for exactly this reason, and this is
    the equivalent: MapTRv2's is
    ``Conv2d(C, C, 3, padding=1, bias=False) -> ReLU -> Conv2d(C, n_cls, 1)``
    with a ``SimpleLoss`` (BCE with ``pos_weight``), which is what is
    reproduced here.

    Orientation: the mask from ``RasterizeMap.line_ego_to_mask`` is drawn
    with ``cv2.polylines`` after translating by ``canvas_size / 2``, so its
    row index increases with y -- row 0 is y_min. That matches both the
    LiDAR BEV out of ``SparseEncoder`` and the convention the detection head
    reads with, so no flip is applied here. ``forward`` asserts the mask and
    the feature agree on H and W, which is what would catch a
    ``canvas_size`` given as (h, w) instead of (w, h).

    Args:
        in_channels (int): channels of the BEV feature.
        num_classes (int): segmentation classes; one channel per map class.
        mid_channels (int): hidden width. Defaults to in_channels.
        pos_weight (float): positive-class weight in the BCE, to counter how
            few BEV cells a thin polyline covers. MapTRv2 uses 4.0 for
            ``bev_seg``.
        loss_weight (float): weight of this loss in the total.
    """

    def __init__(self,
                 in_channels,
                 num_classes=1,
                 mid_channels=None,
                 pos_weight=4.0,
                 loss_weight=1.0):
        super().__init__()
        mid_channels = mid_channels or in_channels
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.register_buffer('pos_weight',
                             torch.tensor(float(pos_weight)))
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    def forward(self, bev_features, semantic_mask):
        """
        Args:
            bev_features (Tensor): [B, C, bev_h, bev_w].
            semantic_mask (Tensor): [B, num_classes, bev_h, bev_w], the
                rasterized GT. Bool/uint8 from the pipeline.

        Returns:
            Tensor: the scalar auxiliary loss.
        """
        logits = self.seg_head(bev_features)
        target = semantic_mask.to(logits.dtype)
        if target.dim() == 3:                      # (B, H, W) -> (B, 1, H, W)
            target = target.unsqueeze(1)
        assert target.shape[1] == self.num_classes, (
            f'semantic_mask has {target.shape[1]} channels but the seg head '
            f'was built for {self.num_classes}; num_classes must match the '
            "config's cat2id")
        assert target.shape[-2:] == logits.shape[-2:], (
            f'semantic_mask is {tuple(target.shape[-2:])} but the BEV is '
            f'{tuple(logits.shape[-2:])}. RasterizeMap takes canvas_size as '
            '(w, h), so it should be (bev_w, bev_h)')

        loss = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight)
        return loss * self.loss_weight
