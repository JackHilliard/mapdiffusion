from typing import List

import torch
import torch.nn as nn
from mmdet.models import NECKS


@NECKS.register_module()
class ConvFuser(nn.Sequential):
    """Concatenate a list of BEV feature maps and project to ``out_channels``.

    Kept identical to the sibling GeMap/MapTRv2 module of the same name so the
    LiDAR branch matches theirs. On the LiDAR-only path the list always has a
    single element -- ``SparseEncoder`` already emits a dense ``(B, C*D, H, W)``
    BEV tensor, and this just channel-projects it to ``embed_dims`` -- so the
    ``torch.cat`` is an identity and this is a 3x3 conv + BN + ReLU. It stays
    a fuser rather than a plain conv so a future camera+LiDAR variant needs no
    new module.

    Note ``in_channels`` is a *list* despite the upstream type annotation; the
    LiDAR-only config passes ``[3200]``.
    """

    def __init__(self, in_channels: List[int], out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))
