"""
UNet++ (Nested UNet) — pure PyTorch implementation.

Reference: Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation" (2018).
No external dependencies beyond PyTorch.

Output: logits tensor of shape (B, out_channels, H, W).
        If deep_supervision=True, returns a list of logits (coarse → fine).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, pad=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class VGGBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(in_ch, mid_ch),
            ConvBnRelu(mid_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class UNetPlusPlus(nn.Module):
    """
    UNet++ with configurable depth and optional deep supervision.

    Args:
        in_channels:      Number of input channels.
        out_channels:     Number of output channels (1 for binary segmentation).
        base_channels:    Number of filters in the first encoder block.
        depth:            Number of encoder levels (default 4).
        deep_supervision: Return multi-scale outputs if True.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.depth = depth
        self.deep_supervision = deep_supervision

        nb_filter = [base_channels * (2 ** i) for i in range(depth + 1)]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Encoder nodes: X_{i,0}
        self.encoder = nn.ModuleList()
        for i in range(depth + 1):
            in_ch = in_channels if i == 0 else nb_filter[i - 1]
            self.encoder.append(VGGBlock(in_ch, nb_filter[i], nb_filter[i]))

        # Dense skip connection nodes: X_{i,j} for j >= 1
        # Stored in a flat ModuleList indexed by (i, j)
        self.dense = nn.ModuleDict()
        for j in range(1, depth + 1):
            for i in range(depth + 1 - j):
                in_ch = nb_filter[i] * j + nb_filter[i + 1]
                key = f"{i}_{j}"
                self.dense[key] = VGGBlock(in_ch, nb_filter[i], nb_filter[i])

        # Output heads (one per dense column if deep_supervision, else just last)
        if deep_supervision:
            self.heads = nn.ModuleList([
                nn.Conv2d(nb_filter[0], out_channels, 1)
                for _ in range(1, depth + 1)
            ])
        else:
            self.heads = nn.ModuleList([
                nn.Conv2d(nb_filter[0], out_channels, 1)
            ])

    def forward(self, x: torch.Tensor):
        # Collect encoder feature maps
        x_enc = []
        for i, enc in enumerate(self.encoder):
            if i == 0:
                x_enc.append(enc(x))
            else:
                x_enc.append(enc(self.pool(x_enc[-1])))

        # x_node[i][j] = X_{i,j}
        x_node = [[None] * (self.depth + 1) for _ in range(self.depth + 1)]
        for i in range(self.depth + 1):
            x_node[i][0] = x_enc[i]

        outputs = []
        for j in range(1, self.depth + 1):
            for i in range(self.depth + 1 - j):
                # Gather all previous skip connections at level i
                prev_skips = [x_node[i][k] for k in range(j)]
                upsampled = self.up(x_node[i + 1][j - 1])
                concat = torch.cat(prev_skips + [upsampled], dim=1)
                key = f"{i}_{j}"
                x_node[i][j] = self.dense[key](concat)

            if self.deep_supervision:
                outputs.append(self.heads[j - 1](x_node[0][j]))

        if self.deep_supervision:
            return outputs  # list, coarse→fine
        else:
            return self.heads[0](x_node[0][self.depth])
