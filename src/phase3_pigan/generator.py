"""
PI-GAN Generator — U-Net with Fourier Feature Embeddings.

Takes HAND flood maps + DTM + building features + terrain features as
input and super-resolves them to HEC-RAS quality at 10 m.

Architecture
------------
Input channels (8 physical + 512 Fourier = 520):
  [HAND_depth, HAND_vu, HAND_vv, DTM, Manning_n, slope, TWI, dist_river]
  + Fourier embeddings of spatial (x, y) coordinates

Output channels (3):
  [depth, velocity_u, velocity_v]  — at 10 m resolution
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.phase3_pigan.fourier_features import (
    FourierFeatureEmbedding,
    make_coordinate_grid,
)


class _DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int, negative_slope: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    """Downsampling: MaxPool → DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class _Up(nn.Module):
    """Upsampling: ConvTranspose → concatenate skip → DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatches from odd dimensions
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = nn.functional.pad(
            x, [diff_x // 2, diff_x - diff_x // 2,
                diff_y // 2, diff_y - diff_y // 2]
        )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class PIGANGenerator(nn.Module):
    """U-Net generator with Fourier Feature Embeddings.

    Parameters
    ----------
    in_physical_channels : int
        Number of physical input channels (default 8):
        HAND_depth, HAND_vu, HAND_vv, DTM, Manning_n, slope, TWI, dist_river.
    out_channels : int
        Number of output channels (default 3):
        depth, velocity_u, velocity_v.
    fourier_mapping_size : int
        Number of Fourier frequency components.
        Total Fourier channels = 2 * mapping_size.
    fourier_scale : float
        Scale parameter for the Fourier embedding Gaussian matrix.
    base_features : int
        Number of features in the first encoder layer.
    """

    def __init__(
        self,
        in_physical_channels: int = 8,
        out_channels: int = 3,
        fourier_mapping_size: int = 256,
        fourier_scale: float = 10.0,
        base_features: int = 64,
    ):
        super().__init__()

        self.fourier_embed = FourierFeatureEmbedding(
            input_dim=2,
            mapping_size=fourier_mapping_size,
            scale=fourier_scale,
        )
        fourier_ch = self.fourier_embed.output_dim  # 2 * mapping_size

        total_in = in_physical_channels + fourier_ch

        # Encoder
        self.inc = _DoubleConv(total_in, base_features)
        self.down1 = _Down(base_features, base_features * 2)
        self.down2 = _Down(base_features * 2, base_features * 4)
        self.down3 = _Down(base_features * 4, base_features * 8)
        self.down4 = _Down(base_features * 8, base_features * 16)

        # Decoder
        self.up1 = _Up(base_features * 16, base_features * 8)
        self.up2 = _Up(base_features * 8, base_features * 4)
        self.up3 = _Up(base_features * 4, base_features * 2)
        self.up4 = _Up(base_features * 2, base_features)

        # Output head
        self.outc = nn.Conv2d(base_features, out_channels, kernel_size=1)

        # Separate activations for depth (non-negative) and velocity (signed)
        self.depth_activation = nn.ReLU()

    def forward(self, x_physical: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_physical : torch.Tensor
            Shape ``(B, C_phys, H, W)`` — the 8 physical input channels.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 3, H, W)`` — [depth, velocity_u, velocity_v].
        """
        B, _, H, W = x_physical.shape

        # Generate coordinate grid and Fourier embeddings
        coord_grid = make_coordinate_grid(H, W, device=x_physical.device)
        coord_grid = coord_grid.expand(B, -1, -1, -1)  # (B, H, W, 2)
        fourier = self.fourier_embed(coord_grid)          # (B, H, W, F)
        fourier = fourier.permute(0, 3, 1, 2)             # (B, F, H, W)

        # Concatenate physical + Fourier channels
        x = torch.cat([x_physical, fourier], dim=1)  # (B, C_phys+F, H, W)

        # U-Net forward
        x1 = self.inc(x)       # (B, 64, H, W)
        x2 = self.down1(x1)    # (B, 128, H/2, W/2)
        x3 = self.down2(x2)    # (B, 256, H/4, W/4)
        x4 = self.down3(x3)    # (B, 512, H/8, W/8)
        x5 = self.down4(x4)    # (B, 1024, H/16, W/16)

        out = self.up1(x5, x4) # (B, 512, H/8, W/8)
        out = self.up2(out, x3) # (B, 256, H/4, W/4)
        out = self.up3(out, x2) # (B, 128, H/2, W/2)
        out = self.up4(out, x1) # (B, 64, H, W)

        out = self.outc(out)    # (B, 3, H, W)

        # Apply activation: depth ≥ 0, velocities unrestricted
        depth = self.depth_activation(out[:, 0:1, :, :])
        velocities = out[:, 1:3, :, :]  # No activation — signed

        return torch.cat([depth, velocities], dim=1)
