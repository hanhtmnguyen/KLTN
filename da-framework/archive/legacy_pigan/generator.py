"""
Supervised PI-GAN Spatiotemporal Generator — U-Net with 3D Fourier Embeddings.

Takes static physical terrain channels + HAND inputs alongside spatiotemporal
Fourier embeddings (x, y, t) to predict super-resolved 10m hydrodynamic
fields `[depth, velocity_u, velocity_v]` at any normalized time step `t ∈ [0, 1]`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.fourier_features import (
    FourierFeatureEmbedding,
    SpatioTemporalFourierEmbedding,
    make_coordinate_grid,
    make_spatiotemporal_grid,
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
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = nn.functional.pad(
            x, [diff_x // 2, diff_x - diff_x // 2,
                diff_y // 2, diff_y - diff_y // 2]
        )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class PIGANGenerator(nn.Module):
    """Spatiotemporal U-Net generator with Fourier Feature Embeddings.

    Parameters
    ----------
    in_physical_channels : int
        Number of physical input channels (default 8).
    out_channels : int
        Number of output channels (default 3: `[depth, velocity_u, velocity_v]`).
    fourier_mapping_size : int
        Number of Fourier frequency components.
    fourier_scale : float
        Scale parameter for the Fourier embedding Gaussian matrix.
    base_features : int
        Number of features in the first encoder layer.
    temporal : bool
        If True, embeds 3D coordinates `(x, y, t)`. If False, embeds 2D `(x, y)`.
    """

    def __init__(
        self,
        in_physical_channels: int = 8,
        out_channels: int = 3,
        fourier_mapping_size: int = 256,
        fourier_scale: float = 10.0,
        base_features: int = 64,
        temporal: bool = True,
    ):
        super().__init__()
        self.temporal = temporal

        if self.temporal:
            self.fourier_embed = SpatioTemporalFourierEmbedding(
                mapping_size=fourier_mapping_size,
                scale_spatial=fourier_scale,
                scale_temporal=fourier_scale * 0.5,
            )
        else:
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
        self.depth_activation = nn.ReLU()

    def forward(
        self,
        x_physical: torch.Tensor,
        time_norm: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate super-resolved hydrodynamic fields for a given time step.

        Parameters
        ----------
        x_physical : torch.Tensor
            Shape `(B, C_phys, H, W)` — physical terrain/HAND channels.
        time_norm : float | torch.Tensor | None
            Normalized time coordinate `t ∈ [0, 1]` or tensor `(B,)`.

        Returns
        -------
        torch.Tensor
            Shape `(B, 3, H, W)` containing `[depth, velocity_u, velocity_v]`.
        """
        B, _, H, W = x_physical.shape

        if self.temporal and time_norm is not None:
            coord_grid = make_spatiotemporal_grid(
                H, W, time_norm, device=x_physical.device, batch_size=B
            )
        else:
            coord_grid = make_coordinate_grid(H, W, device=x_physical.device)
            coord_grid = coord_grid.expand(B, -1, -1, -1)

        fourier = self.fourier_embed(coord_grid)  # (B, H, W, F)
        fourier = fourier.permute(0, 3, 1, 2)     # (B, F, H, W)

        x = torch.cat([x_physical, fourier], dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)

        out = self.outc(out)

        depth = self.depth_activation(out[:, 0:1, :, :])
        velocities = out[:, 1:3, :, :]

        return torch.cat([depth, velocities], dim=1)
