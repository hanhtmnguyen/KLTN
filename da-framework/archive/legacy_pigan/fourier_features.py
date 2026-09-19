"""
Spatiotemporal Fourier Feature Embeddings for Coordinates (x, y, t).

Maps low-dimensional spatial (x, y) or spatiotemporal (x, y, t) coordinates
into a high-dimensional Fourier feature space, allowing the PI-GAN generator
to learn high-frequency urban micro-topography edges and sharp temporal flood
wave dynamics.

References
----------
Tancik et al. (2020) "Fourier Features Let Networks Learn High Frequency
Functions in Low Dimensional Domains" — NeurIPS.

Feng et al. (2022) "Physics-informed neural networks of the Saint-Venant
equations for downscaling a large-scale river model" — WRR.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class FourierFeatureEmbedding(nn.Module):
    """Project spatial (x, y) coordinates into Fourier feature space.

    Given coordinates ``v ∈ ℝ^d``, computes:
        γ(v) = [cos(2π·B·v), sin(2π·B·v)]

    Parameters
    ----------
    input_dim : int
        Dimensionality of input coordinates (2 for x,y).
    mapping_size : int
        Number of random Fourier frequencies. Output dim = 2 * mapping_size.
    scale : float
        Standard deviation of the Gaussian matrix ``B``.
    """

    def __init__(
        self,
        input_dim: int = 2,
        mapping_size: int = 256,
        scale: float = 10.0,
    ):
        super().__init__()
        B = torch.randn(input_dim, mapping_size) * scale
        self.register_buffer("B", B)
        self.output_dim = 2 * mapping_size

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        original_shape = coords.shape
        if coords.dim() > 2:
            coords = coords.reshape(-1, coords.shape[-1])

        proj = 2.0 * math.pi * coords @ self.B
        out = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

        if len(original_shape) > 2:
            out = out.reshape(*original_shape[:-1], self.output_dim)

        return out


class SpatioTemporalFourierEmbedding(nn.Module):
    """Project spatiotemporal coordinates (x, y, t) into Fourier feature space.

    Parameters
    ----------
    mapping_size : int
        Number of random Fourier frequencies. Output dim = 2 * mapping_size.
    scale_spatial : float
        Standard deviation of Gaussian frequencies for spatial axes (x, y).
    scale_temporal : float
        Standard deviation of Gaussian frequencies for temporal axis (t).
    """

    def __init__(
        self,
        mapping_size: int = 256,
        scale_spatial: float = 10.0,
        scale_temporal: float = 5.0,
    ):
        super().__init__()
        B_spatial = torch.randn(2, mapping_size) * scale_spatial
        B_temporal = torch.randn(1, mapping_size) * scale_temporal
        B = torch.cat([B_spatial, B_temporal], dim=0)  # Shape (3, mapping_size)
        self.register_buffer("B", B)
        self.output_dim = 2 * mapping_size

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        coords : torch.Tensor
            Shape `(B, H, W, 3)` with normalized `(x, y, t)` coordinates in `[0, 1]`.

        Returns
        -------
        torch.Tensor
            Shape `(B, H, W, 2 * mapping_size)`.
        """
        original_shape = coords.shape
        if coords.dim() > 2:
            coords = coords.reshape(-1, coords.shape[-1])

        proj = 2.0 * math.pi * coords @ self.B
        out = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

        if len(original_shape) > 2:
            out = out.reshape(*original_shape[:-1], self.output_dim)

        return out


def make_coordinate_grid(
    height: int,
    width: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a normalized 2D (y, x) coordinate grid.

    Returns
    -------
    torch.Tensor
        Shape `(1, H, W, 2)` with values in `[0, 1]`.
    """
    y = torch.linspace(0, 1, height, device=device)
    x = torch.linspace(0, 1, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1)
    return grid.unsqueeze(0)


def make_spatiotemporal_grid(
    height: int,
    width: int,
    time_norm: float | torch.Tensor,
    device: torch.device | None = None,
    batch_size: int = 1,
) -> torch.Tensor:
    """Create a normalized 3D (x, y, t) coordinate grid for a given time step.

    Parameters
    ----------
    height, width : int
        Spatial grid dimensions.
    time_norm : float | torch.Tensor
        Normalized time value in `[0, 1]` or tensor of shape `(batch_size,)`.
    device : torch.device | None
    batch_size : int
        Number of samples in batch.

    Returns
    -------
    torch.Tensor
        Shape `(batch_size, H, W, 3)` containing `[x, y, t]`.
    """
    y = torch.linspace(0, 1, height, device=device)
    x = torch.linspace(0, 1, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid_2d = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)  # (B, H, W, 2)

    if isinstance(time_norm, (float, int)):
        t_val = torch.full((batch_size, height, width, 1), float(time_norm), device=device)
    elif isinstance(time_norm, torch.Tensor):
        if time_norm.dim() == 0:
            t_val = time_norm.view(1, 1, 1, 1).expand(batch_size, height, width, 1)
        elif time_norm.dim() == 1:
            t_val = time_norm.view(batch_size, 1, 1, 1).expand(batch_size, height, width, 1)
        else:
            t_val = time_norm
    else:
        t_val = torch.zeros((batch_size, height, width, 1), device=device)

    grid_3d = torch.cat([grid_2d, t_val], dim=-1)  # (B, H, W, 3)
    return grid_3d
