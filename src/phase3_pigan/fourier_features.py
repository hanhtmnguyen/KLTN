"""
Fourier Feature Embeddings for spatial coordinates.

Maps low-dimensional (x, y) coordinates into a high-dimensional Fourier
space, preventing the network from converging to trivial smooth solutions
and enabling learning of high-frequency urban micro-topography edges.

Reference
---------
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
    """Project spatial coordinates into Fourier feature space.

    Given coordinates ``v ∈ ℝ^d``, computes:
        γ(v) = [cos(2π·B·v), sin(2π·B·v)]

    where ``B`` is a fixed random Gaussian matrix that controls the
    frequency bandwidth of the embedding.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input coordinates (2 for x,y).
    mapping_size : int
        Number of random Fourier frequencies.  Output dim = 2 * mapping_size.
    scale : float
        Standard deviation of the Gaussian matrix ``B``.
        Larger values → higher frequencies captured.
    """

    def __init__(
        self,
        input_dim: int = 2,
        mapping_size: int = 256,
        scale: float = 10.0,
    ):
        super().__init__()
        # B is frozen (not learnable) — registered as a buffer
        B = torch.randn(input_dim, mapping_size) * scale
        self.register_buffer("B", B)
        self.output_dim = 2 * mapping_size

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        coords : torch.Tensor
            Shape ``(B, 2)`` or ``(B, H, W, 2)`` with normalised
            coordinates in [0, 1].

        Returns
        -------
        torch.Tensor
            Shape ``(..., 2 * mapping_size)``.
        """
        original_shape = coords.shape
        if coords.dim() > 2:
            # Flatten spatial dims: (B, H, W, 2) → (B*H*W, 2)
            coords = coords.reshape(-1, coords.shape[-1])

        proj = 2.0 * math.pi * coords @ self.B  # (N, mapping_size)
        out = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # (N, 2*M)

        if len(original_shape) > 2:
            # Restore spatial dims
            out = out.reshape(*original_shape[:-1], self.output_dim)

        return out


def make_coordinate_grid(
    height: int,
    width: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a normalised (y, x) coordinate grid.

    Parameters
    ----------
    height, width : int
        Grid dimensions in pixels.
    device : torch.device | None

    Returns
    -------
    torch.Tensor
        Shape ``(1, H, W, 2)`` with values in [0, 1].
    """
    y = torch.linspace(0, 1, height, device=device)
    x = torch.linspace(0, 1, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1)  # (H, W, 2)
    return grid.unsqueeze(0)  # (1, H, W, 2)
