"""
PI-GAN Discriminator — PatchGAN for Supervised Realism Enforcement.

Classifies whether local patches of flood maps are real (from HEC-RAS 2D)
or fake (from the generator). Encourages sharp local details and realistic
urban flood boundaries.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.logging_config import get_logger

logger = get_logger("da.model.discriminator")


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator for flood map realism assessment.

    Parameters
    ----------
    in_channels : int
        Number of input channels (3: `depth`, `velocity_u`, `velocity_v`).
    base_features : int
        Features in the first conv layer.
    n_layers : int
        Number of intermediate conv layers.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_features: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, base_features, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf = base_features
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers.extend([
                nn.Conv2d(nf_prev, nf, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(nf),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        nf_prev = nf
        nf = min(nf * 2, 512)
        layers.extend([
            nn.Conv2d(nf_prev, nf, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
        ])

        layers.append(nn.Conv2d(nf, 1, 4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape `(B, 3, H, W)` — flood map (real or generated).

        Returns
        -------
        torch.Tensor
            Shape `(B, 1, H', W')` — patch-level real/fake logits.
        """
        return self.model(x)
