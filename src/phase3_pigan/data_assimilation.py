"""
Data Assimilation losses — gauge stations and Sentinel-1 SAR.

Anchors PI-GAN predictions to real-world observations during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GaugeStationLoss(nn.Module):
    """Point-wise loss at water level gauge station locations.

    Penalises deviation of predicted water surface elevation (h + z_b)
    from observed WSE at known gauge locations.

    Parameters
    ----------
    station_pixels : list[tuple[int, int]]
        List of (row, col) pixel coordinates of gauge stations
        within the training tile grid.
    observed_wse : list[float]
        Observed water surface elevation at each station (metres).
    """

    def __init__(
        self,
        station_pixels: list[tuple[int, int]] | None = None,
        observed_wse: list[float] | None = None,
    ):
        super().__init__()
        self.station_pixels = station_pixels or []
        self.observed_wse = observed_wse or []

    def forward(
        self,
        h: torch.Tensor,
        z_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h : torch.Tensor
            Predicted depth (B, 1, H, W).
        z_b : torch.Tensor
            Bed elevation (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Scalar gauge loss.
        """
        if not self.station_pixels:
            return torch.tensor(0.0, device=h.device, requires_grad=True)

        wse_pred = h + z_b  # Water surface elevation
        loss = torch.tensor(0.0, device=h.device, requires_grad=True)

        for (row, col), wse_obs in zip(self.station_pixels, self.observed_wse):
            if row < h.shape[2] and col < h.shape[3]:
                pred_val = wse_pred[:, 0, row, col]
                loss = loss + torch.mean((pred_val - wse_obs) ** 2)

        return loss / max(len(self.station_pixels), 1)


class SARFloodExtentLoss(nn.Module):
    """Binary cross-entropy loss with SAR-derived flood extent.

    Penalises the model when:
    - It predicts dry cells where SAR shows flooding
    - It predicts wet cells where SAR shows no flooding

    Parameters
    ----------
    depth_threshold : float
        Depth threshold for converting predicted depth to binary mask.
    pos_weight : float
        Weight for positive (flooded) class to handle class imbalance.
    """

    def __init__(
        self,
        depth_threshold: float = 0.05,
        pos_weight: float = 2.0,
    ):
        super().__init__()
        self.threshold = depth_threshold
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

    def forward(
        self,
        h: torch.Tensor,
        sar_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h : torch.Tensor
            Predicted depth (B, 1, H, W).
        sar_mask : torch.Tensor
            Binary SAR flood extent (B, 1, H, W).
            1 = wet (flooded), 0 = dry.

        Returns
        -------
        torch.Tensor
            Scalar SAR loss.
        """
        if sar_mask is None:
            return torch.tensor(0.0, device=h.device, requires_grad=True)

        # Convert depth to logits for BCE
        # Scale depth so that threshold maps to logit ≈ 0
        logits = (h - self.threshold) * 20.0  # Steeper sigmoid

        self.bce.pos_weight = self.bce.pos_weight.to(h.device)
        return self.bce(logits, sar_mask.float())
