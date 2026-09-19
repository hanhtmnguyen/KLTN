"""
Unified Data Assimilation Loss — Combining Boundary Conditions, Gauges, & SAR.

Integrates upstream discharge (`L_upstream`), downstream WSE (`L_downstream`),
in-situ gauge stations (`L_gauge`), and spatial Sentinel-1 SAR flood extent
(`L_SAR`) into a unified multi-objective data assimilation regularization term.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.boundary_condition_loss import UpstreamBCLoss, DownstreamBCLoss


class GaugeStationLoss(nn.Module):
    """Point-wise loss at in-situ water level gauge station locations."""

    def __init__(
        self,
        station_pixels: list[tuple[int, int]] | None = None,
        observed_wse: list[float] | None = None,
    ):
        super().__init__()
        self.station_pixels = station_pixels or []
        self.observed_wse = observed_wse or []

    def forward(self, h: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        if not self.station_pixels:
            return torch.tensor(0.0, device=h.device, requires_grad=True)

        wse_pred = h + z_b
        loss = torch.tensor(0.0, device=h.device, requires_grad=True)

        for (row, col), wse_obs in zip(self.station_pixels, self.observed_wse):
            if row < h.shape[2] and col < h.shape[3]:
                pred_val = wse_pred[:, 0, row, col]
                loss = loss + torch.mean((pred_val - wse_obs) ** 2)

        return loss / max(len(self.station_pixels), 1)


class SARFloodExtentLoss(nn.Module):
    """Binary cross-entropy loss with Sentinel-1 SAR-derived flood extent mask."""

    def __init__(
        self,
        depth_threshold: float = 0.05,
        pos_weight: float = 2.0,
    ):
        super().__init__()
        self.threshold = depth_threshold
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, h: torch.Tensor, sar_mask: torch.Tensor) -> torch.Tensor:
        if sar_mask is None or sar_mask.numel() == 0:
            return torch.tensor(0.0, device=h.device, requires_grad=True)

        logits = (h - self.threshold) * 20.0
        self.bce.pos_weight = self.bce.pos_weight.to(h.device)
        return self.bce(logits, sar_mask.float())


class UnifiedDataAssimilationLoss(nn.Module):
    """Unified Data Assimilation regularization bundling all observational constraints.

    Parameters
    ----------
    cell_width : float
        Grid spacing (`dy`) for discharge calculation.
    weights : dict | None
        Mapping of component loss weights (`lambda_upstream`, `lambda_downstream`,
        `lambda_gauge`, `lambda_sar`).
    """

    def __init__(self, cell_width: float = 10.0, weights: dict | None = None):
        super().__init__()
        self.weights = weights or {
            "upstream": 1.0,
            "downstream": 1.0,
            "gauge": 0.5,
            "sar": 0.1,
        }
        self.up_loss = UpstreamBCLoss(cell_width=cell_width)
        self.down_loss = DownstreamBCLoss()
        self.gauge_loss = GaugeStationLoss()
        self.sar_loss = SARFloodExtentLoss()

    def forward(
        self,
        h: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        z_b: torch.Tensor,
        upstream_mask: torch.Tensor | None = None,
        q_target: torch.Tensor | None = None,
        downstream_obs: list[torch.Tensor] | torch.Tensor | None = None,
        sar_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all data assimilation loss components and their weighted total.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing `"total"`, `"upstream"`, `"downstream"`,
            `"gauge"`, and `"sar"` loss tensors.
        """
        l_up = torch.tensor(0.0, device=h.device)
        if upstream_mask is not None and q_target is not None:
            l_up = self.up_loss(h, u, v, upstream_mask, q_target)

        l_down = torch.tensor(0.0, device=h.device)
        if downstream_obs is not None:
            l_down = self.down_loss(h, z_b, downstream_obs)

        l_gauge = self.gauge_loss(h, z_b)

        l_sar = torch.tensor(0.0, device=h.device)
        if sar_mask is not None and sar_mask.numel() > 0:
            l_sar = self.sar_loss(h, sar_mask)

        total = (
            self.weights["upstream"] * l_up
            + self.weights["downstream"] * l_down
            + self.weights["gauge"] * l_gauge
            + self.weights["sar"] * l_sar
        )

        return {
            "total": total,
            "upstream": l_up,
            "downstream": l_down,
            "gauge": l_gauge,
            "sar": l_sar,
        }
