"""
Boundary Condition Loss (`L_BC`) for Multi-Source Remote Sensing Data Assimilation.

Enforces physical agreement between the generated hydrodynamic fields and external
macro-scale observations at the upstream and downstream cross-sections:
1. `UpstreamBCLoss`: Matches simulated cross-sectional discharge Q_pred(t) against
   global hydrological models (GEOGloWS ECMWF Streamflow Service).
2. `DownstreamBCLoss`: Matches simulated Water Surface Elevation (WSE) against
   spaceborne laser altimetry profiles along inland water tracks (ICESat-2 ATL13).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UpstreamBCLoss(nn.Module):
    """Upstream Boundary Condition Loss matching discharge Q(t) to GEOGloWS.

    Computes predicted discharge across the upstream boundary cells:
        Q_pred(t) = ∑ (h_i(t) * u_normal_i(t) * dy_i)
    and penalizes squared deviation from GEOGloWS discharge Q_GEOGloWS(t).

    Parameters
    ----------
    cell_width : float
        Grid resolution in meters (`dy` along boundary cross-section, default 10m).
    """

    def __init__(self, cell_width: float = 10.0):
        super().__init__()
        self.cell_width = cell_width

    def forward(
        self,
        h: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        upstream_mask: torch.Tensor,
        q_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h, u, v : torch.Tensor
            Predicted hydrodynamic fields of shape `(B, 1, H, W)`.
        upstream_mask : torch.Tensor
            Binary mask indicating upstream cross-section cells `(B, H, W)` (`1=boundary`).
        q_target : torch.Tensor
            GEOGloWS discharge target `(B,)` in m³/s for the current time step.

        Returns
        -------
        torch.Tensor
            Scalar discharge constraint loss `L_upstream`.
        """
        if upstream_mask is None or torch.sum(upstream_mask) == 0:
            return torch.tensor(0.0, device=h.device, requires_grad=True)

        B, _, H, W = h.shape
        mask = upstream_mask.unsqueeze(1).float()  # (B, 1, H, W)

        # Approximate normal velocity across upstream cross-section (typically eastward along x=0)
        u_normal = torch.sqrt(u**2 + v**2 + 1e-8)
        flux_per_cell = h * u_normal * self.cell_width * mask

        # Total discharge across cross-section per sample in batch
        q_pred = torch.sum(flux_per_cell.view(B, -1), dim=1)  # (B,)

        loss = torch.mean((q_pred - q_target.to(h.device)) ** 2)
        # Normalize scale relative to typical flood flow magnitudes (10^6 m^6/s^2)
        return loss / 1e6


class DownstreamBCLoss(nn.Module):
    """Downstream Boundary Condition Loss matching WSE to ICESat-2 ATL13 profiles.

    Computes predicted Water Surface Elevation along downstream observation pixels:
        wse_pred(x_i, y_i, t) = h(x_i, y_i, t) + z_b(x_i, y_i)
    and penalizes squared deviation from ATL13 photon elevation observations.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        h: torch.Tensor,
        z_b: torch.Tensor,
        downstream_obs: list[torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h : torch.Tensor
            Predicted depth `(B, 1, H, W)`.
        z_b : torch.Tensor
            Bed elevation / DTM `(B, 1, H, W)`.
        downstream_obs : list[torch.Tensor] | torch.Tensor
            For each sample `b` in batch, tensor of shape `(N_obs, 3)` containing
            `[row_idx, col_idx, observed_wse_value]` from ATL13.

        Returns
        -------
        torch.Tensor
            Scalar WSE constraint loss `L_downstream`.
        """
        if downstream_obs is None:
            return torch.tensor(0.0, device=h.device, requires_grad=True)

        wse_pred = h + z_b  # (B, 1, H, W)
        total_loss = torch.tensor(0.0, device=h.device, requires_grad=True)
        count = 0

        B = h.shape[0]
        for b in range(B):
            if isinstance(downstream_obs, torch.Tensor):
                obs = downstream_obs[b] if downstream_obs.dim() > 2 else downstream_obs
            else:
                obs = downstream_obs[b]

            if obs is None or len(obs) == 0:
                continue

            rows = obs[:, 0].long()
            cols = obs[:, 1].long()
            wse_target = obs[:, 2].to(h.device)

            # Filter coordinates within patch boundaries
            valid = (rows >= 0) & (rows < h.shape[2]) & (cols >= 0) & (cols < h.shape[3])
            if not torch.any(valid):
                continue

            rows_v = rows[valid]
            cols_v = cols[valid]
            wse_target_v = wse_target[valid]

            pred_vals = wse_pred[b, 0, rows_v, cols_v]
            total_loss = total_loss + torch.sum((pred_vals - wse_target_v) ** 2)
            count += len(rows_v)

        if count == 0:
            return torch.tensor(0.0, device=h.device, requires_grad=True)

        return total_loss / count
