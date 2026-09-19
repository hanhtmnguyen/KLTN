"""
Physics-Informed Loss — 2D Shallow Water Equations (Saint-Venant).

Embeds mass conservation and momentum conservation into the training
loss, penalising the generator if it produces physically inconsistent
flood maps (e.g., water flowing uphill, mass violation).

Equations (steady-state 2D SWE)
--------------------------------
Mass:       ∂(hu)/∂x + ∂(hv)/∂y = 0
x-Momentum: ∂(hu² + ½gh²)/∂x + ∂(huv)/∂y = −gh·∂zb/∂x − τbx/ρ
y-Momentum: ∂(huv)/∂x + ∂(hv² + ½gh²)/∂y = −gh·∂zb/∂y − τby/ρ

where τb = ρ·g·n²·|u|·u / h^(1/3)  (Manning friction)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PhysicsInformedLoss(nn.Module):
    """Compute 2D SWE residuals using finite differences on grid data.

    Parameters
    ----------
    gravity : float
        Gravitational acceleration (m/s²).
    dx : float
        Grid spacing in x-direction (metres).
    dy : float
        Grid spacing in y-direction (metres).
    mass_weight : float
        Weight for the mass conservation residual.
    momentum_weight : float
        Weight for the momentum conservation residual.
    """

    def __init__(
        self,
        gravity: float = 9.81,
        dx: float = 10.0,
        dy: float = 10.0,
        mass_weight: float = 1.0,
        momentum_weight: float = 1.0,
    ):
        super().__init__()
        self.g = gravity
        self.dx = dx
        self.dy = dy
        self.mass_weight = mass_weight
        self.momentum_weight = momentum_weight

    def forward(
        self,
        h: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        z_b: torch.Tensor,
        n_manning: torch.Tensor,
    ) -> torch.Tensor:
        """Compute physics residual loss.

        Parameters
        ----------
        h : torch.Tensor
            Predicted water depth (B, 1, H, W).
        u : torch.Tensor
            Predicted x-velocity (B, 1, H, W).
        v : torch.Tensor
            Predicted y-velocity (B, 1, H, W).
        z_b : torch.Tensor
            Bed elevation / DTM (B, 1, H, W).
        n_manning : torch.Tensor
            Manning's roughness coefficient (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Scalar physics loss.
        """
        eps = 1e-6  # Numerical stability

        # Squeeze channel dimension for easier indexing
        h = h.squeeze(1)       # (B, H, W)
        u = u.squeeze(1)
        v = v.squeeze(1)
        z_b = z_b.squeeze(1)
        n = n_manning.squeeze(1)

        # ---- Spatial derivatives via central differences ----
        # Interior cells only (trim 1 pixel on each side)

        # ∂(hu)/∂x  — central difference along x (columns)
        hu = h * u
        dhu_dx = (hu[:, :, 2:] - hu[:, :, :-2]) / (2 * self.dx)

        # ∂(hv)/∂y  — central difference along y (rows)
        hv = h * v
        dhv_dy = (hv[:, 2:, :] - hv[:, :-2, :]) / (2 * self.dy)

        # Trim to common interior region
        # dhu_dx: (B, H, W-2) → trim rows: (B, H-2, W-2)
        # dhv_dy: (B, H-2, W) → trim cols: (B, H-2, W-2)
        dhu_dx_int = dhu_dx[:, 1:-1, :]
        dhv_dy_int = dhv_dy[:, :, 1:-1]

        # ---- MASS CONSERVATION residual ----
        R_mass = dhu_dx_int + dhv_dy_int  # Should be ≈ 0

        # ---- Interior slice of all variables ----
        h_int = h[:, 1:-1, 1:-1]
        u_int = u[:, 1:-1, 1:-1]
        v_int = v[:, 1:-1, 1:-1]
        n_int = n[:, 1:-1, 1:-1]

        # ---- BOTTOM SLOPE terms ----
        dz_dx = (z_b[:, 1:-1, 2:] - z_b[:, 1:-1, :-2]) / (2 * self.dx)
        dz_dy = (z_b[:, 2:, 1:-1] - z_b[:, :-2, 1:-1]) / (2 * self.dy)

        # ---- FRICTION terms (Manning) ----
        speed = torch.sqrt(u_int**2 + v_int**2 + eps)
        h_safe = torch.clamp(h_int, min=eps)
        friction_coeff = self.g * n_int**2 * speed / (h_safe ** (1.0 / 3.0))
        tau_x = friction_coeff * u_int
        tau_y = friction_coeff * v_int

        # ---- X-MOMENTUM residual (simplified steady-state) ----
        # ∂(hu²)/∂x + ∂(huv)/∂y + gh·∂h/∂x + gh·∂zb/∂x + τx = 0
        dh_dx = (h[:, 1:-1, 2:] - h[:, 1:-1, :-2]) / (2 * self.dx)
        pressure_x = self.g * h_int * dh_dx
        slope_x = self.g * h_int * dz_dx

        # Flux terms
        hu2 = h * u * u
        dhu2_dx = (hu2[:, 1:-1, 2:] - hu2[:, 1:-1, :-2]) / (2 * self.dx)
        huv = h * u * v
        dhuv_dy = (huv[:, 2:, 1:-1] - huv[:, :-2, 1:-1]) / (2 * self.dy)

        R_mom_x = dhu2_dx + dhuv_dy + pressure_x + slope_x + tau_x

        # ---- Y-MOMENTUM residual ----
        dh_dy = (h[:, 2:, 1:-1] - h[:, :-2, 1:-1]) / (2 * self.dy)
        pressure_y = self.g * h_int * dh_dy
        slope_y = self.g * h_int * dz_dy

        hv2 = h * v * v
        dhv2_dy = (hv2[:, 2:, 1:-1] - hv2[:, :-2, 1:-1]) / (2 * self.dy)
        dhuv_dx = (huv[:, 1:-1, 2:] - huv[:, 1:-1, :-2]) / (2 * self.dx)

        R_mom_y = dhuv_dx + dhv2_dy + pressure_y + slope_y + tau_y

        # ---- Combine residuals ----
        # Only penalise wet cells (h > threshold)
        wet_mask = (h_int > 0.01).float()

        loss_mass = torch.mean((R_mass * wet_mask) ** 2)
        loss_mom_x = torch.mean((R_mom_x * wet_mask) ** 2)
        loss_mom_y = torch.mean((R_mom_y * wet_mask) ** 2)

        total = (
            self.mass_weight * loss_mass
            + self.momentum_weight * (loss_mom_x + loss_mom_y)
        )

        return total


class NonNegativityLoss(nn.Module):
    """Penalise negative water depths."""

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.relu(-h) ** 2)


class DryConsistencyLoss(nn.Module):
    """Penalise non-zero velocities in dry cells."""

    def __init__(self, depth_threshold: float = 0.01):
        super().__init__()
        self.threshold = depth_threshold

    def forward(
        self, h: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        dry_mask = (h < self.threshold).float()
        return torch.mean(dry_mask * (u**2 + v**2))
