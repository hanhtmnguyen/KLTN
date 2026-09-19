"""
Physics-Informed Loss — Steady & Unsteady 2D Shallow Water Equations.

Embeds mass conservation and momentum conservation into the training
loss, penalizing the generator if it violates hydrodynamic laws across
space `(x, y)` and time `(t)`.

Equations (Unsteady 2D SWE)
---------------------------
Mass:       ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = 0
x-Momentum: ∂(hu)/∂t + ∂(hu² + ½gh²)/∂x + ∂(huv)/∂y = −gh·∂zb/∂x − τbx/ρ
y-Momentum: ∂(hv)/∂t + ∂(huv)/∂x + ∂(hv² + ½gh²)/∂y = −gh·∂zb/∂y − τby/ρ
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PhysicsInformedLoss(nn.Module):
    """Compute steady-state 2D SWE residuals using central finite differences."""

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
        eps = 1e-6

        h = h.squeeze(1)
        u = u.squeeze(1)
        v = v.squeeze(1)
        z_b = z_b.squeeze(1)
        n = n_manning.squeeze(1)

        hu = h * u
        dhu_dx = (hu[:, :, 2:] - hu[:, :, :-2]) / (2 * self.dx)
        hv = h * v
        dhv_dy = (hv[:, 2:, :] - hv[:, :-2, :]) / (2 * self.dy)

        dhu_dx_int = dhu_dx[:, 1:-1, :]
        dhv_dy_int = dhv_dy[:, :, 1:-1]

        R_mass = dhu_dx_int + dhv_dy_int

        h_int = h[:, 1:-1, 1:-1]
        u_int = u[:, 1:-1, 1:-1]
        v_int = v[:, 1:-1, 1:-1]
        n_int = n[:, 1:-1, 1:-1]

        dz_dx = (z_b[:, 1:-1, 2:] - z_b[:, 1:-1, :-2]) / (2 * self.dx)
        dz_dy = (z_b[:, 2:, 1:-1] - z_b[:, :-2, 1:-1]) / (2 * self.dy)

        speed = torch.sqrt(u_int**2 + v_int**2 + eps)
        h_safe = torch.clamp(h_int, min=eps)
        friction_coeff = self.g * n_int**2 * speed / (h_safe ** (1.0 / 3.0))
        tau_x = friction_coeff * u_int
        tau_y = friction_coeff * v_int

        dh_dx = (h[:, 1:-1, 2:] - h[:, 1:-1, :-2]) / (2 * self.dx)
        pressure_x = self.g * h_int * dh_dx
        slope_x = self.g * h_int * dz_dx

        hu2 = h * u * u
        dhu2_dx = (hu2[:, 1:-1, 2:] - hu2[:, 1:-1, :-2]) / (2 * self.dx)
        huv = h * u * v
        dhuv_dy = (huv[:, 2:, 1:-1] - huv[:, :-2, 1:-1]) / (2 * self.dy)

        R_mom_x = dhu2_dx + dhuv_dy + pressure_x + slope_x + tau_x

        dh_dy = (h[:, 2:, 1:-1] - h[:, :-2, 1:-1]) / (2 * self.dy)
        pressure_y = self.g * h_int * dh_dy
        slope_y = self.g * h_int * dz_dy

        hv2 = h * v * v
        dhv2_dy = (hv2[:, 2:, 1:-1] - hv2[:, :-2, 1:-1]) / (2 * self.dy)
        dhuv_dx = (huv[:, 1:-1, 2:] - huv[:, 1:-1, :-2]) / (2 * self.dx)

        R_mom_y = dhuv_dx + dhv2_dy + pressure_y + slope_y + tau_y

        wet_mask = (h_int > 0.01).float()

        loss_mass = torch.mean((R_mass * wet_mask) ** 2)
        loss_mom_x = torch.mean((R_mom_x * wet_mask) ** 2)
        loss_mom_y = torch.mean((R_mom_y * wet_mask) ** 2)

        return (
            self.mass_weight * loss_mass
            + self.momentum_weight * (loss_mom_x + loss_mom_y)
        )


class UnsteadyPhysicsLoss(nn.Module):
    """Compute unsteady 2D SWE residuals between consecutive time steps `t_prev` and `t_curr`.

    Parameters
    ----------
    gravity : float
        Gravitational acceleration (m/s²).
    dx, dy : float
        Spatial grid spacing (m).
    dt : float
        Temporal step duration between frames (s).
    """

    def __init__(
        self,
        gravity: float = 9.81,
        dx: float = 10.0,
        dy: float = 10.0,
        dt: float = 3600.0,
        mass_weight: float = 1.0,
        momentum_weight: float = 1.0,
    ):
        super().__init__()
        self.g = gravity
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.mass_weight = mass_weight
        self.momentum_weight = momentum_weight

    def forward(
        self,
        h_prev: torch.Tensor,
        u_prev: torch.Tensor,
        v_prev: torch.Tensor,
        h_curr: torch.Tensor,
        u_curr: torch.Tensor,
        v_curr: torch.Tensor,
        z_b: torch.Tensor,
        n_manning: torch.Tensor,
    ) -> torch.Tensor:
        """Compute unsteady physics residuals.

        Returns
        -------
        torch.Tensor
            Scalar unsteady physics loss.
        """
        eps = 1e-6

        h0 = h_prev.squeeze(1)
        u0 = u_prev.squeeze(1)
        v0 = v_prev.squeeze(1)
        h1 = h_curr.squeeze(1)
        u1 = u_curr.squeeze(1)
        v1 = v_curr.squeeze(1)
        z_b = z_b.squeeze(1)
        n = n_manning.squeeze(1)

        # Temporal derivatives (backward differences in interior slice)
        dh_dt = (h1[:, 1:-1, 1:-1] - h0[:, 1:-1, 1:-1]) / self.dt
        dhu_dt = (h1[:, 1:-1, 1:-1] * u1[:, 1:-1, 1:-1] - h0[:, 1:-1, 1:-1] * u0[:, 1:-1, 1:-1]) / self.dt
        dhv_dt = (h1[:, 1:-1, 1:-1] * v1[:, 1:-1, 1:-1] - h0[:, 1:-1, 1:-1] * v0[:, 1:-1, 1:-1]) / self.dt

        # Spatial derivatives evaluated at current step t_curr
        hu1 = h1 * u1
        dhu_dx = (hu1[:, :, 2:] - hu1[:, :, :-2]) / (2 * self.dx)
        hv1 = h1 * v1
        dhv_dy = (hv1[:, 2:, :] - hv1[:, :-2, :]) / (2 * self.dy)

        dhu_dx_int = dhu_dx[:, 1:-1, :]
        dhv_dy_int = dhv_dy[:, :, 1:-1]

        # MASS RESIDUAL
        R_mass = dh_dt + dhu_dx_int + dhv_dy_int

        h_int = h1[:, 1:-1, 1:-1]
        u_int = u1[:, 1:-1, 1:-1]
        v_int = v1[:, 1:-1, 1:-1]
        n_int = n[:, 1:-1, 1:-1]

        dz_dx = (z_b[:, 1:-1, 2:] - z_b[:, 1:-1, :-2]) / (2 * self.dx)
        dz_dy = (z_b[:, 2:, 1:-1] - z_b[:, :-2, 1:-1]) / (2 * self.dy)

        speed = torch.sqrt(u_int**2 + v_int**2 + eps)
        h_safe = torch.clamp(h_int, min=eps)
        friction_coeff = self.g * n_int**2 * speed / (h_safe ** (1.0 / 3.0))
        tau_x = friction_coeff * u_int
        tau_y = friction_coeff * v_int

        dh_dx = (h1[:, 1:-1, 2:] - h1[:, 1:-1, :-2]) / (2 * self.dx)
        pressure_x = self.g * h_int * dh_dx
        slope_x = self.g * h_int * dz_dx

        hu2 = h1 * u1 * u1
        dhu2_dx = (hu2[:, 1:-1, 2:] - hu2[:, 1:-1, :-2]) / (2 * self.dx)
        huv = h1 * u1 * v1
        dhuv_dy = (huv[:, 2:, 1:-1] - huv[:, :-2, 1:-1]) / (2 * self.dy)

        # X-MOMENTUM RESIDUAL
        R_mom_x = dhu_dt + dhu2_dx + dhuv_dy + pressure_x + slope_x + tau_x

        dh_dy = (h1[:, 2:, 1:-1] - h1[:, :-2, 1:-1]) / (2 * self.dy)
        pressure_y = self.g * h_int * dh_dy
        slope_y = self.g * h_int * dz_dy

        hv2 = h1 * v1 * v1
        dhv2_dy = (hv2[:, 2:, 1:-1] - hv2[:, :-2, 1:-1]) / (2 * self.dy)
        dhuv_dx = (huv[:, 1:-1, 2:] - huv[:, 1:-1, :-2]) / (2 * self.dx)

        # Y-MOMENTUM RESIDUAL
        R_mom_y = dhv_dt + dhuv_dx + dhv2_dy + pressure_y + slope_y + tau_y

        wet_mask = (h_int > 0.01).float()

        loss_mass = torch.mean((R_mass * wet_mask) ** 2)
        loss_mom_x = torch.mean((R_mom_x * wet_mask) ** 2)
        loss_mom_y = torch.mean((R_mom_y * wet_mask) ** 2)

        return (
            self.mass_weight * loss_mass
            + self.momentum_weight * (loss_mom_x + loss_mom_y)
        )


class NonNegativityLoss(nn.Module):
    """Penalize negative water depths."""

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.relu(-h) ** 2)


class DryConsistencyLoss(nn.Module):
    """Penalize non-zero velocities in dry cells."""

    def __init__(self, depth_threshold: float = 0.01):
        super().__init__()
        self.threshold = depth_threshold

    def forward(
        self, h: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        dry_mask = (h < self.threshold).float()
        return torch.mean(dry_mask * (u**2 + v**2))
