"""
Spatiotemporal Supervised PI-GAN Trainer with Data Assimilation.

Implements the multi-component supervised training loop combining:
- `L_pixel`: Supervised L1 loss against HEC-RAS 2D ground-truth flood maps.
- `L_physics`: Unsteady 2D Shallow Water Equations between consecutive time steps `t-1` and `t`.
- `L_BC`: Boundary condition data assimilation matching upstream GEOGloWS discharge Q(t)
  and downstream ICESat-2 ATL13 water surface elevations.
- `L_SAR` / `L_gauge`: In-situ gauge and Sentinel-1 SAR flood extent constraints.
- `L_adv`: PatchGAN adversarial loss for sharp local texture features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm

from src.model.generator import PIGANGenerator
from src.model.discriminator import PatchGANDiscriminator
from src.model.physics_loss import UnsteadyPhysicsLoss, DryConsistencyLoss
from src.model.data_assimilation import UnifiedDataAssimilationLoss
from src.utils.logging_config import get_logger

logger = get_logger("da.model.trainer")


class PIGANTrainer:
    """Full Spatiotemporal PI-GAN DA training loop.

    Parameters
    ----------
    config : dict
        Full pipeline configuration.
    device : str
        `"cuda"` or `"cpu"` or `"auto"`.
    """

    def __init__(self, config: dict, device: str = "auto"):
        self.config = config

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("Spatiotemporal PI-GAN Trainer initialized on %s", self.device)

        self.generator = PIGANGenerator(
            in_physical_channels=8,
            out_channels=3,
            fourier_mapping_size=256,
            fourier_scale=10.0,
            base_features=64,
            temporal=True,
        ).to(self.device)

        self.discriminator = PatchGANDiscriminator(
            in_channels=3,
            base_features=64,
        ).to(self.device)

        self.l1_loss = nn.L1Loss()
        self.bce_logits = nn.BCEWithLogitsLoss()
        self.unsteady_physics_loss = UnsteadyPhysicsLoss(
            gravity=config["physics"]["gravity"],
            dx=config["resolutions"]["fine"],
            dy=config["resolutions"]["fine"],
            dt=config["temporal"]["dt_hours"] * 3600.0,
        ).to(self.device)
        self.dry_loss = DryConsistencyLoss().to(self.device)
        self.da_loss = UnifiedDataAssimilationLoss(
            cell_width=config["resolutions"]["fine"],
            weights={
                "upstream": config["training"]["loss_weights"]["upstream_bc"],
                "downstream": config["training"]["loss_weights"]["downstream_bc"],
                "gauge": config["training"]["loss_weights"]["gauge_station"],
                "sar": config["training"]["loss_weights"]["sar_extent"],
            },
        ).to(self.device)

        # Loss weights
        lw = config["training"]["loss_weights"]
        self.lambda_pixel = lw["pixel_supervision"]
        self.lambda_adv = lw["adversarial"]
        self.lambda_physics = lw["physics_swe"]

        self.opt_G = torch.optim.Adam(self.generator.parameters(), lr=config["training"]["learning_rate_g"], betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=config["training"]["learning_rate_d"], betas=(0.5, 0.999))

        self.sched_G = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.opt_G, T_0=50, T_mult=2)
        self.sched_D = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.opt_D, T_0=50, T_mult=2)

        self.model_dir = Path(config["paths"]["models"]) / "pigan_da"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(config["paths"]["logs"]) / "pigan_da"
        self.writer = SummaryWriter(str(self.log_dir))
        self.n_time_steps = config["temporal"]["n_time_steps"]

    def _get_curriculum_phase(self, epoch: int) -> str:
        """Determine curriculum phase (`A`: Supervised warmup, `B`: +Physics/BC, `C`: Full GAN)."""
        if epoch <= self.config["training"]["curriculum_epochs"]["phase1_pixel_only"]:
            return "A"
        elif epoch <= self.config["training"]["curriculum_epochs"]["phase2_add_physics_da"]:
            return "B"
        else:
            return "C"

    def train(
        self,
        train_loader,
        val_loader,
        total_epochs: int = 500,
        save_every: int = 25,
    ) -> dict[str, list[float]]:
        """Run the multi-epoch training loop."""
        history = {
            "G_total": [], "G_pixel": [], "G_physics": [],
            "G_DA": [], "G_adv": [], "D_loss": [], "val_pixel": [],
        }
        best_val_loss = float("inf")

        for epoch in range(1, total_epochs + 1):
            phase = self._get_curriculum_phase(epoch)
            g_losses = self._train_epoch(train_loader, phase)
            for key, val in g_losses.items():
                if key in history:
                    history[key].append(val)

            val_loss = self._validate(val_loader)
            history["val_pixel"].append(val_loss)

            self.writer.add_scalar("Loss/G_total", g_losses["G_total"], epoch)
            self.writer.add_scalar("Loss/G_pixel", g_losses["G_pixel"], epoch)
            self.writer.add_scalar("Loss/G_physics", g_losses["G_physics"], epoch)
            self.writer.add_scalar("Loss/G_DA", g_losses["G_DA"], epoch)
            self.writer.add_scalar("Loss/val_pixel", val_loss, epoch)

            self.sched_G.step()
            self.sched_D.step()

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d [Phase %s] — G_total=%.4f, pixel=%.4f, "
                    "physics=%.4f, da=%.4f, val=%.4f",
                    epoch, total_epochs, phase,
                    g_losses["G_total"], g_losses["G_pixel"],
                    g_losses["G_physics"], g_losses["G_DA"], val_loss,
                )

            if epoch % save_every == 0:
                self._save_checkpoint(f"epoch_{epoch:04d}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint("best")

        self.writer.close()
        logger.info("Training finished. Best val loss: %.4f", best_val_loss)
        return history

    def _train_epoch(self, loader, phase: str) -> dict[str, float]:
        self.generator.train()
        self.discriminator.train()

        running = {"G_total": 0, "G_pixel": 0, "G_physics": 0, "G_DA": 0, "G_adv": 0, "D_loss": 0}
        n_batches = 0

        for batch in loader:
            inp = batch["input"].to(self.device)
            tgt = batch["target"].to(self.device)
            up_mask = batch["upstream_mask"].to(self.device)
            q_series = batch["upstream_Q"].to(self.device)
            down_wse_list = batch.get("downstream_wse", [])
            sar = batch.get("sar_mask")
            if sar is not None:
                sar = sar.to(self.device)

            # Pick two consecutive time steps or random time step for spatiotemporal training
            t_idx = torch.randint(0, self.n_time_steps, (1,)).item()
            t_norm_curr = float(t_idx) / max(self.n_time_steps - 1, 1)

            fake_curr = self.generator(inp, time_norm=t_norm_curr)

            d_loss = torch.tensor(0.0, device=self.device)
            if phase == "C":
                self.opt_D.zero_grad()
                real_pred = self.discriminator(tgt)
                fake_pred = self.discriminator(fake_curr.detach())
                d_real = self.bce_logits(real_pred, torch.ones_like(real_pred))
                d_fake = self.bce_logits(fake_pred, torch.zeros_like(fake_pred))
                d_loss = (d_real + d_fake) * 0.5
                d_loss.backward()
                self.opt_D.step()

            self.opt_G.zero_grad()
            loss_pixel = self.l1_loss(fake_curr, tgt)
            g_loss = self.lambda_pixel * loss_pixel

            loss_physics = torch.tensor(0.0, device=self.device)
            loss_da_total = torch.tensor(0.0, device=self.device)

            if phase in ("B", "C"):
                # Evaluate unsteady physics across t_prev and t_curr
                t_idx_prev = max(0, t_idx - 1)
                t_norm_prev = float(t_idx_prev) / max(self.n_time_steps - 1, 1)
                if t_idx_prev != t_idx:
                    fake_prev = self.generator(inp, time_norm=t_norm_prev)
                else:
                    fake_prev = fake_curr

                h0, u0, v0 = fake_prev[:, 0:1, :, :], fake_prev[:, 1:2, :, :], fake_prev[:, 2:3, :, :]
                h1, u1, v1 = fake_curr[:, 0:1, :, :], fake_curr[:, 1:2, :, :], fake_curr[:, 2:3, :, :]
                dtm = inp[:, 3:4, :, :]
                n_m = inp[:, 4:5, :, :]

                loss_physics = self.unsteady_physics_loss(h0, u0, v0, h1, u1, v1, dtm, n_m)
                loss_dry = self.dry_loss(h1, u1, v1)
                loss_physics = loss_physics + 0.1 * loss_dry
                g_loss = g_loss + self.lambda_physics * loss_physics

                # Data Assimilation constraints (upstream Q, downstream ATL13 WSE, SAR)
                q_target = q_series[:, min(t_idx, q_series.shape[1] - 1)] if q_series.dim() > 1 else q_series

                # Filter downstream observations matching current time step
                filtered_down_obs = []
                for obs_table in down_wse_list:
                    if isinstance(obs_table, torch.Tensor) and obs_table.shape[0] > 0:
                        matching = obs_table[obs_table[:, 0] == t_idx]
                        if matching.shape[0] > 0:
                            filtered_down_obs.append(matching[:, 1:])  # [row, col, wse]
                        else:
                            filtered_down_obs.append(torch.empty((0, 3), device=self.device))
                    else:
                        filtered_down_obs.append(torch.empty((0, 3), device=self.device))

                da_dict = self.da_loss(
                    h=h1, u=u1, v=v1, z_b=dtm,
                    upstream_mask=up_mask, q_target=q_target,
                    downstream_obs=filtered_down_obs, sar_mask=sar
                )
                loss_da_total = da_dict["total"]
                g_loss = g_loss + loss_da_total

            loss_adv = torch.tensor(0.0, device=self.device)
            if phase == "C":
                fake_pred = self.discriminator(fake_curr)
                loss_adv = self.bce_logits(fake_pred, torch.ones_like(fake_pred))
                g_loss = g_loss + self.lambda_adv * loss_adv

            g_loss.backward()
            self.opt_G.step()

            running["G_total"] += g_loss.item()
            running["G_pixel"] += loss_pixel.item()
            running["G_physics"] += loss_physics.item()
            running["G_DA"] += loss_da_total.item()
            running["G_adv"] += loss_adv.item()
            running["D_loss"] += d_loss.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    @torch.no_grad()
    def _validate(self, loader) -> float:
        self.generator.eval()
        total_loss = 0
        n = 0
        for batch in loader:
            inp = batch["input"].to(self.device)
            tgt = batch["target"].to(self.device)
            fake = self.generator(inp, time_norm=1.0)
            total_loss += self.l1_loss(fake, tgt).item()
            n += 1
        return total_loss / max(n, 1)

    def _save_checkpoint(self, tag: str) -> None:
        path = self.model_dir / f"pigan_{tag}.pt"
        torch.save({
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "opt_G": self.opt_G.state_dict(),
            "opt_D": self.opt_D.state_dict(),
        }, path)

    def load_checkpoint(self, tag: str = "best") -> None:
        path = self.model_dir / f"pigan_{tag}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.generator.load_state_dict(ckpt["generator"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        logger.info("Loaded checkpoint: %s", path)
