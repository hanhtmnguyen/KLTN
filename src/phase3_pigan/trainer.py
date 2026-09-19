"""
PI-GAN Trainer — complete training loop with multi-component loss.

Implements the 3-phase curriculum:
  Phase A (warm-up):    L_pixel only
  Phase B (physics):    L_pixel + L_physics + L_gauge
  Phase C (full GAN):   L_pixel + L_adv + L_physics + L_gauge + L_SAR
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm

from src.phase3_pigan.generator import PIGANGenerator
from src.phase3_pigan.discriminator import PatchGANDiscriminator
from src.phase3_pigan.physics_loss import (
    PhysicsInformedLoss,
    DryConsistencyLoss,
)
from src.phase3_pigan.data_assimilation import GaugeStationLoss, SARFloodExtentLoss
from src.utils.logging_config import get_logger

logger = get_logger("pigan.phase3.trainer")


class PIGANTrainer:
    """Full PI-GAN training loop.

    Parameters
    ----------
    config : dict
        Full pipeline configuration.
    device : str
        ``"cuda"`` or ``"cpu"`` or ``"auto"``.
    """

    def __init__(self, config: dict, device: str = "auto"):
        self.config = config

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("PI-GAN Trainer on %s", self.device)

        # ---- Models ----
        self.generator = PIGANGenerator(
            in_physical_channels=8,
            out_channels=3,
            fourier_mapping_size=256,
            fourier_scale=10.0,
            base_features=64,
        ).to(self.device)

        self.discriminator = PatchGANDiscriminator(
            in_channels=3,
            base_features=64,
        ).to(self.device)

        # ---- Loss functions ----
        self.l1_loss = nn.L1Loss()
        self.bce_logits = nn.BCEWithLogitsLoss()
        self.physics_loss = PhysicsInformedLoss(
            gravity=config["physics"]["gravity"],
            dx=config["resolutions"]["fine"],
            dy=config["resolutions"]["fine"],
        ).to(self.device)
        self.dry_loss = DryConsistencyLoss().to(self.device)
        self.gauge_loss = GaugeStationLoss()
        self.sar_loss = SARFloodExtentLoss().to(self.device)

        # ---- Loss weights (from implementation plan) ----
        self.lambda_pixel = 1.0
        self.lambda_adv = 0.01
        self.lambda_physics = 0.1
        self.lambda_gauge = 0.5
        self.lambda_sar = 0.1

        # ---- Optimisers ----
        self.opt_G = torch.optim.Adam(
            self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999)
        )
        self.opt_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999)
        )

        # ---- Schedulers ----
        self.sched_G = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.opt_G, T_0=50, T_mult=2
        )
        self.sched_D = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.opt_D, T_0=50, T_mult=2
        )

        # ---- Checkpoints & logging ----
        self.model_dir = Path(config["paths"]["models"]) / "pigan"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(config["paths"]["logs"]) / "pigan"
        self.writer = SummaryWriter(str(self.log_dir))

    def _get_curriculum_phase(self, epoch: int) -> str:
        """Determine the curriculum training phase."""
        if epoch <= 50:
            return "A"   # Warm-up: L_pixel only
        elif epoch <= 150:
            return "B"   # Physics: L_pixel + L_physics + L_gauge
        else:
            return "C"   # Full GAN: all losses

    def train(
        self,
        train_loader,
        val_loader,
        total_epochs: int = 500,
        save_every: int = 25,
    ) -> dict[str, list[float]]:
        """Run the full training loop.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader : DataLoader
        total_epochs : int
        save_every : int

        Returns
        -------
        dict
            Loss history per component.
        """
        history = {
            "G_total": [], "G_pixel": [], "G_physics": [],
            "G_adv": [], "D_loss": [], "val_pixel": [],
        }

        best_val_loss = float("inf")

        for epoch in range(1, total_epochs + 1):
            phase = self._get_curriculum_phase(epoch)

            # ---- Train one epoch ----
            g_losses = self._train_epoch(train_loader, phase)
            for key, val in g_losses.items():
                if key in history:
                    history[key].append(val)

            # ---- Validate ----
            val_loss = self._validate(val_loader)
            history["val_pixel"].append(val_loss)

            # ---- Logging ----
            self.writer.add_scalar("Loss/G_total", g_losses["G_total"], epoch)
            self.writer.add_scalar("Loss/G_pixel", g_losses["G_pixel"], epoch)
            self.writer.add_scalar("Loss/G_physics", g_losses["G_physics"], epoch)
            self.writer.add_scalar("Loss/val_pixel", val_loss, epoch)
            self.writer.add_scalar("Phase", ord(phase) - ord("A"), epoch)

            self.sched_G.step()
            self.sched_D.step()

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d [Phase %s] — G_total=%.4f, pixel=%.4f, "
                    "physics=%.4f, val=%.4f",
                    epoch, total_epochs, phase,
                    g_losses["G_total"], g_losses["G_pixel"],
                    g_losses["G_physics"], val_loss,
                )

            # Save checkpoints
            if epoch % save_every == 0:
                self._save_checkpoint(f"epoch_{epoch:04d}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint("best")

        self.writer.close()
        logger.info("Training complete. Best val loss: %.4f", best_val_loss)
        return history

    def _train_epoch(self, loader, phase: str) -> dict[str, float]:
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()

        running = {"G_total": 0, "G_pixel": 0, "G_physics": 0, "G_adv": 0, "D_loss": 0}
        n_batches = 0

        for batch in loader:
            inp = batch["input"].to(self.device)     # (B, C_in, H, W)
            tgt = batch["target"].to(self.device)    # (B, C_out, H, W)
            sar = batch.get("sar_mask")
            if sar is not None:
                sar = sar.to(self.device)

            fake = self.generator(inp)  # (B, 3, H, W)

            # ---- Discriminator update (Phase C only) ----
            d_loss = torch.tensor(0.0, device=self.device)
            if phase == "C":
                self.opt_D.zero_grad()
                real_pred = self.discriminator(tgt)
                fake_pred = self.discriminator(fake.detach())
                d_real = self.bce_logits(real_pred, torch.ones_like(real_pred))
                d_fake = self.bce_logits(fake_pred, torch.zeros_like(fake_pred))
                d_loss = (d_real + d_fake) * 0.5
                d_loss.backward()
                self.opt_D.step()

            # ---- Generator update ----
            self.opt_G.zero_grad()

            # L_pixel — always active
            loss_pixel = self.l1_loss(fake, tgt)
            g_loss = self.lambda_pixel * loss_pixel

            # L_physics — active in Phase B and C
            loss_physics = torch.tensor(0.0, device=self.device)
            if phase in ("B", "C"):
                h_pred = fake[:, 0:1, :, :]
                u_pred = fake[:, 1:2, :, :]
                v_pred = fake[:, 2:3, :, :]
                dtm = inp[:, 3:4, :, :]      # Channel 3 = DTM
                n_m = inp[:, 4:5, :, :]       # Channel 4 = Manning's n

                loss_physics = self.physics_loss(h_pred, u_pred, v_pred, dtm, n_m)
                loss_dry = self.dry_loss(h_pred, u_pred, v_pred)
                loss_physics = loss_physics + 0.1 * loss_dry
                g_loss = g_loss + self.lambda_physics * loss_physics

            # L_adv — active in Phase C
            loss_adv = torch.tensor(0.0, device=self.device)
            if phase == "C":
                fake_pred = self.discriminator(fake)
                loss_adv = self.bce_logits(fake_pred, torch.ones_like(fake_pred))
                g_loss = g_loss + self.lambda_adv * loss_adv

            # L_SAR — active in Phase C (if available)
            if phase == "C" and sar is not None:
                loss_sar = self.sar_loss(fake[:, 0:1, :, :], sar)
                g_loss = g_loss + self.lambda_sar * loss_sar

            g_loss.backward()
            self.opt_G.step()

            running["G_total"] += g_loss.item()
            running["G_pixel"] += loss_pixel.item()
            running["G_physics"] += loss_physics.item()
            running["G_adv"] += loss_adv.item()
            running["D_loss"] += d_loss.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    @torch.no_grad()
    def _validate(self, loader) -> float:
        """Compute validation pixel loss."""
        self.generator.eval()
        total_loss = 0
        n = 0
        for batch in loader:
            inp = batch["input"].to(self.device)
            tgt = batch["target"].to(self.device)
            fake = self.generator(inp)
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
        logger.debug("Saved checkpoint: %s", path)

    def load_checkpoint(self, tag: str = "best") -> None:
        path = self.model_dir / f"pigan_{tag}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.generator.load_state_dict(ckpt["generator"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        logger.info("Loaded checkpoint: %s", path)
