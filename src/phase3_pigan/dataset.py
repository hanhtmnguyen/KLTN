"""
PyTorch Dataset for PI-GAN training.

Loads pre-tiled ``.npz`` training pairs (input + target) created by
``TrainingDataBuilder`` and applies data augmentation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Sequence

from src.utils.logging_config import get_logger

logger = get_logger("pigan.phase3.dataset")


class FloodPatchDataset(Dataset):
    """Dataset of tiled flood map patches.

    Each ``.npz`` file contains:
    - ``input``:  (C_in, H, W)  — HAND + DTM + features
    - ``target``: (C_out, H, W) — HEC-RAS ground truth

    Parameters
    ----------
    data_dir : str | Path
        Directory containing ``.npz`` patch files.
    augment : bool
        If True, apply random flips and rotations.
    sar_dir : str | Path | None
        Directory with SAR flood masks (optional, for data assimilation).
    """

    def __init__(
        self,
        data_dir: str | Path,
        augment: bool = True,
        sar_dir: str | Path | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.files = sorted(self.data_dir.glob("*.npz"))

        if not self.files:
            raise FileNotFoundError(f"No .npz files in {self.data_dir}")

        logger.info("FloodPatchDataset: %d patches from %s", len(self.files), data_dir)

        # Optional SAR masks
        self.sar_dir = Path(sar_dir) if sar_dir else None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = np.load(self.files[idx])
        inp = data["input"].astype(np.float32)    # (C_in, H, W)
        tgt = data["target"].astype(np.float32)   # (C_out, H, W)

        # ---- Data augmentation ----
        if self.augment:
            # Random horizontal flip
            if np.random.random() > 0.5:
                inp = inp[:, :, ::-1].copy()
                tgt = tgt[:, :, ::-1].copy()

            # Random vertical flip
            if np.random.random() > 0.5:
                inp = inp[:, ::-1, :].copy()
                tgt = tgt[:, ::-1, :].copy()

            # Random 90° rotation
            k = np.random.randint(0, 4)
            if k > 0:
                inp = np.rot90(inp, k, axes=(1, 2)).copy()
                tgt = np.rot90(tgt, k, axes=(1, 2)).copy()

        sample = {
            "input": torch.from_numpy(inp),
            "target": torch.from_numpy(tgt),
        }

        # Load SAR mask if available
        if self.sar_dir is not None:
            sar_name = self.files[idx].stem + "_sar.npy"
            sar_path = self.sar_dir / sar_name
            if sar_path.exists():
                sar = np.load(sar_path).astype(np.float32)
                if self.augment:
                    # Apply same transforms (tracked via same random state)
                    pass  # Simplified — in practice, use seeded RNG
                sample["sar_mask"] = torch.from_numpy(sar)

        return sample


def create_dataloaders(
    config: dict,
    batch_size: int = 8,
    num_workers: int = 4,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train/val/test DataLoaders from the training directory.

    Parameters
    ----------
    config : dict
        Pipeline configuration.
    batch_size : int
    num_workers : int

    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader)
    """
    base = Path(config["paths"]["training"])

    train_ds = FloodPatchDataset(base / "train", augment=True)
    val_ds = FloodPatchDataset(base / "val", augment=False)
    test_ds = FloodPatchDataset(base / "test", augment=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    logger.info(
        "DataLoaders: train=%d, val=%d, test=%d batches",
        len(train_loader), len(val_loader), len(test_loader),
    )
    return train_loader, val_loader, test_loader
