"""
PyTorch Dataset for Spatiotemporal Supervised PI-GAN with Data Assimilation.

Loads pre-tiled `.npz` training patches containing physical inputs, HEC-RAS 2D
ground truth targets, and data assimilation boundary constraints (`upstream_mask`,
`downstream_mask`, `upstream_Q`, `downstream_wse`, `sar_mask`).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from src.utils.logging_config import get_logger

logger = get_logger("da.model.dataset")


class FloodPatchDataset(Dataset):
    """Dataset of tiled flood map patches with boundary condition data.

    Each `.npz` patch contains:
    - `input`: `(C_in, H, W)` — physical terrain/HAND channels
    - `target`: `(C_out, H, W)` — HEC-RAS supervised targets
    - `upstream_mask`: `(H, W)` binary mask
    - `downstream_mask`: `(H, W)` binary mask
    - `upstream_Q`: `(T,)` GEOGloWS discharge time series
    - `downstream_wse`: `(N_obs, 4)` `[t_idx, row, col, wse]` ATL13 observations
    - `sar_mask`: `(H, W)` binary SAR flood extent

    Parameters
    ----------
    data_dir : str | Path
        Directory containing `.npz` patch files.
    augment : bool
        If True, apply random geometric augmentations.
    """

    def __init__(self, data_dir: str | Path, augment: bool = True):
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.files = sorted(self.data_dir.glob("*.npz"))

        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {self.data_dir}")

        logger.info("FloodPatchDataset: %d patches loaded from %s", len(self.files), data_dir)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = np.load(self.files[idx], allow_pickle=True)
        inp = data["input"].astype(np.float32)
        tgt = data["target"].astype(np.float32)

        up_mask = data["upstream_mask"].astype(np.float32) if "upstream_mask" in data else np.zeros(inp.shape[1:], dtype=np.float32)
        down_mask = data["downstream_mask"].astype(np.float32) if "downstream_mask" in data else np.zeros(inp.shape[1:], dtype=np.float32)
        up_q = data["upstream_Q"].astype(np.float32) if "upstream_Q" in data else np.array([2500.0], dtype=np.float32)
        down_wse = data["downstream_wse"].astype(np.float32) if "downstream_wse" in data else np.empty((0, 4), dtype=np.float32)
        sar = data["sar_mask"].astype(np.float32) if "sar_mask" in data else np.empty((0, 0), dtype=np.float32)

        if self.augment:
            if np.random.random() > 0.5:
                inp = inp[:, :, ::-1].copy()
                tgt = tgt[:, :, ::-1].copy()
                up_mask = up_mask[:, ::-1].copy()
                down_mask = down_mask[:, ::-1].copy()
                if sar.size > 0 and sar.ndim == 2:
                    sar = sar[:, ::-1].copy()
                if down_wse.shape[0] > 0 and down_wse.shape[1] >= 3:
                    down_wse = down_wse.copy()
                    down_wse[:, 2] = inp.shape[2] - 1 - down_wse[:, 2]

            if np.random.random() > 0.5:
                inp = inp[:, ::-1, :].copy()
                tgt = tgt[:, ::-1, :].copy()
                up_mask = up_mask[::-1, :].copy()
                down_mask = down_mask[::-1, :].copy()
                if sar.size > 0 and sar.ndim == 2:
                    sar = sar[::-1, :].copy()
                if down_wse.shape[0] > 0 and down_wse.shape[1] >= 2:
                    down_wse = down_wse.copy()
                    down_wse[:, 1] = inp.shape[1] - 1 - down_wse[:, 1]

        sample = {
            "input": torch.from_numpy(inp),
            "target": torch.from_numpy(tgt),
            "upstream_mask": torch.from_numpy(up_mask),
            "downstream_mask": torch.from_numpy(down_mask),
            "upstream_Q": torch.from_numpy(up_q),
            "downstream_wse": torch.from_numpy(down_wse),
        }

        if sar.size > 0 and sar.ndim == 2:
            sample["sar_mask"] = torch.from_numpy(sar).unsqueeze(0)

        return sample


def create_dataloaders(
    config: dict,
    batch_size: int = 8,
    num_workers: int = 4,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train/val/test DataLoaders from the training directory."""
    base = Path(config["paths"]["training"])

    def _collate_fn(batch):
        # Custom collate because downstream_wse table can have varying row counts across tiles
        collated = {}
        for key in batch[0].keys():
            if key == "downstream_wse":
                collated[key] = [item[key] for item in batch]
            else:
                collated[key] = torch.stack([item[key] for item in batch], dim=0)
        return collated

    train_ds = FloodPatchDataset(base / "train", augment=True)
    val_ds = FloodPatchDataset(base / "val", augment=False)
    test_ds = FloodPatchDataset(base / "test", augment=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=_collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=_collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=_collate_fn,
    )

    logger.info("DataLoaders initialized: train=%d, val=%d, test=%d batches",
                len(train_loader), len(val_loader), len(test_loader))
    return train_loader, val_loader, test_loader
