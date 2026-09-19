"""
Spatiotemporal PI-GAN Inference Engine.

Runs fast, continuous multi-step hydrodynamic predictions across `t ∈ [0, 1]`
and exports high-resolution 10m depth and velocity GeoTIFF time-series.
"""

from __future__ import annotations

import numpy as np
import torch
import rasterio
from rasterio.enums import Resampling
from pathlib import Path

from src.model.generator import PIGANGenerator
from src.utils.logging_config import get_logger
from src.utils.geo_utils import load_raster, save_raster

logger = get_logger("da.model.inference")


class PIGANInference:
    """Run trained Spatiotemporal PI-GAN for fast continuous flood simulation.

    Parameters
    ----------
    config : dict
        Pipeline configuration.
    checkpoint_path : str | Path | None
        Path to checkpoint weights.
    device : str
    """

    def __init__(
        self,
        config: dict,
        checkpoint_path: str | Path | None = None,
        device: str = "auto",
    ):
        self.config = config
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.generator = PIGANGenerator(
            in_physical_channels=8,
            out_channels=3,
            temporal=True,
        ).to(self.device)

        if checkpoint_path:
            self.load_model(checkpoint_path)

        self.generator.eval()

    def load_model(self, path: str | Path) -> None:
        """Load generator weights from checkpoint."""
        path = Path(path)
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        if "generator" in ckpt:
            self.generator.load_state_dict(ckpt["generator"])
        else:
            self.generator.load_state_dict(ckpt)
        logger.info("Loaded Spatiotemporal PI-GAN model from %s", path)

    @torch.no_grad()
    def predict_time_series(
        self,
        hand_depth_path: str | Path,
        hand_vu_path: str | Path,
        hand_vv_path: str | Path,
        dtm_path: str | Path,
        manning_n_path: str | Path,
        slope_path: str | Path,
        twi_path: str | Path,
        dist_river_path: str | Path,
        output_dir: str | Path,
        n_steps: int = 48,
        tile_size: int = 256,
        overlap: int = 32,
    ) -> list[dict[str, Path]]:
        """Predict hydrodynamic fields across multiple time steps.

        Returns
        -------
        list[dict[str, Path]]
            List of output file mappings (`{"depth": Path, ...}`) for each step.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Running Spatiotemporal PI-GAN inference across %d steps...", n_steps)

        dtm, profile = load_raster(dtm_path)
        H, W = dtm.shape

        def _load_align(path):
            with rasterio.open(path) as src:
                return src.read(1, out_shape=(H, W),
                                resampling=Resampling.bilinear).astype(np.float32)

        channels = np.stack([
            _load_align(hand_depth_path),
            _load_align(hand_vu_path),
            _load_align(hand_vv_path),
            dtm,
            _load_align(manning_n_path),
            _load_align(slope_path),
            _load_align(twi_path),
            _load_align(dist_river_path),
        ], axis=0)

        stride = tile_size - overlap
        step_outputs = []

        for t_idx in range(n_steps):
            t_norm = float(t_idx) / max(n_steps - 1, 1)
            step_dir = output_dir / f"step_{t_idx:03d}"
            step_dir.mkdir(parents=True, exist_ok=True)

            out_sum = np.zeros((3, H, W), dtype=np.float64)
            out_count = np.zeros((H, W), dtype=np.float64)

            for row in range(0, H - tile_size + 1, stride):
                for col in range(0, W - tile_size + 1, stride):
                    tile = channels[:, row:row+tile_size, col:col+tile_size]
                    tile_tensor = torch.from_numpy(tile).unsqueeze(0).to(self.device)

                    pred = self.generator(tile_tensor, time_norm=t_norm)
                    pred_np = pred.cpu().numpy()[0]

                    out_sum[:, row:row+tile_size, col:col+tile_size] += pred_np
                    out_count[row:row+tile_size, col:col+tile_size] += 1.0

            out_count = np.maximum(out_count, 1.0)
            result = (out_sum / out_count[np.newaxis, :, :]).astype(np.float32)
            result[0] = np.maximum(result[0], 0.0)

            outputs = {}
            for i, name in enumerate(["depth", "velocity_u", "velocity_v"]):
                out_path = step_dir / f"{name}.tif"
                save_raster(out_path, result[i], profile)
                outputs[name] = out_path

            step_outputs.append(outputs)
            logger.info("  Step %03d (t=%.2f): max_depth=%.2f m exported.", t_idx, t_norm, result[0].max())

        return step_outputs
