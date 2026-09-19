"""
PI-GAN Inference — fast flood map super-resolution for new scenarios.

After training, this module takes a HAND flood map + DTM + features
and produces HEC-RAS-quality 10 m flood maps in under 5 minutes.
"""

from __future__ import annotations

import numpy as np
import torch
import rasterio
from rasterio.enums import Resampling
from pathlib import Path

from src.phase3_pigan.generator import PIGANGenerator
from src.utils.logging_config import get_logger
from src.utils.geo_utils import load_raster, save_raster

logger = get_logger("pigan.phase3.inference")


class PIGANInference:
    """Run trained PI-GAN for fast flood map generation.

    Parameters
    ----------
    config : dict
        Pipeline configuration.
    checkpoint_path : str | Path
        Path to trained model checkpoint.
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
        ).to(self.device)

        if checkpoint_path:
            self.load_model(checkpoint_path)

        self.generator.eval()

    def load_model(self, path: str | Path) -> None:
        """Load generator weights from a checkpoint."""
        path = Path(path)
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        if "generator" in ckpt:
            self.generator.load_state_dict(ckpt["generator"])
        else:
            self.generator.load_state_dict(ckpt)
        logger.info("Loaded PI-GAN model from %s", path)

    @torch.no_grad()
    def predict(
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
        tile_size: int = 256,
        overlap: int = 32,
    ) -> dict[str, Path]:
        """Run PI-GAN inference on a full-domain flood scenario.

        Tiles the domain into patches, runs the generator on each,
        then stitches the results back together.

        Parameters
        ----------
        hand_depth_path, hand_vu_path, hand_vv_path : Path
            HAND model outputs (input channels 0-2).
        dtm_path, manning_n_path : Path
            Static terrain inputs (channels 3-4).
        slope_path, twi_path, dist_river_path : Path
            Terrain features (channels 5-7).
        output_dir : str | Path
            Directory for output rasters.
        tile_size : int
        overlap : int

        Returns
        -------
        dict[str, Path]
            ``{"depth": Path, "velocity_u": Path, "velocity_v": Path}``
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Running PI-GAN inference...")

        # Load reference grid
        dtm, profile = load_raster(dtm_path)
        H, W = dtm.shape

        # Load and align all input channels
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
        ], axis=0)  # (8, H, W)

        # Output accumulation arrays
        out_sum = np.zeros((3, H, W), dtype=np.float64)
        out_count = np.zeros((H, W), dtype=np.float64)

        stride = tile_size - overlap
        n_tiles = 0

        for row in range(0, H - tile_size + 1, stride):
            for col in range(0, W - tile_size + 1, stride):
                tile = channels[:, row:row+tile_size, col:col+tile_size]
                tile_tensor = torch.from_numpy(tile).unsqueeze(0).to(self.device)

                pred = self.generator(tile_tensor)  # (1, 3, H, W)
                pred_np = pred.cpu().numpy()[0]     # (3, H, W)

                out_sum[:, row:row+tile_size, col:col+tile_size] += pred_np
                out_count[row:row+tile_size, col:col+tile_size] += 1.0
                n_tiles += 1

        # Handle remaining edges
        # Right edge
        if W > tile_size:
            for row in range(0, H - tile_size + 1, stride):
                col = W - tile_size
                tile = channels[:, row:row+tile_size, col:col+tile_size]
                tile_tensor = torch.from_numpy(tile).unsqueeze(0).to(self.device)
                pred = self.generator(tile_tensor).cpu().numpy()[0]
                out_sum[:, row:row+tile_size, col:col+tile_size] += pred
                out_count[row:row+tile_size, col:col+tile_size] += 1.0

        # Bottom edge
        if H > tile_size:
            row = H - tile_size
            for col in range(0, W - tile_size + 1, stride):
                tile = channels[:, row:row+tile_size, col:col+tile_size]
                tile_tensor = torch.from_numpy(tile).unsqueeze(0).to(self.device)
                pred = self.generator(tile_tensor).cpu().numpy()[0]
                out_sum[:, row:row+tile_size, col:col+tile_size] += pred
                out_count[row:row+tile_size, col:col+tile_size] += 1.0

        # Average overlapping tiles
        out_count = np.maximum(out_count, 1.0)
        result = (out_sum / out_count[np.newaxis, :, :]).astype(np.float32)

        # Ensure non-negative depth
        result[0] = np.maximum(result[0], 0.0)

        # Save outputs
        outputs = {}
        for i, name in enumerate(["depth", "velocity_u", "velocity_v"]):
            out_path = output_dir / f"{name}.tif"
            save_raster(out_path, result[i], profile)
            outputs[name] = out_path

        logger.info(
            "PI-GAN inference complete: %d tiles, max_depth=%.2f m",
            n_tiles, result[0].max(),
        )
        return outputs
