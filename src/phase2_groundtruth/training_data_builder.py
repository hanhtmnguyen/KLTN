"""
Step ⑪ — Training data builder.

Aligns HAND outputs (input), HEC-RAS outputs (target), DTM, Manning's n,
and terrain features onto the same 10 m grid and creates tiled training
pairs for the PI-GAN.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.enums import Resampling
from pathlib import Path
from typing import Sequence
import json

from src.utils.logging_config import get_logger
from src.utils.geo_utils import load_raster, save_raster

logger = get_logger("pigan.phase2.builder")


class TrainingDataBuilder:
    """Build aligned training pairs from HAND (input) and HEC-RAS (target).

    Parameters
    ----------
    config : dict
        Full pipeline configuration.
    """

    def __init__(self, config: dict):
        self.config = config
        self.coarse_res = config["resolutions"]["coarse"]
        self.fine_res = config["resolutions"]["fine"]
        self.training_dir = Path(config["paths"]["training"])
        self.training_dir.mkdir(parents=True, exist_ok=True)

    def build_event_pair(
        self,
        event_name: str,
        hand_depth_path: str | Path,
        hand_vu_path: str | Path,
        hand_vv_path: str | Path,
        hecras_depth_path: str | Path,
        dtm_path: str | Path,
        manning_n_path: str | Path,
        terrain_feature_paths: dict[str, str | Path],
        hecras_vu_path: str | Path | None = None,
        hecras_vv_path: str | Path | None = None,
    ) -> Path:
        """Create an aligned training pair for one flood event.

        All inputs are resampled/aligned to the 10 m DTM grid and saved
        as multi-band GeoTIFFs.

        Parameters
        ----------
        event_name : str
        hand_depth_path, hand_vu_path, hand_vv_path : Path
            HAND model outputs (may be at 30m — will be upsampled).
        hecras_depth_path : Path
            HEC-RAS depth at 10m (target).
        dtm_path, manning_n_path : Path
            Static inputs (already at 10m).
        terrain_feature_paths : dict
            ``{"slope": Path, "twi": Path, "distance_to_river": Path, ...}``
        hecras_vu_path, hecras_vv_path : Path | None
            HEC-RAS velocity components (target). Optional.

        Returns
        -------
        Path
            Directory containing the event training files.
        """
        event_dir = self.training_dir / event_name
        event_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Building training pair: %s", event_name)

        # Load reference grid from DTM
        dtm, ref_profile = load_raster(dtm_path)
        h, w = dtm.shape

        # ---- Load and align HAND outputs (may need upsampling) ----
        hand_depth = self._load_and_align(hand_depth_path, h, w)
        hand_vu = self._load_and_align(hand_vu_path, h, w)
        hand_vv = self._load_and_align(hand_vv_path, h, w)

        # ---- Load HEC-RAS targets (should already be at 10m) ----
        hecras_depth = self._load_and_align(hecras_depth_path, h, w)

        # Stack input channels:
        # [0] HAND depth, [1] HAND vu, [2] HAND vv,
        # [3] DTM, [4] Manning's n, [5+] terrain features
        manning_n = self._load_and_align(manning_n_path, h, w)

        input_channels = [hand_depth, hand_vu, hand_vv, dtm, manning_n]
        channel_names = ["hand_depth", "hand_vu", "hand_vv", "dtm", "manning_n"]

        # Add terrain features
        for feat_name in ["slope", "twi", "distance_to_river"]:
            if feat_name in terrain_feature_paths:
                feat = self._load_and_align(terrain_feature_paths[feat_name], h, w)
                input_channels.append(feat)
                channel_names.append(feat_name)

        input_stack = np.stack(input_channels, axis=0)  # (C_in, H, W)

        # Stack target channels
        target_channels = [hecras_depth]
        target_names = ["hecras_depth"]

        if hecras_vu_path:
            target_channels.append(self._load_and_align(hecras_vu_path, h, w))
            target_names.append("hecras_vu")
        if hecras_vv_path:
            target_channels.append(self._load_and_align(hecras_vv_path, h, w))
            target_names.append("hecras_vv")

        target_stack = np.stack(target_channels, axis=0)  # (C_out, H, W)

        # Save stacked arrays
        input_profile = {**ref_profile, "count": input_stack.shape[0], "dtype": "float32"}
        target_profile = {**ref_profile, "count": target_stack.shape[0], "dtype": "float32"}

        save_raster(event_dir / "input.tif", input_stack, input_profile)
        save_raster(event_dir / "target.tif", target_stack, target_profile)

        # Save metadata
        metadata = {
            "event_name": event_name,
            "input_channels": channel_names,
            "target_channels": target_names,
            "grid_shape": [h, w],
            "resolution_m": self.fine_res,
        }
        with open(event_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "  Input: %d channels (%dx%d), Target: %d channels",
            input_stack.shape[0], h, w, target_stack.shape[0],
        )
        return event_dir

    def create_tiles(
        self,
        tile_size: int = 256,
        overlap: int = 32,
        min_wet_fraction: float = 0.05,
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    ) -> dict[str, int]:
        """Tile all event pairs into train/val/test patches.

        Parameters
        ----------
        tile_size : int
            Patch size in pixels.
        overlap : int
            Overlap between adjacent tiles.
        min_wet_fraction : float
            Minimum fraction of wet pixels for a tile to be included.
        split_ratios : tuple
            (train, val, test) fractions.  Split is done **by event**.

        Returns
        -------
        dict
            ``{"train": n_tiles, "val": n_tiles, "test": n_tiles}``.
        """
        event_dirs = sorted([
            d for d in self.training_dir.iterdir()
            if d.is_dir() and (d / "input.tif").exists()
        ])

        if not event_dirs:
            raise FileNotFoundError("No event pairs found in %s" % self.training_dir)

        # Split events
        n = len(event_dirs)
        n_train = max(1, int(n * split_ratios[0]))
        n_val = max(1, int(n * split_ratios[1]))

        train_events = event_dirs[:n_train]
        val_events = event_dirs[n_train:n_train + n_val]
        test_events = event_dirs[n_train + n_val:]

        logger.info(
            "Event split: train=%d, val=%d, test=%d",
            len(train_events), len(val_events), len(test_events),
        )

        counts = {}
        for split_name, events in [
            ("train", train_events),
            ("val", val_events),
            ("test", test_events),
        ]:
            split_dir = self.training_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            tile_count = 0

            for event_dir in events:
                input_data, _ = load_raster(event_dir / "input.tif")
                # input_data is just band 1; load all bands
                with rasterio.open(event_dir / "input.tif") as src:
                    input_all = src.read().astype(np.float32)
                with rasterio.open(event_dir / "target.tif") as src:
                    target_all = src.read().astype(np.float32)

                _, H, W = input_all.shape
                stride = tile_size - overlap

                for row in range(0, H - tile_size + 1, stride):
                    for col in range(0, W - tile_size + 1, stride):
                        inp_tile = input_all[:, row:row+tile_size, col:col+tile_size]
                        tgt_tile = target_all[:, row:row+tile_size, col:col+tile_size]

                        # Skip tiles with too few wet cells
                        wet_frac = (tgt_tile[0] > 0.01).sum() / (tile_size ** 2)
                        if wet_frac < min_wet_fraction:
                            continue

                        tile_name = f"{event_dir.name}_r{row:04d}_c{col:04d}"
                        np.savez_compressed(
                            split_dir / f"{tile_name}.npz",
                            input=inp_tile,
                            target=tgt_tile,
                        )
                        tile_count += 1

            counts[split_name] = tile_count
            logger.info("  %s: %d tiles", split_name, tile_count)

        return counts

    @staticmethod
    def _load_and_align(path: str | Path, target_h: int, target_w: int) -> np.ndarray:
        """Load a single-band raster and resample to target dimensions."""
        with rasterio.open(path) as src:
            data = src.read(
                1,
                out_shape=(target_h, target_w),
                resampling=Resampling.bilinear,
            ).astype(np.float32)
        return data
