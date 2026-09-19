"""
Step ⑪ — Training Data Builder for Supervised PI-GAN with Data Assimilation.

Aligns static terrain features, HAND outputs (input), HEC-RAS outputs (target),
and boundary condition constraints (GEOGloWS discharge Q(t), ICESat-2 ATL13 WSE,
and Sentinel-1 SAR masks) onto the same 10 m grid and creates tiled training
pairs (`.npz`) with associated spatiotemporal masks for `L_upstream` and `L_downstream`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from pathlib import Path
import json

from src.utils.logging_config import get_logger
from src.utils.geo_utils import load_raster, save_raster

logger = get_logger("da.gt.builder")


class TrainingDataBuilder:
    """Build aligned training pairs and boundary condition datasets.

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
        self.n_time_steps = config["temporal"]["n_time_steps"]

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
        upstream_q_csv: str | Path | None = None,
        downstream_wse_csv: str | Path | None = None,
        upstream_mask_path: str | Path | None = None,
        downstream_mask_path: str | Path | None = None,
        sar_mask_path: str | Path | None = None,
        hecras_vu_path: str | Path | None = None,
        hecras_vv_path: str | Path | None = None,
    ) -> Path:
        """Create an aligned training pair with boundary condition data for one event.

        Parameters
        ----------
        event_name : str
            Identifier for the simulation event.
        hand_depth_path, hand_vu_path, hand_vv_path : Path
            HAND model outputs.
        hecras_depth_path : Path
            HEC-RAS supervised depth at 10m (target).
        dtm_path, manning_n_path : Path
            Static physical inputs.
        terrain_feature_paths : dict
            Mapping of terrain feature name to raster path.
        upstream_q_csv : Path | None
            GEOGloWS discharge time series CSV (`datetime`, `discharge_m3s`).
        downstream_wse_csv : Path | None
            ATL13 WSE observations table (`t_idx`, `row`, `col`, `wse_value`).
        upstream_mask_path, downstream_mask_path : Path | None
            Binary masks identifying boundary pixels on the 10m grid.
        sar_mask_path : Path | None
            Binary flood extent mask from Sentinel-1 SAR.

        Returns
        -------
        Path
            Directory containing stacked rasters, time series, and metadata.
        """
        event_dir = self.training_dir / event_name
        event_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Building DA training package for event: %s", event_name)

        dtm, ref_profile = load_raster(dtm_path)
        h, w = dtm.shape

        hand_depth = self._load_and_align(hand_depth_path, h, w)
        hand_vu = self._load_and_align(hand_vu_path, h, w)
        hand_vv = self._load_and_align(hand_vv_path, h, w)
        hecras_depth = self._load_and_align(hecras_depth_path, h, w)
        manning_n = self._load_and_align(manning_n_path, h, w)

        input_channels = [hand_depth, hand_vu, hand_vv, dtm, manning_n]
        channel_names = ["hand_depth", "hand_vu", "hand_vv", "dtm", "manning_n"]

        for feat_name in ["slope", "twi", "distance_to_river"]:
            if feat_name in terrain_feature_paths:
                feat = self._load_and_align(terrain_feature_paths[feat_name], h, w)
                input_channels.append(feat)
                channel_names.append(feat_name)

        input_stack = np.stack(input_channels, axis=0)

        target_channels = [hecras_depth]
        target_names = ["hecras_depth"]
        if hecras_vu_path:
            target_channels.append(self._load_and_align(hecras_vu_path, h, w))
            target_names.append("hecras_vu")
        if hecras_vv_path:
            target_channels.append(self._load_and_align(hecras_vv_path, h, w))
            target_names.append("hecras_vv")
        target_stack = np.stack(target_channels, axis=0)

        input_profile = {**ref_profile, "count": input_stack.shape[0], "dtype": "float32"}
        target_profile = {**ref_profile, "count": target_stack.shape[0], "dtype": "float32"}

        save_raster(event_dir / "input.tif", input_stack, input_profile)
        save_raster(event_dir / "target.tif", target_stack, target_profile)

        # Process and save Boundary Condition masks
        if upstream_mask_path and Path(upstream_mask_path).exists():
            up_mask = self._load_and_align(upstream_mask_path, h, w) > 0.5
        else:
            # Default upstream boundary mask: leftmost column
            up_mask = np.zeros((h, w), dtype=bool)
            up_mask[:, 0:2] = True
        np.save(event_dir / "upstream_mask.npy", up_mask.astype(np.uint8))

        if downstream_mask_path and Path(downstream_mask_path).exists():
            down_mask = self._load_and_align(downstream_mask_path, h, w) > 0.5
        else:
            # Default downstream boundary mask: bottom row
            down_mask = np.zeros((h, w), dtype=bool)
            down_mask[-2:, :] = True
        np.save(event_dir / "downstream_mask.npy", down_mask.astype(np.uint8))

        if sar_mask_path and Path(sar_mask_path).exists():
            sar_mask = self._load_and_align(sar_mask_path, h, w)
            np.save(event_dir / "sar_mask.npy", sar_mask.astype(np.float32))

        # Copy or generate boundary condition time series tables
        if upstream_q_csv and Path(upstream_q_csv).exists():
            pd.read_csv(upstream_q_csv).to_csv(event_dir / "upstream_Q.csv", index=False)
        else:
            # Generate default hydrograph vector for testing
            q_vec = np.linspace(1500.0, 4500.0, self.n_time_steps, dtype=np.float32)
            pd.DataFrame({"t_idx": np.arange(self.n_time_steps), "discharge_m3s": q_vec}).to_csv(
                event_dir / "upstream_Q.csv", index=False
            )

        if downstream_wse_csv and Path(downstream_wse_csv).exists():
            pd.read_csv(downstream_wse_csv).to_csv(event_dir / "downstream_wse.csv", index=False)
        else:
            # Sparse synthetic ATL13 observations along downstream mask
            sparse_obs = []
            down_rows, down_cols = np.where(down_mask)
            for t_idx in [10, 25, 40]:  # Sparse satellite overpasses
                if len(down_rows) > 0:
                    idx = np.random.choice(len(down_rows), min(15, len(down_rows)), replace=False)
                    for r, c in zip(down_rows[idx], down_cols[idx]):
                        # Baseline WSE approx DTM + depth
                        sparse_obs.append({"t_idx": t_idx, "row": r, "col": c, "wse_value": float(dtm[r, c] + 2.5)})
            pd.DataFrame(sparse_obs).to_csv(event_dir / "downstream_wse.csv", index=False)

        metadata = {
            "event_name": event_name,
            "input_channels": channel_names,
            "target_channels": target_names,
            "grid_shape": [h, w],
            "resolution_m": self.fine_res,
            "n_time_steps": self.n_time_steps,
        }
        with open(event_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("  Package ready: %d input channels, %d targets, BC masks stored.",
                    input_stack.shape[0], target_stack.shape[0])
        return event_dir

    def create_tiles(
        self,
        tile_size: int = 256,
        overlap: int = 32,
        min_wet_fraction: float = 0.05,
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    ) -> dict[str, int]:
        """Tile all event pairs along with their boundary condition masks and series."""
        event_dirs = sorted([
            d for d in self.training_dir.iterdir()
            if d.is_dir() and (d / "input.tif").exists()
        ])

        if not event_dirs:
            raise FileNotFoundError("No event pairs found in %s" % self.training_dir)

        n = len(event_dirs)
        n_train = max(1, int(n * split_ratios[0]))
        n_val = max(1, int(n * split_ratios[1]))

        train_events = event_dirs[:n_train]
        val_events = event_dirs[n_train:n_train + n_val]
        test_events = event_dirs[n_train + n_val:]

        logger.info("Event split: train=%d, val=%d, test=%d",
                    len(train_events), len(val_events), len(test_events))

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
                with rasterio.open(event_dir / "input.tif") as src:
                    input_all = src.read().astype(np.float32)
                with rasterio.open(event_dir / "target.tif") as src:
                    target_all = src.read().astype(np.float32)

                up_mask_all = np.load(event_dir / "upstream_mask.npy")
                down_mask_all = np.load(event_dir / "downstream_mask.npy")
                sar_mask_path = event_dir / "sar_mask.npy"
                sar_mask_all = np.load(sar_mask_path) if sar_mask_path.exists() else None

                # Load boundary tables
                up_df = pd.read_csv(event_dir / "upstream_Q.csv")
                down_df = pd.read_csv(event_dir / "downstream_wse.csv")
                q_series = up_df["discharge_m3s"].to_numpy(dtype=np.float32)

                _, H, W = input_all.shape
                stride = tile_size - overlap

                for row in range(0, H - tile_size + 1, stride):
                    for col in range(0, W - tile_size + 1, stride):
                        inp_tile = input_all[:, row:row+tile_size, col:col+tile_size]
                        tgt_tile = target_all[:, row:row+tile_size, col:col+tile_size]

                        wet_frac = (tgt_tile[0] > 0.01).sum() / (tile_size ** 2)
                        if wet_frac < min_wet_fraction:
                            continue

                        up_tile = up_mask_all[row:row+tile_size, col:col+tile_size]
                        down_tile = down_mask_all[row:row+tile_size, col:col+tile_size]

                        # Filter downstream observations inside this tile window
                        if not down_df.empty and "row" in down_df.columns:
                            in_tile = down_df[
                                (down_df["row"] >= row) & (down_df["row"] < row + tile_size) &
                                (down_df["col"] >= col) & (down_df["col"] < col + tile_size)
                            ].copy()
                            in_tile["row"] -= row
                            in_tile["col"] -= col
                            wse_obs = in_tile[["t_idx", "row", "col", "wse_value"]].to_numpy(dtype=np.float32)
                        else:
                            wse_obs = np.empty((0, 4), dtype=np.float32)

                        sar_tile = sar_mask_all[row:row+tile_size, col:col+tile_size] if sar_mask_all is not None else np.empty((0, 0), dtype=np.float32)

                        tile_name = f"{event_dir.name}_r{row:04d}_c{col:04d}"
                        np.savez_compressed(
                            split_dir / f"{tile_name}.npz",
                            input=inp_tile,
                            target=tgt_tile,
                            upstream_mask=up_tile,
                            downstream_mask=down_tile,
                            upstream_Q=q_series,
                            downstream_wse=wse_obs,
                            sar_mask=sar_tile,
                        )
                        tile_count += 1

            counts[split_name] = tile_count
            logger.info("  %s: %d tiles generated", split_name, tile_count)

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
