"""
Step ⑩ — HAND (Height Above Nearest Drainage) flood model.

A topography-based flood inundation model that produces flood depth
and velocity maps for any given water level.  Now supports **dual-DEM
comparison** to evaluate cascading impact of DTM quality on flood extent.

For a given water level W at a gauge station:
    depth(x, y) = max(0,  W − HAND(x, y))
    velocity estimated via Manning's equation.

Multi-threshold mode generates inundation maps at HAND ≤ 1, 2, ..., 10 m
following the methodology of Sun et al. (2026) / manyoarcane/HAND.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

from src.utils.logging_config import get_logger
from src.utils.geo_utils import load_raster, save_raster

logger = get_logger("da.prep.hand")


@dataclass
class FloodResult:
    """Container for flood simulation outputs."""
    depth: np.ndarray
    velocity_u: np.ndarray
    velocity_v: np.ndarray
    profile: dict
    water_level: float = 0.0
    dem_label: str = ""


@dataclass
class DualDEMComparison:
    """Container for side-by-side flood results from two DEMs."""
    baseline: FloodResult          # FabDEM 30 m
    improved: FloodResult          # ANN-DTM 10 m
    depth_diff: np.ndarray = field(default=None)        # baseline - improved
    wet_area_baseline_km2: float = 0.0
    wet_area_improved_km2: float = 0.0


class HANDFloodModel:
    """HAND-based flood inundation model with dual-DEM comparison support.

    Parameters
    ----------
    config : dict
        Full pipeline configuration.
    """

    def __init__(self, config: dict):
        self.config = config
        self.gravity = config["physics"]["gravity"]
        self.output_dir = Path(config["paths"]["data_processed"]) / "hand_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        hand_raster_path: str | Path,
        slope_raster_path: str | Path,
        manning_n_path: str | Path,
        aspect_raster_path: str | Path,
        water_level: float,
        event_name: str = "event",
        dem_label: str = "",
    ) -> FloodResult:
        """Run the HAND flood model for a single water level."""
        logger.info(
            "Running HAND model: water_level=%.2f m, event=%s, dem=%s",
            water_level, event_name, dem_label or "default",
        )

        hand, profile = load_raster(hand_raster_path)
        slope_deg, _ = load_raster(slope_raster_path)
        manning_n, _ = load_raster(manning_n_path)
        aspect_deg, _ = load_raster(aspect_raster_path)

        depth = np.maximum(0.0, water_level - hand).astype(np.float32)

        slope_rad = np.deg2rad(np.clip(slope_deg, 0.01, 89.0))
        slope_gradient = np.tan(slope_rad)

        n_safe = np.clip(manning_n, 0.01, 1.0)
        depth_safe = np.clip(depth, 0.0, None)

        velocity_mag = np.where(
            depth_safe > 0.01,
            (1.0 / n_safe) * (depth_safe ** (2.0 / 3.0)) * (slope_gradient ** 0.5),
            0.0,
        ).astype(np.float32)

        velocity_mag = np.clip(velocity_mag, 0.0, 10.0)

        aspect_rad = np.deg2rad(aspect_deg)
        velocity_u = (velocity_mag * np.sin(aspect_rad)).astype(np.float32)
        velocity_v = (velocity_mag * np.cos(aspect_rad)).astype(np.float32)

        event_dir = self.output_dir / event_name
        event_dir.mkdir(parents=True, exist_ok=True)

        save_raster(event_dir / "depth.tif", depth, profile)
        save_raster(event_dir / "velocity_u.tif", velocity_u, profile)
        save_raster(event_dir / "velocity_v.tif", velocity_v, profile)

        wet_cells = (depth > 0.01).sum()
        logger.info(
            "  HAND result: wet_cells=%d (%.1f%%), max_depth=%.2f m, max_vel=%.2f m/s",
            wet_cells,
            100 * wet_cells / depth.size,
            depth.max(),
            velocity_mag.max(),
        )

        return FloodResult(
            depth=depth,
            velocity_u=velocity_u,
            velocity_v=velocity_v,
            profile=profile,
            water_level=water_level,
            dem_label=dem_label,
        )

    def run_multi_threshold(
        self,
        hand_raster_path: str | Path,
        thresholds: list[float] | None = None,
        event_name: str = "multi_threshold",
    ) -> dict[float, np.ndarray]:
        """Generate binary inundation maps at multiple HAND thresholds.

        Following Sun et al. (2026) methodology:
        For each threshold T, inundated = HAND ≤ T.

        Parameters
        ----------
        hand_raster_path : str | Path
            Path to the HAND raster.
        thresholds : list[float]
            HAND thresholds in metres (default from config).
        event_name : str
            Output subdirectory name.

        Returns
        -------
        dict[float, np.ndarray]
            Mapping threshold → binary inundation mask.
        """
        if thresholds is None:
            thresholds = self.config.get("flood_simulation", {}).get(
                "hand_thresholds", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            )

        hand, profile = load_raster(hand_raster_path)
        event_dir = self.output_dir / event_name
        event_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        res_m = self.config.get("resolutions", {}).get("fine", 10)

        for t in thresholds:
            inundated = (hand <= t).astype(np.float32)
            area_km2 = inundated.sum() * (res_m ** 2) / 1e6
            results[t] = inundated

            save_raster(event_dir / f"inundation_hand_le_{t}m.tif", inundated, profile)
            logger.info("  HAND ≤ %.0f m: %.2f km² inundated", t, area_km2)

        return results

    def run_dual_comparison(
        self,
        baseline_hand_path: str | Path,
        improved_hand_path: str | Path,
        slope_path: str | Path,
        manning_n_path: str | Path,
        aspect_path: str | Path,
        water_level: float,
        event_name: str = "dual_comparison",
    ) -> DualDEMComparison:
        """Run HAND flood model on TWO DEMs and compare results.

        This is the core of the cascading impact evaluation: how does
        improving the DTM (FabDEM 30m → ANN-DTM 10m) affect flood extent?

        Parameters
        ----------
        baseline_hand_path : str | Path
            HAND raster from FabDEM 30 m (coarse).
        improved_hand_path : str | Path
            HAND raster from ANN-DTM 10 m (corrected).
        slope_path, manning_n_path, aspect_path : str | Path
            Shared terrain rasters (from corrected DTM grid).
        water_level : float
            Gauge water level (m).
        event_name : str
            Event identifier.

        Returns
        -------
        DualDEMComparison
        """
        logger.info(
            "Running dual-DEM HAND comparison: WL=%.2f m, event=%s",
            water_level, event_name,
        )

        baseline = self.run(
            hand_raster_path=baseline_hand_path,
            slope_raster_path=slope_path,
            manning_n_path=manning_n_path,
            aspect_raster_path=aspect_path,
            water_level=water_level,
            event_name=f"{event_name}_fabdem30m",
            dem_label="FabDEM_30m",
        )

        improved = self.run(
            hand_raster_path=improved_hand_path,
            slope_raster_path=slope_path,
            manning_n_path=manning_n_path,
            aspect_raster_path=aspect_path,
            water_level=water_level,
            event_name=f"{event_name}_anndtm10m",
            dem_label="ANN-DTM_10m",
        )

        # Align shapes (handle resolution differences)
        min_h = min(baseline.depth.shape[0], improved.depth.shape[0])
        min_w = min(baseline.depth.shape[1], improved.depth.shape[1])
        b_depth = baseline.depth[:min_h, :min_w]
        i_depth = improved.depth[:min_h, :min_w]

        depth_diff = b_depth - i_depth
        res_m = self.config.get("resolutions", {}).get("fine", 10)
        cell_area_km2 = (res_m ** 2) / 1e6

        comparison = DualDEMComparison(
            baseline=baseline,
            improved=improved,
            depth_diff=depth_diff,
            wet_area_baseline_km2=(b_depth > 0.01).sum() * cell_area_km2,
            wet_area_improved_km2=(i_depth > 0.01).sum() * cell_area_km2,
        )

        logger.info(
            "  Baseline (FabDEM 30m): wet area = %.2f km²",
            comparison.wet_area_baseline_km2,
        )
        logger.info(
            "  Improved (ANN-DTM 10m): wet area = %.2f km²",
            comparison.wet_area_improved_km2,
        )
        logger.info(
            "  Depth diff: mean=%.3f m, std=%.3f m",
            float(np.mean(depth_diff[b_depth > 0.01])) if (b_depth > 0.01).any() else 0,
            float(np.std(depth_diff[b_depth > 0.01])) if (b_depth > 0.01).any() else 0,
        )

        # Save depth difference map
        diff_dir = self.output_dir / event_name
        diff_dir.mkdir(parents=True, exist_ok=True)
        save_raster(
            diff_dir / "depth_diff_baseline_vs_improved.tif",
            depth_diff,
            baseline.profile,
        )

        return comparison

    def run_batch(
        self,
        hand_raster_path: str | Path,
        slope_raster_path: str | Path,
        manning_n_path: str | Path,
        aspect_raster_path: str | Path,
        events: list[dict],
    ) -> list[FloodResult]:
        """Run the HAND model for multiple water level scenarios."""
        results = []
        for event in events:
            result = self.run(
                hand_raster_path=hand_raster_path,
                slope_raster_path=slope_raster_path,
                manning_n_path=manning_n_path,
                aspect_raster_path=aspect_raster_path,
                water_level=event["water_level"],
                event_name=event["name"],
            )
            results.append(result)
        return results
