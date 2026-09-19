"""
Flood Risk Mapper — Classify and Overlay Flood Envelopes.

Produces flood risk maps and spatial planning recommendations for the
Red River Landscape Corridor (12 BT plots, monorail route, 16 wards).
Supports dual-DEM comparison (FabDEM 30 m vs ANN-DTM 10 m).
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from src.utils.logging_config import get_logger
from src.utils.geo_utils import load_raster, save_raster
from src.utils.visualization import plot_risk_map, plot_comparison

logger = get_logger("da.assessment.risk")


RISK_LEVELS = {
    "low":       (0.0, 0.3),
    "moderate":  (0.3, 1.0),
    "high":      (1.0, 2.0),
    "very_high": (2.0, float("inf")),
}

RISK_CODES = {"low": 1, "moderate": 2, "high": 3, "very_high": 4}


class RiskMapper:
    """Generate flood risk maps from HAND-based flood simulations.

    Supports overlay of flood envelopes onto urban planning boundaries
    for the Red River Landscape Corridor.

    Parameters
    ----------
    config : dict
        Pipeline configuration.
    """

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config["paths"]["outputs"]) / "risk_maps"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def classify_risk(
        self,
        depth_path: str | Path,
        output_name: str = "risk_map",
    ) -> Path:
        """Classify flood depth into risk levels."""
        depth, profile = load_raster(depth_path)
        risk = np.zeros_like(depth, dtype=np.int16)

        for level, (lo, hi) in RISK_LEVELS.items():
            mask = (depth >= lo) & (depth < hi) & (depth > 0.01)
            risk[mask] = RISK_CODES[level]

        out_path = self.output_dir / f"{output_name}.tif"
        save_raster(out_path, risk.astype(np.float32), profile)

        res_m = self.config.get("resolutions", {}).get("fine", 10)
        for level, code in RISK_CODES.items():
            count = (risk == code).sum()
            area_km2 = count * (res_m ** 2) / 1e6
            logger.info("  %s: %d cells (%.2f km²)", level, count, area_km2)

        fig_path = self.output_dir / f"{output_name}.png"
        plot_risk_map(depth, title=f"Flood Risk — {output_name}", save_path=fig_path)

        return out_path

    def compare_dem_risk(
        self,
        baseline_depth_path: str | Path,
        improved_depth_path: str | Path,
        output_name: str = "dem_comparison",
    ) -> dict:
        """Compare flood risk classification between FabDEM and ANN-DTM.

        Parameters
        ----------
        baseline_depth_path : str | Path
            Flood depth from HAND on FabDEM 30 m.
        improved_depth_path : str | Path
            Flood depth from HAND on ANN-DTM 10 m.
        output_name : str
            Output file prefix.

        Returns
        -------
        dict
            Per-risk-level area comparison (km²).
        """
        baseline_depth, _ = load_raster(baseline_depth_path)
        improved_depth, _ = load_raster(improved_depth_path)

        # Comparison figure
        fig_path = self.output_dir / f"{output_name}_comparison.png"
        plot_comparison(
            baseline_depth, improved_depth,
            title_left="FabDEM 30 m (Baseline)",
            title_right="ANN-DTM 10 m (Improved)",
            cmap="Blues",
            vmax=5.0,
            save_path=fig_path,
        )

        # Risk classification for each
        risk_baseline = self.classify_risk(
            baseline_depth_path, output_name=f"{output_name}_baseline_risk",
        )
        risk_improved = self.classify_risk(
            improved_depth_path, output_name=f"{output_name}_improved_risk",
        )

        # Area comparison table
        res_m = self.config.get("resolutions", {}).get("fine", 10)
        cell_area_km2 = (res_m ** 2) / 1e6
        comparison = {}

        for level, code in RISK_CODES.items():
            b_area = (load_raster(risk_baseline)[0] == code).sum() * cell_area_km2
            i_area = (load_raster(risk_improved)[0] == code).sum() * cell_area_km2
            comparison[level] = {
                "baseline_km2": round(float(b_area), 4),
                "improved_km2": round(float(i_area), 4),
                "delta_km2": round(float(b_area - i_area), 4),
            }
            logger.info(
                "  %s: baseline=%.2f km², improved=%.2f km², Δ=%.2f km²",
                level, b_area, i_area, b_area - i_area,
            )

        return comparison

    def overlay_planning_zones(
        self,
        depth_path: str | Path,
        zones_path: str | Path,
        output_name: str = "zone_risk",
    ) -> dict:
        """Overlay flood depth on planning zone boundaries.

        Computes inundation area and average depth for each zone
        (e.g., BT plots, monorail segments, ward boundaries).

        Parameters
        ----------
        depth_path : str | Path
            Flood depth raster.
        zones_path : str | Path
            Vector file (GeoPackage/Shapefile) with planning zone polygons.
            Must have a ``name`` or ``zone_id`` column.
        output_name : str
            Output file prefix.

        Returns
        -------
        dict
            Per-zone flood statistics.
        """
        import geopandas as gpd
        import rasterio
        from rasterio.mask import mask as rio_mask

        zones = gpd.read_file(zones_path)
        id_col = "name" if "name" in zones.columns else zones.columns[0]

        res_m = self.config.get("resolutions", {}).get("fine", 10)
        cell_area_km2 = (res_m ** 2) / 1e6
        zone_results = {}

        with rasterio.open(depth_path) as src:
            # Ensure same CRS
            if zones.crs != src.crs:
                zones = zones.to_crs(src.crs)

            for _, row in zones.iterrows():
                zone_name = str(row[id_col])
                geom = row.geometry

                try:
                    out_image, _ = rio_mask(src, [geom], crop=True, nodata=-9999)
                    depth_clipped = out_image[0]
                    valid = depth_clipped[depth_clipped > -9998]
                    wet = valid[valid > 0.01]

                    zone_results[zone_name] = {
                        "total_cells": int(len(valid)),
                        "wet_cells": int(len(wet)),
                        "wet_area_km2": round(float(len(wet) * cell_area_km2), 4),
                        "mean_depth_m": round(float(wet.mean()), 3) if len(wet) > 0 else 0.0,
                        "max_depth_m": round(float(wet.max()), 3) if len(wet) > 0 else 0.0,
                        "pct_inundated": round(
                            100 * len(wet) / max(len(valid), 1), 2,
                        ),
                    }
                except Exception as e:
                    logger.warning("  Zone '%s': %s", zone_name, e)
                    zone_results[zone_name] = {"error": str(e)}

        for name, stats in zone_results.items():
            if "error" not in stats:
                logger.info(
                    "  Zone '%s': %.2f km² wet (%.1f%%), mean depth=%.2f m",
                    name, stats["wet_area_km2"],
                    stats["pct_inundated"], stats["mean_depth_m"],
                )

        return zone_results

    def compare_scenarios(
        self,
        depth_current: str | Path,
        depth_scenario: str | Path,
        scenario_name: str = "boulevard",
    ) -> Path:
        """Compare risk maps between current state and a planning scenario."""
        current, _ = load_raster(depth_current)
        scenario, _ = load_raster(depth_scenario)

        fig_path = self.output_dir / f"comparison_{scenario_name}.png"
        plot_comparison(
            current, scenario,
            title_left="Current State",
            title_right=f"With {scenario_name.title()}",
            cmap="Blues",
            vmax=5.0,
            save_path=fig_path,
        )

        diff = current - scenario
        improved = (diff > 0.05).sum()
        worsened = (diff < -0.05).sum()
        res = self.config["resolutions"]["fine"]

        logger.info(
            "Scenario '%s': improved=%.2f km², worsened=%.2f km²",
            scenario_name,
            improved * res**2 / 1e6,
            worsened * res**2 / 1e6,
        )

        diff_path = self.output_dir / f"depth_diff_{scenario_name}.tif"
        _, profile = load_raster(depth_current)
        save_raster(diff_path, diff, profile)

        return fig_path
