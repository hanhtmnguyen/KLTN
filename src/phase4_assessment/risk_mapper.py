"""
Flood Risk Mapper — classify and overlay flood maps with land use.

Produces the final flood risk maps and spatial planning recommendations
for the Red River corridor.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from src.utils.logging_config import get_logger
from src.utils.geo_utils import load_raster, save_raster
from src.utils.visualization import plot_risk_map, plot_comparison

logger = get_logger("pigan.phase4.risk")


# Risk classification thresholds (metres)
RISK_LEVELS = {
    "low":       (0.0, 0.3),
    "moderate":  (0.3, 1.0),
    "high":      (1.0, 2.0),
    "very_high": (2.0, float("inf")),
}

RISK_CODES = {"low": 1, "moderate": 2, "high": 3, "very_high": 4}


class RiskMapper:
    """Generate flood risk maps from PI-GAN depth output.

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
        """Classify flood depth into risk levels.

        Parameters
        ----------
        depth_path : str | Path
            PI-GAN predicted depth raster.
        output_name : str
            Base name for output files.

        Returns
        -------
        Path
            Path to the risk classification raster.
        """
        depth, profile = load_raster(depth_path)
        risk = np.zeros_like(depth, dtype=np.int16)

        for level, (lo, hi) in RISK_LEVELS.items():
            mask = (depth >= lo) & (depth < hi) & (depth > 0.01)
            risk[mask] = RISK_CODES[level]

        out_path = self.output_dir / f"{output_name}.tif"
        save_raster(out_path, risk.astype(np.float32), profile)

        # Log statistics
        for level, code in RISK_CODES.items():
            count = (risk == code).sum()
            area_km2 = count * (self.config["resolutions"]["fine"] ** 2) / 1e6
            logger.info("  %s: %d cells (%.2f km²)", level, count, area_km2)

        # Generate figure
        fig_path = self.output_dir / f"{output_name}.png"
        plot_risk_map(depth, title=f"Flood Risk — {output_name}", save_path=fig_path)

        return out_path

    def compare_scenarios(
        self,
        depth_current: str | Path,
        depth_scenario: str | Path,
        scenario_name: str = "boulevard",
    ) -> Path:
        """Compare risk maps between current state and a planning scenario.

        Parameters
        ----------
        depth_current : Path
            Depth raster for current state.
        depth_scenario : Path
            Depth raster after infrastructure change.
        scenario_name : str

        Returns
        -------
        Path
            Path to comparison figure.
        """
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

        # Compute improvement
        diff = current - scenario
        improved = (diff > 0.05).sum()  # Areas with reduced depth
        worsened = (diff < -0.05).sum()
        res = self.config["resolutions"]["fine"]

        logger.info(
            "Scenario '%s': improved=%.2f km², worsened=%.2f km²",
            scenario_name,
            improved * res**2 / 1e6,
            worsened * res**2 / 1e6,
        )

        # Save diff raster
        diff_path = self.output_dir / f"depth_diff_{scenario_name}.tif"
        _, profile = load_raster(depth_current)
        save_raster(diff_path, diff, profile)

        return fig_path
