"""
Dual-DEM Cascading Impact Scenario Runner.

Runs the HAND flood model on FabDEM 30 m (baseline) and ANN-DTM 10 m
(improved), compares both against a reference flood extent (Sentinel-1
SAR or HEC-RAS), and evaluates the cascading impact of DTM quality
on flood inundation accuracy.
"""

from __future__ import annotations

import time
import json
import numpy as np
from pathlib import Path

from src.preprocessing.hand_model import HANDFloodModel
from src.assessment.metrics import (
    compute_flood_metrics,
    compute_cascading_impact,
)
from src.utils.logging_config import get_logger
from src.utils.geo_utils import load_raster

logger = get_logger("da.assessment.scenario")


class ScenarioRunner:
    """Run dual-DEM HAND flood simulations and cascading impact analysis.

    Parameters
    ----------
    config : dict
        Pipeline configuration.
    """

    def __init__(self, config: dict):
        self.config = config
        self.hand_model = HANDFloodModel(config)
        self.output_dir = Path(config["paths"]["outputs"]) / "scenarios"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_scenario(
        self,
        scenario_name: str,
        baseline_hand_path: str | Path,
        improved_hand_path: str | Path,
        slope_path: str | Path,
        manning_n_path: str | Path,
        aspect_path: str | Path,
        water_level: float,
        reference_flood_path: str | Path | None = None,
    ) -> dict:
        """Run a single scenario: HAND on two DEMs + cascading impact.

        Parameters
        ----------
        scenario_name : str
            Human-readable label for this scenario.
        baseline_hand_path : str | Path
            HAND raster derived from FabDEM 30 m.
        improved_hand_path : str | Path
            HAND raster derived from ANN-DTM 10 m.
        slope_path, manning_n_path, aspect_path : str | Path
            Shared terrain rasters.
        water_level : float
            Gauge water level (m).
        reference_flood_path : str | Path | None
            Reference flood depth/extent for validation (SAR or HEC-RAS).

        Returns
        -------
        dict
            Scenario results including timing and metrics.
        """
        logger.info("Running scenario: %s (WL=%.2f m)", scenario_name, water_level)
        scenario_dir = self.output_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # ── Run dual-DEM comparison ──────────────────────────────────
        t0 = time.time()
        comparison = self.hand_model.run_dual_comparison(
            baseline_hand_path=baseline_hand_path,
            improved_hand_path=improved_hand_path,
            slope_path=slope_path,
            manning_n_path=manning_n_path,
            aspect_path=aspect_path,
            water_level=water_level,
            event_name=scenario_name,
        )
        t_total = time.time() - t0

        results = {
            "scenario": scenario_name,
            "water_level": water_level,
            "time_s": round(t_total, 2),
            "wet_area_baseline_km2": round(comparison.wet_area_baseline_km2, 4),
            "wet_area_improved_km2": round(comparison.wet_area_improved_km2, 4),
        }

        # ── Compare against reference (if available) ─────────────────
        if reference_flood_path and Path(reference_flood_path).exists():
            reference_depth, _ = load_raster(reference_flood_path)

            # Align shapes
            b_depth = comparison.baseline.depth
            i_depth = comparison.improved.depth
            min_h = min(b_depth.shape[0], i_depth.shape[0], reference_depth.shape[0])
            min_w = min(b_depth.shape[1], i_depth.shape[1], reference_depth.shape[1])
            b_depth = b_depth[:min_h, :min_w]
            i_depth = i_depth[:min_h, :min_w]
            reference_depth = reference_depth[:min_h, :min_w]

            impact = compute_cascading_impact(
                baseline_flood=b_depth,
                improved_flood=i_depth,
                reference_flood=reference_depth,
            )
            results["cascading_impact"] = impact.to_dict()
            logger.info("  Cascading impact: %s", impact)

        # ── Multi-threshold inundation maps ──────────────────────────
        thresholds = self.config.get("flood_simulation", {}).get("hand_thresholds")
        if thresholds:
            baseline_inundation = self.hand_model.run_multi_threshold(
                hand_raster_path=baseline_hand_path,
                thresholds=thresholds,
                event_name=f"{scenario_name}_baseline_thresholds",
            )
            improved_inundation = self.hand_model.run_multi_threshold(
                hand_raster_path=improved_hand_path,
                thresholds=thresholds,
                event_name=f"{scenario_name}_improved_thresholds",
            )
            results["n_thresholds"] = len(thresholds)

        # ── Save results ─────────────────────────────────────────────
        with open(scenario_dir / "scenario_metrics.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        return results

    def compare_scenarios(self, scenarios: list[dict]) -> list[dict]:
        """Run multiple scenarios and produce a comparison table.

        Parameters
        ----------
        scenarios : list[dict]
            Each dict is kwargs for ``run_scenario()``.

        Returns
        -------
        list[dict]
        """
        all_results = []
        for scenario in scenarios:
            result = self.run_scenario(**scenario)
            all_results.append(result)

        comparison_path = self.output_dir / "comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        logger.info("Comparison saved → %s", comparison_path)
        return all_results
