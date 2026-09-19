"""
Scenario runner — compare current state vs. urban planning alternatives.

Runs the full HAND → PI-GAN pipeline for different infrastructure
scenarios and computes comparative metrics.
"""

from __future__ import annotations

import time
import json
import numpy as np
from pathlib import Path

from src.phase2_groundtruth.hand_model import HANDFloodModel
from src.phase3_pigan.inference import PIGANInference
from src.phase4_assessment.metrics import compute_flood_metrics, compute_speedup
from src.utils.logging_config import get_logger
from src.utils.geo_utils import load_raster

logger = get_logger("pigan.phase4.scenario")


class ScenarioRunner:
    """Run PI-GAN for multiple planning scenarios and compare results.

    Parameters
    ----------
    config : dict
        Pipeline configuration.
    pigan_checkpoint : str | Path
        Path to the trained PI-GAN checkpoint.
    """

    def __init__(self, config: dict, pigan_checkpoint: str | Path):
        self.config = config
        self.hand_model = HANDFloodModel(config)
        self.pigan = PIGANInference(config, pigan_checkpoint)
        self.output_dir = Path(config["paths"]["outputs"]) / "scenarios"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_scenario(
        self,
        scenario_name: str,
        hand_raster_path: str | Path,
        slope_path: str | Path,
        manning_n_path: str | Path,
        aspect_path: str | Path,
        dtm_path: str | Path,
        twi_path: str | Path,
        dist_river_path: str | Path,
        water_level: float,
        hecras_depth_path: str | Path | None = None,
    ) -> dict:
        """Run a single scenario through HAND → PI-GAN pipeline.

        Parameters
        ----------
        scenario_name : str
        hand_raster_path ... dist_river_path : Path
            Input raster paths.
        water_level : float
            Gauge water level for HAND model.
        hecras_depth_path : Path | None
            If provided, compute accuracy metrics vs HEC-RAS.

        Returns
        -------
        dict
            Results including timing, metrics, and output paths.
        """
        logger.info("Running scenario: %s (WL=%.2f m)", scenario_name, water_level)
        scenario_dir = self.output_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # ---- Step 1: HAND model (ultra-fast) ----
        t0 = time.time()
        hand_result = self.hand_model.run(
            hand_raster_path=hand_raster_path,
            slope_raster_path=slope_path,
            manning_n_path=manning_n_path,
            aspect_raster_path=aspect_path,
            water_level=water_level,
            event_name=f"{scenario_name}_hand",
        )
        t_hand = time.time() - t0

        # ---- Step 2: PI-GAN super-resolution ----
        hand_dir = Path(self.config["paths"]["data_processed"]) / "hand_results" / f"{scenario_name}_hand"

        t0 = time.time()
        pigan_outputs = self.pigan.predict(
            hand_depth_path=hand_dir / "depth.tif",
            hand_vu_path=hand_dir / "velocity_u.tif",
            hand_vv_path=hand_dir / "velocity_v.tif",
            dtm_path=dtm_path,
            manning_n_path=manning_n_path,
            slope_path=slope_path,
            twi_path=twi_path,
            dist_river_path=dist_river_path,
            output_dir=scenario_dir,
        )
        t_pigan = time.time() - t0

        total_time = t_hand + t_pigan

        results = {
            "scenario": scenario_name,
            "water_level": water_level,
            "time_hand_s": round(t_hand, 2),
            "time_pigan_s": round(t_pigan, 2),
            "time_total_s": round(total_time, 2),
            "outputs": {k: str(v) for k, v in pigan_outputs.items()},
        }

        # ---- Step 3: Compare with HEC-RAS (if available) ----
        if hecras_depth_path:
            pigan_depth, _ = load_raster(pigan_outputs["depth"])
            hecras_depth, _ = load_raster(hecras_depth_path)

            # Ensure same shape
            min_h = min(pigan_depth.shape[0], hecras_depth.shape[0])
            min_w = min(pigan_depth.shape[1], hecras_depth.shape[1])
            pigan_depth = pigan_depth[:min_h, :min_w]
            hecras_depth = hecras_depth[:min_h, :min_w]

            metrics = compute_flood_metrics(pigan_depth, hecras_depth)
            results["metrics"] = metrics.to_dict()

            # Estimate HEC-RAS time (from training phase)
            estimated_hecras_time = 3 * 3600  # ~3 hours typical
            results["speedup"] = round(
                compute_speedup(estimated_hecras_time, total_time), 1
            )

            logger.info("  Metrics: %s", metrics)
            logger.info("  Speedup: %.0f×", results["speedup"])

        # Save results
        with open(scenario_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        return results

    def compare_scenarios(
        self,
        scenarios: list[dict],
    ) -> list[dict]:
        """Run multiple scenarios and produce a comparison table.

        Parameters
        ----------
        scenarios : list[dict]
            Each dict should have keys matching ``run_scenario()`` params.

        Returns
        -------
        list[dict]
            All scenario results.
        """
        all_results = []
        for scenario in scenarios:
            result = self.run_scenario(**scenario)
            all_results.append(result)

        # Save comparison
        comparison_path = self.output_dir / "comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(all_results, f, indent=2)

        logger.info("Comparison saved → %s", comparison_path)
        return all_results
