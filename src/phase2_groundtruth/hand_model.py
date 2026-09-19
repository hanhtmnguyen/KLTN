"""
Step ⑩ — HAND (Height Above Nearest Drainage) flood model.

An ultra-fast, topography-based flood inundation model that produces
the INPUT flood maps fed into the PI-GAN generator.  Runs in seconds.

For a given water level W at a gauge station:
    depth(x, y) = max(0,  W − HAND(x, y))
    velocity estimated via Manning's equation.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from dataclasses import dataclass

from src.utils.logging_config import get_logger
from src.utils.geo_utils import load_raster, save_raster

logger = get_logger("pigan.phase2.hand")


@dataclass
class FloodResult:
    """Container for flood simulation outputs."""
    depth: np.ndarray
    velocity_u: np.ndarray
    velocity_v: np.ndarray
    profile: dict        # rasterio profile for georeferencing


class HANDFloodModel:
    """HAND-based flood inundation model.

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
    ) -> FloodResult:
        """Run the HAND flood model for a single water level.

        Parameters
        ----------
        hand_raster_path : str | Path
            HAND raster (height above nearest drainage, metres).
        slope_raster_path : str | Path
            Terrain slope (degrees).
        manning_n_path : str | Path
            Manning's n roughness raster.
        aspect_raster_path : str | Path
            Terrain aspect (degrees, 0=North, clockwise).
        water_level : float
            Water surface elevation at gauge (metres).
        event_name : str
            Identifier for this scenario.

        Returns
        -------
        FloodResult
        """
        logger.info(
            "Running HAND model: water_level=%.2f m, event=%s",
            water_level, event_name,
        )

        # Load rasters
        hand, profile = load_raster(hand_raster_path)
        slope_deg, _ = load_raster(slope_raster_path)
        manning_n, _ = load_raster(manning_n_path)
        aspect_deg, _ = load_raster(aspect_raster_path)

        # ---- Flood depth ----
        # depth = max(0, water_level - HAND)
        depth = np.maximum(0.0, water_level - hand).astype(np.float32)

        # ---- Velocity estimation (Manning's equation) ----
        # v = (1/n) * R^(2/3) * S^(1/2)
        # For wide shallow flow: hydraulic radius R ≈ depth
        # S = slope (convert from degrees to dimensionless gradient)
        slope_rad = np.deg2rad(np.clip(slope_deg, 0.01, 89.0))
        slope_gradient = np.tan(slope_rad)

        # Prevent division by zero
        n_safe = np.clip(manning_n, 0.01, 1.0)
        depth_safe = np.clip(depth, 0.0, None)

        # Manning velocity magnitude
        velocity_mag = np.where(
            depth_safe > 0.01,
            (1.0 / n_safe) * (depth_safe ** (2.0 / 3.0)) * (slope_gradient ** 0.5),
            0.0,
        ).astype(np.float32)

        # Clip unrealistic velocities
        velocity_mag = np.clip(velocity_mag, 0.0, 10.0)

        # Decompose into u (east) and v (north) components using aspect
        aspect_rad = np.deg2rad(aspect_deg)
        velocity_u = (velocity_mag * np.sin(aspect_rad)).astype(np.float32)  # east
        velocity_v = (velocity_mag * np.cos(aspect_rad)).astype(np.float32)  # north

        # ---- Save outputs ----
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
        )

    def run_batch(
        self,
        hand_raster_path: str | Path,
        slope_raster_path: str | Path,
        manning_n_path: str | Path,
        aspect_raster_path: str | Path,
        events: list[dict],
    ) -> list[FloodResult]:
        """Run the HAND model for multiple water level scenarios.

        Parameters
        ----------
        events : list[dict]
            Each dict must have ``name`` and ``water_level`` keys.
            Example: ``[{"name": "2018_flood", "water_level": 11.5}, ...]``

        Returns
        -------
        list[FloodResult]
        """
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
