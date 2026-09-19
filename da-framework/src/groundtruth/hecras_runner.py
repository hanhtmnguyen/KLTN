"""
Steps ⑧ & ⑨ — HEC-RAS 2D automation (setup helpers + simulation runner).

HEC-RAS produces the ultra-accurate 10 m flood maps used as the
TARGET / ground truth for supervised PI-GAN training.
"""

from __future__ import annotations

import numpy as np
import h5py
from pathlib import Path

from src.utils.logging_config import get_logger
from src.utils.geo_utils import save_raster

logger = get_logger("da.gt.hecras")


class HECRASRunner:
    """Automate HEC-RAS 2D simulation runs and result extraction.

    Parameters
    ----------
    config : dict
        Full pipeline configuration.
    project_path : str | Path
        Path to the HEC-RAS project file (``.prj``).
    """

    def __init__(self, config: dict, project_path: str | Path):
        self.config = config
        self.project_path = Path(project_path)
        self.output_dir = Path(config["paths"]["data_processed"]) / "hecras_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_simulation(self, plan_name: str | None = None) -> bool:
        """Execute a HEC-RAS simulation via the COM controller."""
        try:
            import win32com.client
        except ImportError:
            logger.error(
                "pywin32 not installed. Install with: pip install pywin32"
            )
            raise

        logger.info("Opening HEC-RAS project: %s", self.project_path)

        prog_ids = [
            "RAS701.HECRASController",
            "RAS70.HECRASController",
            "HECRASController",
            "RAS641.HECRASController",
            "RC2.HECRASController",
        ]
        ras = None
        last_err = None
        for pid in prog_ids:
            try:
                ras = win32com.client.Dispatch(pid)
                logger.info("  Successfully connected to COM controller: %s", pid)
                break
            except Exception as e:
                last_err = e

        if ras is None:
            logger.error("Could not connect to HEC-RAS COM controller. Last error: %s", last_err)
            raise RuntimeError(f"Failed to connect to any HEC-RAS COM controller ({prog_ids}): {last_err}")

        ras.Project_Open(str(self.project_path))

        if plan_name:
            ras.Plan_SetCurrent(plan_name)
            logger.info("  Set active plan: %s", plan_name)

        logger.info("  Starting compute...")
        n_msg = 0
        compute_msgs = None
        block_msgs = True
        result = ras.Compute_CurrentPlan(n_msg, compute_msgs, block_msgs)

        success = result[0] if isinstance(result, tuple) else result
        logger.info("  Compute finished. Success=%s", success)

        ras.QuitRas()
        return bool(success)

    def extract_results(
        self,
        hdf_path: str | Path,
        event_name: str,
        time_step: int = -1,
    ) -> dict[str, Path]:
        """Extract flood depth and velocity grids from HEC-RAS HDF5 output."""
        hdf_path = Path(hdf_path)
        logger.info("Extracting results from %s (time_step=%d)", hdf_path.name, time_step)

        event_dir = self.output_dir / event_name
        event_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(hdf_path, "r") as f:
            results_root = f["Results"]["Unsteady"]["Output"]["Output Blocks"]
            base_output = results_root["Base Output"]["Unsteady Time Series"]
            flow_areas = base_output["2D Flow Areas"]

            area_name = list(flow_areas.keys())[0]
            area = flow_areas[area_name]

            if "Depth" in area:
                depth_all = area["Depth"][:]
                depth = depth_all[time_step]
            else:
                logger.warning("  'Depth' dataset not found in HDF5")
                depth = None

            if "Face Velocity" in area:
                vel_all = area["Face Velocity"][:]
                vel = vel_all[time_step]
            else:
                vel = None

        logger.info(
            "  Extracted: depth shape=%s",
            depth.shape if depth is not None else "N/A",
        )

        outputs = {}

        if depth is not None:
            depth_path = event_dir / "depth.npy"
            np.save(depth_path, depth.astype(np.float32))
            outputs["depth"] = depth_path

        if vel is not None:
            vel_path = event_dir / "velocity.npy"
            np.save(vel_path, vel.astype(np.float32))
            outputs["velocity"] = vel_path

        logger.info("  Results saved to %s", event_dir)
        return outputs

    def extract_results_raster(
        self,
        hdf_path: str | Path,
        event_name: str,
        reference_profile: dict,
        grid_shape: tuple[int, int],
        time_step: int = -1,
    ) -> dict[str, Path]:
        """Extract results and interpolate onto a regular 10 m raster grid."""
        from scipy.interpolate import griddata

        hdf_path = Path(hdf_path)
        event_dir = self.output_dir / event_name
        event_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(hdf_path, "r") as f:
            results_root = f["Results"]["Unsteady"]["Output"]["Output Blocks"]
            base_output = results_root["Base Output"]["Unsteady Time Series"]
            flow_areas = base_output["2D Flow Areas"]
            area_name = list(flow_areas.keys())[0]
            area = flow_areas[area_name]

            geom_areas = f["Geometry"]["2D Flow Areas"][area_name]
            cell_x = geom_areas["Cells Center Coordinate"][:][:, 0]
            cell_y = geom_areas["Cells Center Coordinate"][:][:, 1]

            depth_all = area["Depth"][:] if "Depth" in area else None

        if depth_all is None:
            logger.warning("No depth data in %s", hdf_path)
            return {}

        depth_values = depth_all[time_step]

        transform = reference_profile["transform"]
        height, width = grid_shape
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        grid_x = transform.c + cols * transform.a
        grid_y = transform.f + rows * transform.e

        logger.info("  Interpolating %d cells → %dx%d grid...",
                     len(cell_x), width, height)

        depth_grid = griddata(
            (cell_x, cell_y), depth_values,
            (grid_x, grid_y),
            method="linear",
            fill_value=0.0,
        ).astype(np.float32)

        depth_path = event_dir / "depth.tif"
        save_raster(depth_path, depth_grid, reference_profile)

        logger.info(
            "  HEC-RAS raster: max_depth=%.2f m, wet_cells=%d",
            depth_grid.max(), (depth_grid > 0.01).sum(),
        )
        return {"depth": depth_path}
