"""
Step ③ — FabDEM 30 m bare-earth DSM download.

Downloads FabDEM tiles for the study area and merges them into a
single GeoTIFF, which serves as the coarse starting point for
ANN-based DEM correction (Step ④).
"""

from __future__ import annotations

from pathlib import Path

from src.utils.logging_config import get_logger
from src.utils.geo_utils import load_config, reproject_raster

logger = get_logger("pigan.phase1.fabdem")


class FabDEMDownloader:
    """Download and merge FabDEM tiles.

    Parameters
    ----------
    config : dict
        Full pipeline config.
    output_dir : str | Path | None
        Where to store the downloaded/merged DEM.
    """

    def __init__(self, config: dict, output_dir: str | Path | None = None):
        self.bbox = config["study_area"]["bbox"]
        self.crs_projected = config["study_area"]["crs_projected"]
        self.output_dir = Path(
            output_dir or Path(config["paths"]["data_raw"]) / "fabdem"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> Path:
        """Download FabDEM tiles and merge into a single GeoTIFF.

        Uses the ``fabdem`` Python package which handles tile selection
        and stitching automatically.

        Returns
        -------
        Path
            Path to the merged GeoTIFF (in geographic CRS, ~30 m).
        """
        import fabdem as fabdem_pkg

        output_path = self.output_dir / "fabdem_raw.tif"

        if output_path.exists():
            logger.info("FabDEM already downloaded: %s", output_path)
            return output_path

        west, south, east, north = self.bbox
        logger.info(
            "Downloading FabDEM: bbox=(%.2f, %.2f, %.2f, %.2f)",
            west, south, east, north,
        )

        fabdem_pkg.download(
            bounds=(west, south, east, north),
            output_path=str(output_path),
            cache=self.output_dir / "cache",
            show_progress=True,
        )

        logger.info("FabDEM downloaded → %s", output_path)
        return output_path

    def download_and_reproject(self) -> Path:
        """Download FabDEM and reproject to the project CRS (UTM).

        Returns
        -------
        Path
            Path to the reprojected GeoTIFF.
        """
        raw_path = self.download()

        reprojected_path = self.output_dir / "fabdem_utm.tif"
        if reprojected_path.exists():
            logger.info("Reprojected FabDEM already exists: %s", reprojected_path)
            return reprojected_path

        reproject_raster(
            src_path=raw_path,
            dst_path=reprojected_path,
            dst_crs=self.crs_projected,
        )
        return reprojected_path
