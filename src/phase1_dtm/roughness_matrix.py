"""
Step ⑥ — Manning's roughness coefficient matrix generation.

Converts building footprints and land cover into a spatially variable
Manning's n raster at 10 m resolution, used as both a HEC-RAS input
and a PI-GAN conditioning channel.
"""

from __future__ import annotations

import numpy as np
import geopandas as gpd
from pathlib import Path

from src.utils.logging_config import get_logger
from src.utils.geo_utils import (
    load_raster,
    save_raster,
    rasterize_vector,
)

logger = get_logger("pigan.phase1.roughness")


class RoughnessMatrixGenerator:
    """Generate a Manning's n roughness raster from building footprints + land cover.

    Parameters
    ----------
    config : dict
        Full pipeline configuration.
    """

    def __init__(self, config: dict):
        self.config = config
        self.n_values = config["mannings_n"]
        self.output_dir = Path(config["paths"]["data_processed"]) / "roughness"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        buildings_path: str | Path,
        dtm_path: str | Path,
        sentinel2_path: str | Path | None = None,
    ) -> Path:
        """Create Manning's n raster.

        The roughness is determined by:
        1. Building footprints (highest n values)
        2. NDVI-based vegetation classification (from Sentinel-2)
        3. Background floodplain (default bare earth)

        Parameters
        ----------
        buildings_path : str | Path
            GeoPackage with classified building footprints.
        dtm_path : str | Path
            Reference 10m DTM (used for grid definition).
        sentinel2_path : str | Path | None
            Sentinel-2 composite for NDVI/NDWI classification.
            If None, only building-based roughness is applied.

        Returns
        -------
        Path
            Path to the Manning's n GeoTIFF.
        """
        output_path = self.output_dir / "manning_n_10m.tif"
        logger.info("Generating Manning's n roughness matrix...")

        # Load reference grid
        dtm, profile = load_raster(dtm_path)
        height, width = dtm.shape

        # Initialise with default floodplain roughness
        n_raster = np.full((height, width), self.n_values["floodplain_bare"],
                           dtype=np.float32)

        # ---- Apply vegetation-based roughness (from Sentinel-2 NDVI) ----
        if sentinel2_path is not None:
            logger.info("  Computing NDVI for vegetation roughness...")
            from src.utils.geo_utils import load_raster_multiband
            import rasterio
            from rasterio.enums import Resampling

            with rasterio.open(sentinel2_path) as src:
                s2 = src.read(
                    out_shape=(src.count, height, width),
                    resampling=Resampling.bilinear,
                ).astype(np.float32)

            # NDVI = (NIR - Red) / (NIR + Red) = (B8 - B4) / (B8 + B4)
            # Bands order: B2(0), B3(1), B4(2), B8(3)
            nir = s2[3]
            red = s2[2]
            ndvi = np.where(
                (nir + red) > 0,
                (nir - red) / (nir + red + 1e-10),
                0.0,
            )

            # NDWI = (Green - NIR) / (Green + NIR) = (B3 - B8) / (B3 + B8)
            green = s2[1]
            ndwi = np.where(
                (green + nir) > 0,
                (green - nir) / (green + nir + 1e-10),
                0.0,
            )

            # Classify and assign
            water_mask = ndwi > 0.3
            veg_mask = ndvi > 0.3

            n_raster[water_mask] = self.n_values["open_water"]
            n_raster[veg_mask & ~water_mask] = self.n_values["floodplain_vegetated"]

            logger.info(
                "  Water cells: %d, Vegetation cells: %d",
                water_mask.sum(), veg_mask.sum(),
            )

        # ---- Apply building-based roughness ----
        logger.info("  Rasterizing building footprints...")
        buildings = gpd.read_file(buildings_path)

        # Map building class → Manning's n
        class_to_n = {
            "residential": self.n_values["light_residential"],
            "commercial": self.n_values["commercial_industrial"],
            "industrial": self.n_values["commercial_industrial"],
            "other": self.n_values["dense_residential"],
        }

        buildings["manning_n"] = buildings["building_class"].map(class_to_n)
        buildings["manning_n"] = buildings["manning_n"].fillna(
            self.n_values["dense_residential"]
        )

        # Rasterize buildings onto the grid
        building_n_path = self.output_dir / "_temp_building_n.tif"
        rasterize_vector(
            gdf=buildings,
            value_column="manning_n",
            reference_raster=dtm_path,
            output_path=building_n_path,
            fill_value=0.0,
        )

        building_n, _ = load_raster(building_n_path)

        # Overlay: buildings override the base roughness
        building_mask = building_n > 0
        n_raster[building_mask] = building_n[building_mask]

        logger.info(
            "  Building cells overridden: %d (%.1f%% of domain)",
            building_mask.sum(),
            100 * building_mask.sum() / n_raster.size,
        )

        # ---- Save ----
        save_raster(output_path, n_raster, profile)
        logger.info("Manning's n saved → %s", output_path)

        # Clean temp file
        building_n_path.unlink(missing_ok=True)

        return output_path
