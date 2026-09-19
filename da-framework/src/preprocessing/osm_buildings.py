"""
Step ⑤ — OpenStreetMap building footprint extraction.

Downloads building polygons from OSM for the study area using ``osmnx``,
classifies building types, and exports as GeoPackage.
"""

from __future__ import annotations

import geopandas as gpd
from pathlib import Path

from src.utils.logging_config import get_logger

logger = get_logger("da.prep.osm")


class OSMBuildingExtractor:
    """Extract building footprints from OpenStreetMap.

    Parameters
    ----------
    config : dict
        Full pipeline configuration.
    output_dir : str | Path | None
        Directory for output files.
    """

    def __init__(self, config: dict, output_dir: str | Path | None = None):
        self.bbox = config["study_area"]["bbox"]
        self.crs_projected = config["study_area"]["crs_projected"]
        self.output_dir = Path(
            output_dir or Path(config["paths"]["data_raw"]) / "osm"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_buildings(self) -> gpd.GeoDataFrame:
        """Download all building footprints within the study area."""
        import osmnx as ox

        west, south, east, north = self.bbox
        logger.info(
            "Downloading OSM buildings: bbox=(%.3f, %.3f, %.3f, %.3f)",
            west, south, east, north,
        )

        buildings = ox.features_from_bbox(
            bbox=(north, south, east, west),
            tags={"building": True},
        )

        buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]

        logger.info("Downloaded %d building footprints", len(buildings))
        return buildings

    def classify_buildings(
        self, buildings: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Classify buildings into categories for Manning's n assignment."""
        residential_tags = {
            "yes", "house", "residential", "apartments", "detached",
            "semidetached_house", "terrace", "dormitory",
        }
        commercial_tags = {
            "commercial", "retail", "office", "hotel", "supermarket",
        }
        industrial_tags = {
            "industrial", "warehouse", "factory", "manufacture",
        }

        def _classify(val):
            if not isinstance(val, str):
                return "other"
            val = val.lower().strip()
            if val in residential_tags:
                return "residential"
            elif val in commercial_tags:
                return "commercial"
            elif val in industrial_tags:
                return "industrial"
            else:
                return "other"

        buildings = buildings.copy()

        if "building" in buildings.columns:
            buildings["building_class"] = buildings["building"].apply(_classify)
        else:
            buildings["building_class"] = "other"

        counts = buildings["building_class"].value_counts()
        for cls, cnt in counts.items():
            logger.info("  %s: %d buildings", cls, cnt)

        return buildings

    def process(self) -> gpd.GeoDataFrame:
        """Download, classify, reproject, and save building footprints."""
        output_path = self.output_dir / "buildings.gpkg"

        if output_path.exists():
            logger.info("Loading cached buildings from %s", output_path)
            return gpd.read_file(output_path)

        buildings = self.download_buildings()
        buildings = self.classify_buildings(buildings)

        buildings = buildings.to_crs(self.crs_projected)
        buildings["area_m2"] = buildings.geometry.area

        keep_cols = [
            "geometry", "building", "building_class", "area_m2",
        ]
        available = [c for c in keep_cols if c in buildings.columns]
        buildings = buildings[available]

        buildings.to_file(output_path, driver="GPKG")
        logger.info("Buildings saved → %s (%d features)", output_path, len(buildings))

        return buildings
