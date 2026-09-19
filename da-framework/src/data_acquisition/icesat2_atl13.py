"""
ICESat-2 ATL13 Inland Water Surface Height Downloader & Extractor.

Queries and downloads ICESat-2 ATL13 granules along inland water bodies
(Sông Hồng / Red River) and extracts along-track Water Surface Elevation (WSE)
observations. These sparse WSE profiles serve as downstream data assimilation
constraints (`L_downstream`) during PI-GAN training.
"""

from __future__ import annotations

import h5py
import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely.geometry import Point

from src.utils.logging_config import get_logger

logger = get_logger("da.data.icesat2_atl13")


class ICESat2ATL13Downloader:
    """Download and process ICESat-2 ATL13 inland water surface height data.

    Parameters
    ----------
    config : dict
        Full pipeline configuration dictionary.
    output_dir : str | Path | None
        Directory to store raw HDF5 files and extracted WSE tables.
    """

    BEAM_GROUPS = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]

    def __init__(self, config: dict, output_dir: str | Path | None = None):
        self.bbox = config["study_area"]["bbox"]
        self.date_range = config["data_sources"]["icesat2_atl13"]["date_range"]
        self.quality_threshold = config["data_sources"]["icesat2_atl13"]["quality_threshold"]
        self.output_dir = Path(
            output_dir or Path(config["paths"]["data_raw"]) / "icesat2_atl13"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> list[Path]:
        """Query and download ATL13 granules covering the study area bounding box.

        Returns
        -------
        list[Path]
            Paths to downloaded ATL13 HDF5 files.
        """
        import icepyx as ipx

        logger.info(
            "Querying ICESat-2 ATL13: bbox=%s, dates=%s",
            self.bbox, self.date_range,
        )

        query = ipx.Query(
            "ATL13",
            self.bbox,
            self.date_range,
        )

        avail = query.avail_granules()
        logger.info("Found %d ATL13 granules", len(avail))

        if not avail:
            logger.warning("No ATL13 granules found for the specified query.")
            return []

        query.earthdata_login()
        query.download_all(path=str(self.output_dir))

        downloaded = sorted(self.output_dir.glob("*.h5"))
        logger.info("Downloaded %d ATL13 files to %s", len(downloaded), self.output_dir)
        return downloaded

    def extract_water_surface(self, hdf5_path: str | Path) -> gpd.GeoDataFrame:
        """Extract valid water surface height observations from an ATL13 HDF5 file.

        Parameters
        ----------
        hdf5_path : str | Path
            Path to the ATL13 `.h5` granule.

        Returns
        -------
        gpd.GeoDataFrame
            Columns: `lon`, `lat`, `datetime`, `wse`, `quality_flag`, `beam`.
            CRS: EPSG:4326.
        """
        hdf5_path = Path(hdf5_path)
        logger.info("Extracting ATL13 water surface heights from %s", hdf5_path.name)

        records = []

        with h5py.File(hdf5_path, "r") as f:
            # Extract granule start UTC time if available
            try:
                sc_time = f["ancillary_data/data_start_utc"][:][0].decode("utf-8")
            except Exception:
                sc_time = "2023-01-01T12:00:00Z"

            for beam in self.BEAM_GROUPS:
                if beam not in f:
                    continue

                try:
                    lats = f[f"{beam}/segment_lat"][:]
                    lons = f[f"{beam}/segment_lon"][:]
                    # ht_water_surf: orthometric water surface height
                    wse = f[f"{beam}/ht_water_surf"][:]
                    # qf_bckgrd / quality flags or segment quality
                    if f"{beam}/qf_bckgrd" in f:
                        qf = f[f"{beam}/qf_bckgrd"][:]
                    else:
                        qf = np.ones_like(wse, dtype=int)
                except KeyError:
                    logger.debug("Missing ATL13 keys in beam %s of %s", beam, hdf5_path.name)
                    continue

                # Filter valid WSE (nodata is typically > 1e30 or < -1000) and check quality flag
                valid_mask = (wse > -100.0) & (wse < 9000.0) & (qf <= self.quality_threshold)
                n_valid = np.sum(valid_mask)

                if n_valid == 0:
                    continue

                for i in np.where(valid_mask)[0]:
                    records.append({
                        "lon": float(lons[i]),
                        "lat": float(lats[i]),
                        "datetime": pd.to_datetime(sc_time),
                        "wse": float(wse[i]),
                        "quality_flag": int(qf[i]),
                        "beam": beam,
                    })

                logger.debug("  Beam %s: %d valid WSE observations extracted", beam, n_valid)

        if not records:
            return gpd.GeoDataFrame(
                columns=["lon", "lat", "datetime", "wse", "quality_flag", "beam", "geometry"],
                crs="EPSG:4326",
            )

        gdf = gpd.GeoDataFrame(
            records,
            geometry=[Point(r["lon"], r["lat"]) for r in records],
            crs="EPSG:4326",
        )
        return gdf

    def process_all(self) -> gpd.GeoDataFrame:
        """Download and extract WSE from all available ATL13 granules.

        Returns
        -------
        gpd.GeoDataFrame
            Merged ATL13 water surface elevation observations across all dates/tracks.
        """
        h5_files = sorted(self.output_dir.glob("*.h5"))
        if not h5_files:
            logger.info("No local ATL13 granules found, initiating download...")
            h5_files = self.download()

        all_records = []
        for h5_path in h5_files:
            gdf = self.extract_water_surface(h5_path)
            if not gdf.empty:
                all_records.append(gdf)

        if not all_records:
            logger.warning("No valid ATL13 water surface observations extracted.")
            return gpd.GeoDataFrame(
                columns=["lon", "lat", "datetime", "wse", "quality_flag", "beam", "geometry"],
                crs="EPSG:4326",
            )

        merged = gpd.pd.concat(all_records, ignore_index=True)
        merged = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")
        merged = merged.sort_values("datetime").reset_index(drop=True)

        out_path = self.output_dir / "atl13_water_surface.gpkg"
        merged.to_file(out_path, driver="GPKG")
        logger.info("Total ATL13 WSE observations: %d → saved to %s", len(merged), out_path)
        return merged
