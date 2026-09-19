"""
Step ① — ICESat-2 ATL03 photon cloud download & ground photon extraction.

Downloads ICESat-2 ATL03 data along the Red River using the `icepyx`
library, then extracts high-confidence ground-touching photons to be
used as training labels for the ANN DEM corrector (Step ④).
"""

from __future__ import annotations

import h5py
import numpy as np
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point

from src.utils.logging_config import get_logger
from src.utils.geo_utils import load_config

logger = get_logger("pigan.phase1.icesat2")


class ICESat2Downloader:
    """Download and process ICESat-2 ATL03 photon cloud data.

    Parameters
    ----------
    config : dict
        Full pipeline config (from ``base_config.yaml``).
    output_dir : str | Path
        Directory to save downloaded HDF5 files.
    """

    # ATL03 beam groups (3 strong beams preferred)
    BEAM_GROUPS = [
        "gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r",
    ]

    def __init__(self, config: dict, output_dir: str | Path | None = None):
        self.bbox = config["study_area"]["bbox"]
        icesat_cfg = config["data_sources"]["icesat2"]
        if "date_ranges" in icesat_cfg:
            self.date_ranges = icesat_cfg["date_ranges"]
        else:
            self.date_ranges = [icesat_cfg["date_range"]]
        self.confidence_threshold = icesat_cfg["confidence_threshold"]
        self.output_dir = Path(
            output_dir or Path(config["paths"]["data_raw"]) / "icesat2"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> list[Path]:
        """Query and download ATL03 granules covering the study area using `earthaccess`.

        Returns
        -------
        list[Path]
            Paths to downloaded HDF5 files.

        Notes
        -----
        Requires a valid NASA Earthdata Login.  Credentials should be
        stored in a ``.netrc`` file or will be prompted interactively.
        """
        import earthaccess

        west, south, east, north = self.bbox

        logger.info("Starting earthaccess search...")

        # 1. Login (interactive prompt if .netrc not present, persist=True saves .netrc)
        earthaccess.login(strategy="interactive", persist=True)

        # 2. Search data directly from NASA Harmony/Cloud
        results = []
        for dr in self.date_ranges:
            logger.info(
                "Querying ICESat-2 ATL03 via earthaccess: bbox=(%.2f, %.2f, %.2f, %.2f), dates=%s",
                west, south, east, north, dr,
            )
            r = earthaccess.search_data(
                short_name="ATL03",
                bounding_box=(west, south, east, north),
                temporal=tuple(dr),
            )
            results.extend(r)

        logger.info("Found %d granules via earthaccess", len(results))

        # 3. Check existing files for corruption (from partial/truncated downloads due to network drop)
        for existing_h5 in self.output_dir.glob("*.h5"):
            try:
                with h5py.File(existing_h5, "r") as f:
                    pass
            except (OSError, Exception) as e:
                logger.warning("Removing incomplete/truncated file before resume: %s (%s)", existing_h5.name, e)
                try:
                    existing_h5.unlink()
                except Exception:
                    pass

        # 4. Multi-threaded download (earthaccess automatically skips valid existing files)
        if results:
            earthaccess.download(results, str(self.output_dir), threads=8)

        downloaded = sorted(self.output_dir.glob("*.h5"))
        logger.info("Downloaded %d HDF5 files to %s", len(downloaded), self.output_dir)
        return downloaded

    def extract_ground_photons(
        self,
        hdf5_path: str | Path,
    ) -> gpd.GeoDataFrame:
        """Extract high-confidence ground photons from a single ATL03 file.

        Parameters
        ----------
        hdf5_path : str | Path
            Path to an ATL03 HDF5 file.

        Returns
        -------
        gpd.GeoDataFrame
            Columns: ``lon``, ``lat``, ``elevation``, ``beam``, ``confidence``.
            CRS: EPSG:4326.
        """
        hdf5_path = Path(hdf5_path)
        logger.info("Extracting ground photons from %s", hdf5_path.name)

        import pandas as pd
        beam_dfs = []

        # Spatial bounding box for early filtering (avoids loading millions
        # of photons outside the study area into memory as geometry objects).
        west, south, east, north = self.bbox

        empty_gdf = gpd.GeoDataFrame(
            columns=["lon", "lat", "elevation", "beam", "confidence", "geometry"],
            crs="EPSG:4326",
        )

        try:
            with h5py.File(hdf5_path, "r") as f:
                for beam in self.BEAM_GROUPS:
                    heights_group = f"{beam}/heights"
                    if heights_group not in f:
                        continue

                    try:
                        lats = f[f"{heights_group}/lat_ph"][:]
                        lons = f[f"{heights_group}/lon_ph"][:]
                        elevations = f[f"{heights_group}/h_ph"][:]
                        # signal_conf_ph shape: (n_photons, 5) —
                        # columns = [background, buffer, low, medium, high]
                        # For land classification, use column index 0 (land)
                        conf = f[f"{heights_group}/signal_conf_ph"][:]
                        if conf.ndim == 2:
                            conf_land = conf[:, 0]  # Land surface confidence
                        else:
                            conf_land = conf
                    except KeyError:
                        logger.debug("Missing keys in beam %s, skipping", beam)
                        continue

                    # ── Step 1: Spatial filter (cheap numpy ops) ──
                    bbox_mask = (
                        (lons >= west) & (lons <= east) &
                        (lats >= south) & (lats <= north)
                    )
                    if bbox_mask.sum() == 0:
                        logger.debug(
                            "  Beam %s: 0 / %d photons inside study area bbox, skipping",
                            beam, len(lats),
                        )
                        continue

                    # ── Step 2: Confidence filter (only on bbox-passing photons) ──
                    combined_mask = bbox_mask & (conf_land >= self.confidence_threshold)
                    n_valid = combined_mask.sum()

                    if n_valid == 0:
                        logger.debug(
                            "  Beam %s: %d in bbox but 0 passed confidence filter",
                            beam, int(bbox_mask.sum()),
                        )
                        continue

                    valid_idx = np.where(combined_mask)[0]

                    df = pd.DataFrame({
                        "lon": lons[valid_idx],
                        "lat": lats[valid_idx],
                        "elevation": elevations[valid_idx],
                        "beam": beam,
                        "confidence": conf_land[valid_idx]
                    })
                    beam_dfs.append(df)

                    logger.debug(
                        "  Beam %s: %d / %d photons passed bbox + confidence filter",
                        beam, n_valid, len(lats),
                    )
        except (OSError, Exception) as e:
            logger.error("Corrupted or incomplete HDF5 file %s (%s). Deleting so it can be re-downloaded.", hdf5_path.name, e)
            try:
                hdf5_path.unlink()
            except Exception:
                pass
            return empty_gdf

        if not beam_dfs:
            logger.debug("No ground photons in study area from %s", hdf5_path.name)
            return empty_gdf

        df_all = pd.concat(beam_dfs, ignore_index=True)

        # Geometry creation is now safe — df_all only contains photons
        # inside the study bbox (typically hundreds–thousands, not millions).
        gdf = gpd.GeoDataFrame(
            df_all,
            geometry=gpd.points_from_xy(df_all["lon"], df_all["lat"]),
            crs="EPSG:4326",
        )

        logger.info(
            "Extracted %d ground photons from %s", len(gdf), hdf5_path.name
        )
        return gdf

    def process_all(self, extract_only: bool = False) -> gpd.GeoDataFrame:
        """Download (if needed) and extract ground photons from all granules.

        Returns
        -------
        gpd.GeoDataFrame
            Merged ground photons from all files.
        """
        if extract_only:
            logger.info("Skipping download step, extracting from existing local files...")
            h5_files = sorted(self.output_dir.glob("*.h5"))
        else:
            # Always check and complete any missing/truncated granules via earthaccess
            h5_files = self.download()

        all_photons = []
        for h5_path in h5_files:
            gdf = self.extract_ground_photons(h5_path)
            if len(gdf) > 0:
                all_photons.append(gdf)

        if not all_photons:
            raise RuntimeError("No ground photons extracted from any file.")

        merged = gpd.pd.concat(all_photons, ignore_index=True)
        merged = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")

        # Remove duplicates (same location within ~1m)
        merged = merged.drop_duplicates(subset=["lon", "lat"])

        # Save to disk
        out_path = self.output_dir / "ground_photons.gpkg"
        merged.to_file(out_path, driver="GPKG")
        logger.info(
            "Total ground photons: %d → saved to %s", len(merged), out_path
        )

        return merged
