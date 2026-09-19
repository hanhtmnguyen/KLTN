"""
Step ① — ICESat-2 ATL03 & ATL08 Data Fusion for Ground Photon Extraction.

Downloads ICESat-2 ATL03 and ATL08 granules, and fuses them by mapping
ATL08 officially filtered ground labels (classed_pc_flag == 1) back to
ATL03 high-resolution photon coordinates via index mapping:
Global_Index_ATL03 = classed_pc_indx_ATL08 + ph_index_beg_ATL03 - 1

Applies strict quality flags (SNR, cloud, MSW).
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from collections import defaultdict

from src.utils.logging_config import get_logger
from src.utils.geo_utils import load_config

logger = get_logger("da.data.icesat2_atl03")


class ICESat2Downloader:
    """Download and process ICESat-2 ATL03/ATL08 photon cloud data.

    Parameters
    ----------
    config : dict
        Full pipeline config.
    output_dir : str | Path
        Directory to save downloaded HDF5 files.
    """

    BEAM_GROUPS = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]

    def __init__(self, config: dict, output_dir: str | Path | None = None):
        self.config = config
        self.bbox = config["study_area"]["bbox"]
        icesat_cfg = config["data_sources"]["icesat2"]
        
        self.date_ranges = icesat_cfg.get("date_ranges", [icesat_cfg.get("date_range")])
        self.products = icesat_cfg.get("products", ["ATL03", "ATL08"])
        
        qf = icesat_cfg.get("quality_flags", {})
        self.snr_min = qf.get("snr_min", 10)
        self.cloud_max = qf.get("cloud_flag_atm_max", 2)
        self.msw_max = qf.get("msw_flag_max", 3)
        
        self.output_dir = Path(
            output_dir or Path(config["paths"]["data_raw"]) / "icesat2_atl03"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> dict[str, dict[str, Path]]:
        """Query and download ATL03/ATL08 granules via earthaccess.

        Returns
        -------
        dict[str, dict[str, Path]]
            Dictionary pairing ATL03 and ATL08 paths by granule ID.
            e.g. {"20201014...": {"ATL03": path3, "ATL08": path8}}
        """
        import earthaccess
        west, south, east, north = self.bbox

        logger.info("Starting earthaccess login...")
        earthaccess.login(strategy="interactive", persist=True)

        for product in self.products:
            results = []
            for dr in self.date_ranges:
                logger.info("Querying %s: bbox=%s, dates=%s", product, self.bbox, dr)
                r = earthaccess.search_data(
                    short_name=product,
                    bounding_box=(west, south, east, north),
                    temporal=tuple(dr),
                )
                results.extend(r)
            
            logger.info("Found %d %s granules.", len(results), product)
            if results:
                earthaccess.download(results, str(self.output_dir), threads=1)

        # Pair up ATL03 and ATL08 files
        # Naming convention: ATL03_YYYYMMDDHHMMSS_TTTTCCSS_VVV_RR.h5
        pairs = defaultdict(dict)
        for h5_file in self.output_dir.glob("*.h5"):
            # Ensure file is readable
            try:
                with h5py.File(h5_file, "r") as f:
                    pass
            except Exception as e:
                logger.warning("Removing corrupted file: %s (%s)", h5_file.name, e)
                h5_file.unlink(missing_ok=True)
                continue

            name_parts = h5_file.stem.split("_")
            if len(name_parts) >= 4:
                product = name_parts[0]
                granule_id = "_".join(name_parts[1:4])  # YYYYMMDDHHMMSS_TTTTCCSS_VVV
                pairs[granule_id][product] = h5_file

        # Filter to only complete pairs
        complete_pairs = {
            gid: paths for gid, paths in pairs.items() 
            if "ATL03" in paths and "ATL08" in paths
        }
        logger.info("Found %d complete ATL03/ATL08 overlapping pairs.", len(complete_pairs))
        return complete_pairs

    def process_pair(
        self,
        atl03_path: Path,
        atl08_path: Path,
        granule_id: str,
    ) -> gpd.GeoDataFrame:
        """Extract ground photons using ATL08 labels and ATL03 coordinates."""
        logger.info("Processing pair: %s", granule_id)

        # Extract track_id from granule_id (TTTT is first 4 chars of the second part)
        parts = granule_id.split("_")
        track_id = parts[1][:4] if len(parts) > 1 and len(parts[1]) >= 4 else granule_id

        beam_dfs = []
        try:
            with h5py.File(atl03_path, "r") as f3, h5py.File(atl08_path, "r") as f8:
                # Read spacecraft orientation to label strong/weak later, but process all beams
                sc_orient = f3["/orbit_info/sc_orient"][0] if "/orbit_info/sc_orient" in f3 else 2
                
                if sc_orient == 1:
                    strong_beams = ["gt1r", "gt2r", "gt3r"]
                elif sc_orient == 0:
                    strong_beams = ["gt1l", "gt2l", "gt3l"]
                else:
                    strong_beams = []
                
                logger.debug("sc_orient=%s: Extracting both strong and weak beams", sc_orient)

                for beam in self.BEAM_GROUPS:
                    if f"{beam}/heights" not in f3 or f"{beam}/signal_photons" not in f8:
                        continue
                    
                    # ─── 1. ATL08 Data (Labels & Segment Quality) ───
                    # Photon-level in ATL08
                    classed_pc_flag = f8[f"{beam}/signal_photons/classed_pc_flag"][:]
                    classed_pc_indx = f8[f"{beam}/signal_photons/classed_pc_indx"][:]
                    ph_segment_id = f8[f"{beam}/signal_photons/ph_segment_id"][:]
                    
                    # Segment-level in ATL08
                    seg_id_8 = f8[f"{beam}/land_segments/segment_id_beg"][:]
                    cloud_flag = f8[f"{beam}/land_segments/cloud_flag_atm"][:]
                    msw_flag = f8[f"{beam}/land_segments/msw_flag"][:]
                    snr = f8[f"{beam}/land_segments/snr_signif"][:] if "snr_signif" in f8[f"{beam}/land_segments"] else f8[f"{beam}/land_segments/snr"][:]

                    # Filter for Ground (1)
                    ground_mask = (classed_pc_flag == 1)
                    if not ground_mask.any():
                        continue

                    # Extract ground photons
                    classed_pc_indx_g = classed_pc_indx[ground_mask]
                    ph_segment_id_g = ph_segment_id[ground_mask]
                    
                    # ─── 2. Map Quality Flags from Segments to Photons ───
                    # Find indices of ph_segment_id_g in seg_id_8
                    # searchsorted requires the array to be sorted, which segment IDs are.
                    seg_indices = np.searchsorted(seg_id_8, ph_segment_id_g)
                    
                    # Ensure indices are within bounds and exactly match
                    valid_seg_mask = (seg_indices < len(seg_id_8)) & (seg_id_8[np.clip(seg_indices, 0, len(seg_id_8)-1)] == ph_segment_id_g)
                    
                    # Filter ground photons that have matching segments
                    classed_pc_indx_g = classed_pc_indx_g[valid_seg_mask]
                    ph_segment_id_g = ph_segment_id_g[valid_seg_mask]
                    seg_indices = seg_indices[valid_seg_mask]

                    photon_clouds = cloud_flag[seg_indices]
                    photon_msw = msw_flag[seg_indices]
                    photon_snr = snr[seg_indices]

                    # Apply strict quality filters
                    qf_mask = (
                        (photon_clouds <= self.cloud_max) & 
                        (photon_msw <= self.msw_max) & 
                        (photon_snr > self.snr_min)
                    )
                    
                    classed_pc_indx_q = classed_pc_indx_g[qf_mask]
                    ph_segment_id_q = ph_segment_id_g[qf_mask]
                    
                    if len(classed_pc_indx_q) == 0:
                        continue

                    # ─── 3. Map to ATL03 High-Res Coordinates ───
                    # ATL03 geolocation mapping
                    seg_id_3 = f3[f"{beam}/geolocation/segment_id"][:]
                    ph_index_beg_3 = f3[f"{beam}/geolocation/ph_index_beg"][:]
                    geoid_3 = f3[f"{beam}/geophys_corr/geoid"][:] if f"{beam}/geophys_corr/geoid" in f3 else np.zeros_like(seg_id_3, dtype=float)
                    
                    seg3_indices = np.searchsorted(seg_id_3, ph_segment_id_q)
                    valid_seg3_mask = (seg3_indices < len(seg_id_3)) & (seg_id_3[np.clip(seg3_indices, 0, len(seg_id_3)-1)] == ph_segment_id_q)
                    
                    classed_pc_indx_q = classed_pc_indx_q[valid_seg3_mask]
                    seg3_indices = seg3_indices[valid_seg3_mask]
                    
                    # Equation: Global_Index = classed_pc_indx_ATL08 + ph_index_beg_ATL03 - 1
                    ph_index_beg = ph_index_beg_3[seg3_indices]
                    geoid_vals = geoid_3[seg3_indices]
                    global_index_1based = classed_pc_indx_q + ph_index_beg - 1
                    python_index = global_index_1based - 1  # 0-based for numpy indexing
                    
                    # Ensure indices don't exceed array bounds
                    max_idx = len(f3[f"{beam}/heights/h_ph"]) - 1
                    valid_idx_mask = (python_index >= 0) & (python_index <= max_idx)
                    python_index = python_index[valid_idx_mask]
                    geoid_vals = geoid_vals[valid_idx_mask]
                    
                    if len(python_index) == 0:
                        continue

                    lats = f3[f"{beam}/heights/lat_ph"][:][python_index]
                    lons = f3[f"{beam}/heights/lon_ph"][:][python_index]
                    elevs = f3[f"{beam}/heights/h_ph"][:][python_index]
                    
                    beam_strength = "strong" if beam in strong_beams else "weak"
                    if sc_orient not in [0, 1]:
                        beam_strength = "unknown"

                    df = pd.DataFrame({
                        "lon": lons,
                        "lat": lats,
                        "elevation": elevs,
                        "geoid_h": geoid_vals,
                        "beam": beam,
                        "beam_strength": beam_strength,
                        "track_id": track_id,
                    })
                    beam_dfs.append(df)
                    logger.debug("  Beam %s: %d strict ground photons extracted", beam, len(lats))

        except (OSError, Exception) as e:
            logger.error("Error processing pair %s: %s", granule_id, e)
            return gpd.GeoDataFrame()

        if not beam_dfs:
            logger.warning("No ground photons passed quality filters in %s", granule_id)
            return gpd.GeoDataFrame()

        df_all = pd.concat(beam_dfs, ignore_index=True)
        gdf = gpd.GeoDataFrame(
            df_all,
            geometry=gpd.points_from_xy(df_all["lon"], df_all["lat"]),
            crs="EPSG:4326",
        )
        logger.info("Extracted %d high-quality ground photons from %s", len(gdf), granule_id)
        return gdf

    def process_all(self, extract_only: bool = False) -> gpd.GeoDataFrame:
        """Process all ATL03/ATL08 pairs and merge them into a single GeoDataFrame."""
        if extract_only:
            logger.info("Skipping download step, pairing existing local files...")
            pairs = defaultdict(dict)
            for h5_file in self.output_dir.glob("*.h5"):
                name_parts = h5_file.stem.split("_")
                if len(name_parts) >= 4:
                    product = name_parts[0]
                    granule_id = "_".join(name_parts[1:4])
                    pairs[granule_id][product] = h5_file
            
            complete_pairs = {
                gid: paths for gid, paths in pairs.items() 
                if "ATL03" in paths and "ATL08" in paths
            }
        else:
            complete_pairs = self.download()

        if not complete_pairs:
            raise RuntimeError("No complete ATL03/ATL08 pairs found to process.")

        all_photons = []
        for granule_id, paths in complete_pairs.items():
            gdf = self.process_pair(paths["ATL03"], paths["ATL08"], granule_id)
            if not gdf.empty:
                all_photons.append(gdf)

        if not all_photons:
            raise RuntimeError("No ground photons extracted from any file after filtering.")

        merged = pd.concat(all_photons, ignore_index=True)
        merged = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")

        # Remove duplicates
        merged = merged.drop_duplicates(subset=["lon", "lat"])

        out_path = self.output_dir / "ground_photons.gpkg"
        merged.to_file(out_path, driver="GPKG")
        logger.info("Total ground photons: %d → saved to %s", len(merged), out_path)

        return merged
