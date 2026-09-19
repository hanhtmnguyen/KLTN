"""
Step ⑦ — Terrain feature engineering layer.

Derives 6 terrain-based rasters from the corrected 10 m DTM using
WhiteboxTools.  These features are concatenated as additional input
channels to the PI-GAN generator, giving it landscape context.

Features
--------
1. Slope           — terrain steepness (degrees)
2. Curvature       — surface concavity/convexity
3. TWI             — Topographic Wetness Index = ln(α / tan β)
4. Distance-to-river — Euclidean distance to nearest stream cell
5. HAND raster     — Height Above Nearest Drainage
6. Aspect          — flow direction (compass bearing)
"""

from __future__ import annotations

from pathlib import Path

from src.utils.logging_config import get_logger

logger = get_logger("pigan.phase1.terrain")


class TerrainFeatureGenerator:
    """Derive terrain features from a DEM using WhiteboxTools.

    Parameters
    ----------
    config : dict
        Full pipeline configuration.
    """

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config["paths"]["data_processed"]) / "terrain_features"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_wbt(self):
        """Initialise WhiteboxTools with verbose off."""
        from whitebox import WhiteboxTools

        wbt = WhiteboxTools()
        wbt.set_verbose_mode(False)
        wbt.set_working_dir(str(self.output_dir))
        return wbt

    def generate_all(self, dtm_path: str | Path, stream_threshold: int = 1000) -> dict[str, Path]:
        """Generate all 6 terrain feature rasters.

        Parameters
        ----------
        dtm_path : str | Path
            Path to the corrected 10 m DTM GeoTIFF.
        stream_threshold : int
            Flow-accumulation cell count threshold for stream extraction.

        Returns
        -------
        dict[str, Path]
            Mapping of feature name to output raster path.
        """
        dtm_path = Path(dtm_path)
        wbt = self._get_wbt()

        # Copy DTM to working directory if not already there
        import shutil
        work_dtm = self.output_dir / "dtm.tif"
        if not work_dtm.exists():
            shutil.copy2(dtm_path, work_dtm)

        outputs = {}

        # ---- 1. Slope ----
        logger.info("Computing slope...")
        slope_path = self.output_dir / "slope.tif"
        wbt.slope(dem=str(work_dtm), output=str(slope_path))
        outputs["slope"] = slope_path

        # ---- 2. Curvature ----
        logger.info("Computing curvature...")
        curv_path = self.output_dir / "curvature.tif"
        wbt.total_curvature(dem=str(work_dtm), output=str(curv_path))
        outputs["curvature"] = curv_path

        # ---- 3. Aspect ----
        logger.info("Computing aspect...")
        aspect_path = self.output_dir / "aspect.tif"
        wbt.aspect(dem=str(work_dtm), output=str(aspect_path))
        outputs["aspect"] = aspect_path

        # ---- 4–6 require hydrological processing ----
        logger.info("Filling depressions...")
        filled_path = self.output_dir / "dtm_filled.tif"
        wbt.fill_depressions(dem=str(work_dtm), output=str(filled_path))

        logger.info("Computing flow direction (D8)...")
        flow_dir_path = self.output_dir / "flow_dir.tif"
        wbt.d8_pointer(dem=str(filled_path), output=str(flow_dir_path))

        logger.info("Computing flow accumulation...")
        flow_acc_path = self.output_dir / "flow_accum.tif"
        wbt.d8_flow_accumulation(
            i=str(filled_path), output=str(flow_acc_path),
            out_type="cells",
        )

        logger.info("Extracting stream network (threshold=%d cells)...", stream_threshold)
        streams_path = self.output_dir / "streams.tif"
        wbt.extract_streams(
            flow_accum=str(flow_acc_path),
            output=str(streams_path),
            threshold=stream_threshold,
        )

        # ---- 4. TWI (Topographic Wetness Index) ----
        logger.info("Computing TWI...")
        twi_path = self.output_dir / "twi.tif"
        wbt.wetness_index(
            sca=str(flow_acc_path),
            slope=str(slope_path),
            output=str(twi_path),
        )
        outputs["twi"] = twi_path

        # ---- 5. Distance to River ----
        logger.info("Computing distance to river...")
        dist_path = self.output_dir / "distance_to_river.tif"
        wbt.euclidean_distance(
            i=str(streams_path),
            output=str(dist_path),
        )
        outputs["distance_to_river"] = dist_path

        # ---- 6. HAND (Height Above Nearest Drainage) ----
        logger.info("Computing HAND raster...")
        hand_path = self.output_dir / "hand.tif"
        wbt.elevation_above_stream(
            dem=str(filled_path),
            streams=str(streams_path),
            output=str(hand_path),
        )
        outputs["hand"] = hand_path

        # ---- Summary ----
        logger.info("All terrain features generated:")
        for name, path in outputs.items():
            logger.info("  %-20s → %s", name, path.name)

        return outputs
