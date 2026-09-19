"""
Step ⑦ — Terrain feature engineering layer.

Derives 6 terrain-based rasters from the corrected 10 m DTM using
WhiteboxTools. These features are concatenated as additional input
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
import shutil

from src.utils.logging_config import get_logger

logger = get_logger("da.prep.terrain")


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
        return wbt

    def generate_all(
        self,
        dtm_path: str | Path,
        stream_threshold: int = 1000,
        river_shp_path: str | Path | None = None,
    ) -> dict[str, Path]:
        """Generate all 6 terrain feature rasters.

        Parameters
        ----------
        dtm_path : path
            Input DEM raster.
        stream_threshold : int
            Flow-accumulation cell count for WhiteboxTools stream extraction.
        river_shp_path : path, optional
            Real river/waterway shapefile.  When provided the
            distance-to-river raster is computed from this vector
            instead of from the DEM-derived stream network.
        """
        dtm_path = Path(dtm_path).resolve()
        wbt = self._get_wbt()

        work_dtm = (self.output_dir / "dtm.tif").resolve()
        if not work_dtm.exists():
            shutil.copy2(dtm_path, work_dtm)

        outputs = {}

        logger.info("Computing slope...")
        slope_path = (self.output_dir / "slope.tif").resolve()
        wbt.slope(dem=str(work_dtm), output=str(slope_path))
        outputs["slope"] = slope_path

        logger.info("Computing curvature...")
        curv_path = (self.output_dir / "curvature.tif").resolve()
        wbt.total_curvature(dem=str(work_dtm), output=str(curv_path))
        outputs["curvature"] = curv_path

        logger.info("Computing aspect...")
        aspect_path = (self.output_dir / "aspect.tif").resolve()
        wbt.aspect(dem=str(work_dtm), output=str(aspect_path))
        outputs["aspect"] = aspect_path

        logger.info("Filling depressions...")
        filled_path = (self.output_dir / "dtm_filled.tif").resolve()
        wbt.fill_depressions(dem=str(work_dtm), output=str(filled_path))

        logger.info("Computing flow direction (D8)...")
        flow_dir_path = (self.output_dir / "flow_dir.tif").resolve()
        wbt.d8_pointer(dem=str(filled_path), output=str(flow_dir_path))

        logger.info("Computing flow accumulation...")
        flow_acc_path = (self.output_dir / "flow_accum.tif").resolve()
        wbt.d8_flow_accumulation(
            i=str(filled_path), output=str(flow_acc_path),
            out_type="cells",
        )

        logger.info("Extracting stream network (threshold=%d cells)...", stream_threshold)
        streams_path = (self.output_dir / "streams.tif").resolve()
        wbt.extract_streams(
            flow_accum=str(flow_acc_path),
            output=str(streams_path),
            threshold=stream_threshold,
        )

        logger.info("Computing TWI...")
        twi_path = (self.output_dir / "twi.tif").resolve()
        wbt.wetness_index(
            sca=str(flow_acc_path),
            slope=str(slope_path),
            output=str(twi_path),
        )
        outputs["twi"] = twi_path

        # ── Distance to river ─────────────────────────────────────────
        logger.info("Computing distance to river...")
        dist_path = (self.output_dir / "distance_to_river.tif").resolve()

        if river_shp_path is not None:
            self._dist2river_from_shp(
                river_shp_path, work_dtm, dist_path,
            )
        else:
            wbt.euclidean_distance(
                i=str(streams_path),
                output=str(dist_path),
            )
        outputs["distance_to_river"] = dist_path

        logger.info("Computing HAND raster...")
        hand_path = (self.output_dir / "hand.tif").resolve()
        wbt.elevation_above_stream(
            dem=str(filled_path),
            streams=str(streams_path),
            output=str(hand_path),
        )
        outputs["hand"] = hand_path

        logger.info("All terrain features generated:")
        for name, path in outputs.items():
            logger.info("  %-20s -> %s", name, path.name)

        return outputs

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _dist2river_from_shp(
        river_shp_path: str | Path,
        dem_path: str | Path,
        out_path: str | Path,
    ) -> None:
        """Rasterize a river shapefile and compute euclidean distance (m)."""
        import numpy as np
        import rasterio
        import geopandas as gpd
        from rasterio.features import rasterize
        from scipy.ndimage import distance_transform_edt

        with rasterio.open(dem_path) as src:
            profile = dict(src.profile)
            transform = src.transform
            height, width = src.height, src.width
            dem_crs = src.crs
            res = abs(transform.a)  # pixel size in CRS units (metres for UTM)

        rivers = gpd.read_file(river_shp_path)
        if rivers.crs != dem_crs:
            rivers = rivers.to_crs(dem_crs)

        # Rasterize: river cells = 1, background = 0
        river_mask = rasterize(
            [(geom, 1) for geom in rivers.geometry],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8",
        )

        logger.info("  Rasterized %d river features onto %dx%d grid (%d river pixels).",
                     len(rivers), width, height, int(river_mask.sum()))

        # Euclidean distance in pixels, then multiply by resolution
        dist_pixels = distance_transform_edt(river_mask == 0)
        dist_metres = (dist_pixels * res).astype(np.float32)

        profile.update(dtype="float32", count=1, nodata=-9999.0)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(dist_metres, 1)

        logger.info("  Distance-to-river: range [%.1f, %.1f] m", dist_metres.min(), dist_metres.max())

    def get_feature_paths(self) -> dict[str, Path]:
        """Return a dict of expected feature raster paths (for use by FeatureEngineer).

        Only returns paths for features that exist on disk.
        """
        expected = {
            "slope": self.output_dir / "slope.tif",
            "aspect": self.output_dir / "aspect.tif",
            "distance_to_river": self.output_dir / "distance_to_river.tif",
            "hand": self.output_dir / "hand.tif",
            "twi": self.output_dir / "twi.tif",
            "curvature": self.output_dir / "curvature.tif",
        }
        return {k: v for k, v in expected.items() if v.exists()}

