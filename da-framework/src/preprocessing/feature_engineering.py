"""
Step ②b — Spatial Feature Engineering (10-Feature Builder).

Constructs the unified 10-feature matrix used by all DEM correction
models (Residual MLP, XGBoost, LightGBM).  Features are sampled at
ICESat-2 photon locations for training and computed wall-to-wall
for full-area DTM prediction.

Features
--------
1. B2  — Sentinel-2 Blue reflectance
2. B3  — Sentinel-2 Green reflectance
3. B4  — Sentinel-2 Red reflectance
4. B8  — Sentinel-2 NIR reflectance
5. NDVI — Normalised Difference Vegetation Index  (B8−B4)/(B8+B4)
6. MNDWI — Modified Normalised Difference Water Index  (B3−B11)/(B3+B11)
7. FabDEM elevation — resampled to 10 m (bilinear)
8. Slope — terrain steepness (degrees) from FabDEM
9. Aspect — terrain flow direction (degrees) from FabDEM
10. Dist2River — Euclidean distance to nearest stream cell (metres)
"""

from __future__ import annotations

import numpy as np
import geopandas as gpd
from pathlib import Path

from src.utils.logging_config import get_logger
from src.utils.geo_utils import (
    load_raster,
    load_raster_multiband,
    sample_raster_at_points,
    save_raster,
)

logger = get_logger("da.prep.features")


# ── Index computation helpers ─────────────────────────────────────────────

def compute_ndvi(b8: np.ndarray, b4: np.ndarray) -> np.ndarray:
    """NDVI = (B8 − B4) / (B8 + B4), clipped to [−1, 1]."""
    denom = b8 + b4
    ndvi = np.divide((b8 - b4), denom, out=np.zeros_like(denom, dtype=np.float32), where=denom!=0)
    return np.clip(ndvi, -1.0, 1.0).astype(np.float32)


def compute_mndwi(b3: np.ndarray, b11: np.ndarray) -> np.ndarray:
    """MNDWI = (B3 − B11) / (B3 + B11), clipped to [−1, 1]."""
    denom = b3 + b11
    mndwi = np.divide((b3 - b11), denom, out=np.zeros_like(denom, dtype=np.float32), where=denom!=0)
    return np.clip(mndwi, -1.0, 1.0).astype(np.float32)


# ── Feature Engineer class ───────────────────────────────────────────────

class FeatureEngineer:
    """Build the 10-feature spatial stack for DEM correction models.

    Parameters
    ----------
    config : dict
        Full pipeline configuration.
    """

    FEATURE_NAMES = [
        "B2", "B3", "B4", "B8",
        "NDVI", "MNDWI",
        "fabdem_elev",
        "slope", "aspect", "dist2river",
    ]

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config["paths"]["data_processed"]) / "features"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Point-level sampling (for training) ───────────────────────────

    def sample_features_at_photons(
        self,
        photons: gpd.GeoDataFrame,
        sentinel2_path: str | Path,
        fabdem_path: str | Path,
        slope_path: str | Path,
        aspect_path: str | Path,
        dist2river_path: str | Path,
    ) -> gpd.GeoDataFrame:
        """Sample all 10 features at ICESat-2 photon locations.

        Parameters
        ----------
        photons : gpd.GeoDataFrame
            Must have ``geometry`` (Point) and ``elevation`` columns.
        sentinel2_path : str | Path
            Multi-band Sentinel-2 GeoTIFF (B2, B3, B4, B8, B11 as bands 1–5).
        fabdem_path, slope_path, aspect_path, dist2river_path : str | Path
            Single-band rasters.

        Returns
        -------
        gpd.GeoDataFrame
            Input photons with 10 new feature columns added.
        """
        logger.info("Sampling 10 features at %d photon locations...", len(photons))

        import rasterio
        from shapely.geometry import box
        
        # ── 0.5. Bounding Box Filter ──
        bbox = self.config["study_area"]["bbox"]
        bbox_geom = box(bbox[0], bbox[1], bbox[2], bbox[3])
        
        # We need to create a GeoDataFrame for the bbox and reproject it to the photons' crs
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs=self.config["study_area"]["crs_geographic"])
        bbox_gdf_proj = bbox_gdf.to_crs(photons.crs)
        
        initial_len = len(photons)
        photons = photons.clip(bbox_gdf_proj)
        logger.info("  Dropped %d photons outside bbox.", initial_len - len(photons))

        # ── 0. Reproject to match raster CRS ──
        with rasterio.open(sentinel2_path) as src:
            target_crs = src.crs
            transform = src.transform
            
        if photons.crs != target_crs:
            logger.info("Reprojecting photons from %s to %s to match Sentinel-2 raster...", photons.crs, target_crs)
            photons = photons.to_crs(target_crs)
            
        # ── 1. Filter out obvious noise using 3-sigma bounds on residual errors ──
        logger.info("Sampling FabDEM at raw photon locations for statistical 3-sigma filtering...")
        photons = sample_raster_at_points(fabdem_path, photons, column_name="fabdem_elev")
        
        diffs = photons["fabdem_elev"] - photons["elevation"]
        
        # 3-sigma logic
        mean_diff = diffs.mean()
        std_diff = diffs.std()
        lower_bound = mean_diff - 3 * std_diff
        upper_bound = mean_diff + 3 * std_diff
        
        valid_mask = (diffs >= lower_bound) & (diffs <= upper_bound)
        
        initial_count = len(photons)
        photons = photons[valid_mask].copy()
        logger.info(
            "  Dropped %d residual noise photons outside 3-sigma bounds [%.2f, %.2f].",
            initial_count - len(photons), lower_bound, upper_bound
        )
        # ── 2. Spatial Grid-Thinning (10m Resampling) ──
        logger.info("Aggregating filtered photons to 10m grid cells...")
            
        xs = photons.geometry.x.values
        ys = photons.geometry.y.values
        rows, cols = rasterio.transform.rowcol(transform, xs, ys)
        
        photons["row"] = rows
        photons["col"] = cols
        
        photons["proj_x"] = photons.geometry.x
        photons["proj_y"] = photons.geometry.y
        
        # Group by grid cell and track_id, then calculate median
        cols_to_agg = {"elevation": "median", "proj_x": "median", "proj_y": "median", "fabdem_elev": "median"}
        grouped = photons.groupby(["row", "col", "track_id"], as_index=False).agg(cols_to_agg)
        
        # Recreate GeoDataFrame from projected median coordinates
        photons = gpd.GeoDataFrame(
            grouped,
            geometry=gpd.points_from_xy(grouped["proj_x"], grouped["proj_y"]),
            crs=photons.crs
        )
        logger.info("  Thinned down to %d unique 10m grid-aligned samples.", len(photons))

        # ── 2. Feature Extraction ──
        # Sentinel-2 spectral bands (band indices 1-5 in stacked GeoTIFF)
        band_map = {"B2": 1, "B3": 2, "B4": 3, "B8": 4, "B11": 5}
        
        import rasterio
        with rasterio.open(sentinel2_path) as src:
            s2_band_count = src.count
            
        for name, band_idx in band_map.items():
            if band_idx <= s2_band_count:
                photons = sample_raster_at_points(
                    sentinel2_path, photons, band=band_idx, column_name=name,
                )
            else:
                photons[name] = 0.0

        # Derived indices
        photons["NDVI"] = compute_ndvi(
            photons["B8"].values.astype(np.float32),
            photons["B4"].values.astype(np.float32),
        )
        
        # If B11 is missing (all zeros), fallback to NDWI (B3 and B8)
        b11_vals = photons["B11"].values.astype(np.float32)
        if np.all(b11_vals == 0):
            b11_vals = photons["B8"].values.astype(np.float32)
            
        photons["MNDWI"] = compute_mndwi(
            photons["B3"].values.astype(np.float32),
            b11_vals,
        )

        # Terrain rasters (single-band)
        # fabdem_elev is already sampled and aggregated
        photons = sample_raster_at_points(slope_path, photons, column_name="slope")
        photons = sample_raster_at_points(aspect_path, photons, column_name="aspect")
        photons = sample_raster_at_points(dist2river_path, photons, column_name="dist2river")

        # Drop B11 intermediate column (not a model feature)
        photons = photons.drop(columns=["B11"], errors="ignore")

        # ── 4. Final Quality Filters ──
        
        # Water masking (drop if MNDWI > 0)
        initial_len = len(photons)
        water_mask = photons["MNDWI"] <= 0.0
        photons = photons[water_mask]
        logger.info("  Dropped %d water surface points (MNDWI > 0).", initial_len - len(photons))
        
        # Drop NO DATA (points outside Sentinel-2 or FabDEM bounds)
        initial_len = len(photons)
        valid_data_mask = (photons["B2"] > 0) & (photons["fabdem_elev"] > -9000)
        photons = photons[valid_data_mask]
        logger.info("  Dropped %d points outside valid raster bounds (NO DATA).", initial_len - len(photons))

        logger.info("  Feature sampling & filtering complete — %d points, %d columns total", len(photons), len(photons.columns))
        return photons

    # ── Wall-to-wall raster stack (for prediction) ────────────────────

    def build_feature_stack(
        self,
        sentinel2_path: str | Path,
        fabdem_path: str | Path,
        slope_path: str | Path,
        aspect_path: str | Path,
        dist2river_path: str | Path,
        output_path: str | Path | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Build a (10, H, W) feature stack for wall-to-wall DTM prediction.

        All rasters must be co-registered to the same grid (CRS, resolution,
        extent).  FabDEM should already be resampled to 10 m.

        Returns
        -------
        tuple[np.ndarray, dict]
            Feature stack ``(10, H, W)`` and rasterio profile.
        """
        import rasterio
        from rasterio.enums import Resampling

        logger.info("Building wall-to-wall 10-feature stack...")

        # Sentinel-2 (bands: B2=0, B3=1, B4=2, B8=3, B11=4)
        with rasterio.open(sentinel2_path) as src:
            s2_data = src.read().astype(np.float32)
            profile = dict(src.profile)
            height, width = s2_data.shape[1], s2_data.shape[2]

        b2, b3, b4, b8 = s2_data[0], s2_data[1], s2_data[2], s2_data[3]
        b11 = s2_data[4] if s2_data.shape[0] >= 5 else b8.copy()

        ndvi = compute_ndvi(b8, b4)
        mndwi = compute_mndwi(b3, b11)

        # FabDEM — resample to match Sentinel-2 grid
        with rasterio.open(fabdem_path) as src:
            fabdem = src.read(
                1,
                out_shape=(height, width),
                resampling=Resampling.bilinear,
            ).astype(np.float32)

        # Terrain features — read and match grid
        def _read_match(path):
            with rasterio.open(path) as src:
                return src.read(
                    1,
                    out_shape=(height, width),
                    resampling=Resampling.bilinear,
                ).astype(np.float32)

        slope = _read_match(slope_path)
        aspect = _read_match(aspect_path)
        dist2river = _read_match(dist2river_path)

        # Stack: (10, H, W)
        feature_stack = np.stack([
            b2, b3, b4, b8,
            ndvi, mndwi,
            fabdem,
            slope, aspect, dist2river,
        ], axis=0)

        logger.info(
            "  Feature stack shape: %s (features=%d, H=%d, W=%d)",
            feature_stack.shape, feature_stack.shape[0], height, width,
        )

        if output_path:
            output_path = Path(output_path)
            profile.update(count=10, dtype="float32")
            save_raster(output_path, feature_stack, profile, nodata=-9999.0)
            logger.info("  Saved feature stack → %s", output_path)

        return feature_stack, profile

    def get_feature_names(self) -> list[str]:
        """Return ordered list of feature names."""
        return list(self.FEATURE_NAMES)
