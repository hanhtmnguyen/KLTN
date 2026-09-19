"""
Geospatial utility functions shared across all pipeline phases.

Handles raster I/O, CRS transforms, reprojection, resampling,
point sampling, and vector rasterization.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize as _rio_rasterize
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject
from rasterio.mask import mask as rio_mask
import geopandas as gpd
from pathlib import Path
from typing import Sequence
import yaml

from src.utils.logging_config import get_logger

logger = get_logger("pigan.geo")


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path = "configs/base_config.yaml") -> dict:
    """Load a YAML configuration file and return as dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Raster I/O
# ---------------------------------------------------------------------------

def load_raster(
    path: str | Path,
    band: int = 1,
) -> tuple[np.ndarray, rasterio.profiles.Profile]:
    """Read a single-band raster and return (array, profile).

    Parameters
    ----------
    path : str | Path
        Path to a GeoTIFF file.
    band : int
        Band index (1-based).

    Returns
    -------
    tuple[np.ndarray, dict]
        The 2-D array and the rasterio profile (metadata).
    """
    path = Path(path)
    with rasterio.open(path) as src:
        data = src.read(band).astype(np.float32)
        profile = dict(src.profile)
    return data, profile


def load_raster_multiband(
    path: str | Path,
) -> tuple[np.ndarray, rasterio.profiles.Profile]:
    """Read all bands of a raster → (C, H, W) array + profile."""
    path = Path(path)
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        profile = dict(src.profile)
    return data, profile


def save_raster(
    path: str | Path,
    data: np.ndarray,
    profile: dict,
    nodata: float | None = -9999.0,
) -> Path:
    """Write a 2-D or 3-D array to a GeoTIFF.

    Parameters
    ----------
    path : str | Path
        Destination file path.
    data : np.ndarray
        Shape ``(H, W)`` for single band or ``(C, H, W)`` for multi-band.
    profile : dict
        Rasterio profile (must contain ``transform``, ``crs``, etc.).
    nodata : float | None
        NoData value to embed in the file.

    Returns
    -------
    Path
        The written file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        data = data[np.newaxis, ...]  # (1, H, W)

    count, height, width = data.shape
    write_profile = {
        **profile,
        "driver": "GTiff",
        "dtype": str(data.dtype),
        "count": count,
        "height": height,
        "width": width,
        "compress": "deflate",
    }
    if nodata is not None:
        write_profile["nodata"] = nodata

    with rasterio.open(path, "w", **write_profile) as dst:
        dst.write(data)

    logger.debug("Saved raster %s  (%d bands, %dx%d)", path.name, count, height, width)
    return path


# ---------------------------------------------------------------------------
# Reprojection & resampling
# ---------------------------------------------------------------------------

def reproject_raster(
    src_path: str | Path,
    dst_path: str | Path,
    dst_crs: str = "EPSG:32648",
    dst_resolution: float | None = None,
    resampling: Resampling = Resampling.bilinear,
) -> Path:
    """Reproject a raster to a new CRS (and optionally resolution).

    Parameters
    ----------
    src_path, dst_path : str | Path
        Input / output file paths.
    dst_crs : str
        Target coordinate reference system.
    dst_resolution : float | None
        Target pixel size in CRS units (metres for UTM).
        ``None`` keeps the native resolution.
    resampling : rasterio.enums.Resampling
        Resampling algorithm.

    Returns
    -------
    Path
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=dst_resolution,
        )
        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            compress="deflate",
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                )

    logger.info("Reprojected %s → %s  (CRS=%s, res=%s)",
                src_path.name, dst_path.name, dst_crs, dst_resolution)
    return dst_path


def resample_raster(
    src_path: str | Path,
    dst_path: str | Path,
    target_resolution: float,
    resampling: Resampling = Resampling.bilinear,
) -> Path:
    """Resample a raster to a new resolution *in its existing CRS*.

    Parameters
    ----------
    src_path, dst_path : str | Path
    target_resolution : float
        New pixel size in CRS units.
    resampling : Resampling

    Returns
    -------
    Path
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        scale_x = src.res[0] / target_resolution
        scale_y = src.res[1] / target_resolution
        new_width = int(src.width * scale_x)
        new_height = int(src.height * scale_y)
        new_transform = src.transform * src.transform.scale(
            src.width / new_width,
            src.height / new_height,
        )

        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=resampling,
        )

        profile = src.profile.copy()
        profile.update(
            transform=new_transform,
            width=new_width,
            height=new_height,
            compress="deflate",
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data)

    logger.info("Resampled %s → %s  (res=%.1fm)", src_path.name, dst_path.name,
                target_resolution)
    return dst_path


# ---------------------------------------------------------------------------
# Point sampling
# ---------------------------------------------------------------------------

def sample_raster_at_points(
    raster_path: str | Path,
    points: gpd.GeoDataFrame,
    band: int = 1,
    column_name: str = "sampled_value",
) -> gpd.GeoDataFrame:
    """Sample a raster at point locations and attach values to the GeoDataFrame.

    Parameters
    ----------
    raster_path : str | Path
        Path to the raster.
    points : gpd.GeoDataFrame
        Must have a ``geometry`` column with Point geometries.
    band : int
        Band to sample.
    column_name : str
        Name for the new column holding sampled values.

    Returns
    -------
    gpd.GeoDataFrame
        Copy of *points* with the new column added.
    """
    points = points.copy()
    coords = [(pt.x, pt.y) for pt in points.geometry]

    with rasterio.open(raster_path) as src:
        values = [v[0] for v in src.sample(coords, indexes=band)]

    points[column_name] = values
    return points


# ---------------------------------------------------------------------------
# Vector rasterization
# ---------------------------------------------------------------------------

def rasterize_vector(
    gdf: gpd.GeoDataFrame,
    value_column: str,
    reference_raster: str | Path,
    output_path: str | Path,
    fill_value: float = 0.0,
    dtype: str = "float32",
) -> Path:
    """Rasterize a vector GeoDataFrame to match a reference raster grid.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Vector data with a ``geometry`` column and a column to burn.
    value_column : str
        Column whose values will be burned into the raster.
    reference_raster : str | Path
        A raster whose grid (transform, shape, CRS) to match.
    output_path : str | Path
        Destination GeoTIFF path.
    fill_value : float
        Value for cells without any geometry.
    dtype : str
        Output data type.

    Returns
    -------
    Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(reference_raster) as ref:
        transform = ref.transform
        out_shape = (ref.height, ref.width)
        crs = ref.crs
        profile = ref.profile.copy()

    # Ensure same CRS
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    shapes = ((geom, val) for geom, val in zip(gdf.geometry, gdf[value_column]))

    raster = _rio_rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=fill_value,
        dtype=dtype,
    )

    profile.update(count=1, dtype=dtype, compress="deflate", nodata=fill_value)
    save_raster(output_path, raster, profile, nodata=fill_value)
    return output_path


# ---------------------------------------------------------------------------
# Bbox / transform helpers
# ---------------------------------------------------------------------------

def bbox_to_transform(
    bbox: Sequence[float],
    resolution: float,
) -> tuple[rasterio.Affine, int, int]:
    """Convert a (west, south, east, north) bbox to a rasterio Affine transform.

    Returns
    -------
    tuple[Affine, int, int]
        (transform, width, height)
    """
    west, south, east, north = bbox
    width = int(np.ceil((east - west) / resolution))
    height = int(np.ceil((north - south) / resolution))
    transform = from_bounds(west, south, east, north, width, height)
    return transform, width, height


def clip_raster_to_bbox(
    raster_path: str | Path,
    bbox: Sequence[float],
    output_path: str | Path,
) -> Path:
    """Clip a raster to a bounding box.

    Parameters
    ----------
    raster_path : str | Path
    bbox : sequence of float
        (west, south, east, north).
    output_path : str | Path

    Returns
    -------
    Path
    """
    from shapely.geometry import box

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    geom = box(*bbox)
    with rasterio.open(raster_path) as src:
        out_image, out_transform = rio_mask(
            src, [geom], crop=True, nodata=src.nodata
        )
        profile = src.profile.copy()
        profile.update(
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform,
            compress="deflate",
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(out_image)

    logger.info("Clipped %s → %s", Path(raster_path).name, output_path.name)
    return output_path
