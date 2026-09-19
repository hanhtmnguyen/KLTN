"""
Step ② — Sentinel-2 spectral band download via Google Earth Engine.

Downloads a cloud-free composite of B2 (Blue), B3 (Green), B4 (Red),
B8 (NIR) at 10 m resolution for training the ANN DEM corrector (Step ④).
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from src.utils.logging_config import get_logger
from src.utils.geo_utils import load_config

logger = get_logger("pigan.phase1.sentinel2")


class Sentinel2Downloader:
    """Download Sentinel-2 spectral bands via Google Earth Engine.

    Parameters
    ----------
    config : dict
        Full pipeline config.
    output_dir : str | Path | None
        Directory for exported GeoTIFFs.
    """

    def __init__(self, config: dict, output_dir: str | Path | None = None):
        self.bbox = config["study_area"]["bbox"]
        self.bands = config["data_sources"]["sentinel2"]["bands"]
        self.max_cloud = config["data_sources"]["sentinel2"]["max_cloud_cover"]
        self.collection = config["data_sources"]["sentinel2"]["collection"]
        self.ee_project = config.get("ee_project")  # GEE Cloud Project ID
        self.output_dir = Path(
            output_dir or Path(config["paths"]["data_raw"]) / "sentinel2"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _mask_clouds(image):
        """Apply cloud mask using QA60 band."""
        import ee

        qa = image.select("QA60")
        # Bits 10 and 11 are clouds and cirrus
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = (
            qa.bitwiseAnd(cloud_bit_mask).eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        )
        return image.updateMask(mask)

    def download_composite(
        self,
        date_range: tuple[str, str] = ("2023-01-01", "2023-12-31"),
        export_to_drive: bool = True,
        drive_folder: str = "KLTN_Sentinel2",
    ) -> str:
        """Create and export a cloud-free median composite.

        Parameters
        ----------
        date_range : tuple[str, str]
            Start and end dates for image collection filtering.
        export_to_drive : bool
            If True, export to Google Drive (async).
            If False, attempt direct download (small areas only).
        drive_folder : str
            Google Drive folder for export.

        Returns
        -------
        str
            Task ID (if export) or local file path.
        """
        import ee

        init_kwargs = {}
        if self.ee_project:
            init_kwargs["project"] = self.ee_project
        try:
            ee.Initialize(**init_kwargs)
        except Exception:
            ee.Authenticate()
            ee.Initialize(**init_kwargs)

        west, south, east, north = self.bbox
        aoi = ee.Geometry.Rectangle([west, south, east, north])

        logger.info(
            "Filtering Sentinel-2: dates=%s, cloud<%d%%, bands=%s",
            date_range, self.max_cloud, self.bands,
        )

        collection = (
            ee.ImageCollection(self.collection)
            .filterBounds(aoi)
            .filterDate(*date_range)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", self.max_cloud))
            .map(self._mask_clouds)
        )

        composite = collection.median().select(self.bands).clip(aoi)

        logger.info("Created median composite from %d images",
                     collection.size().getInfo())

        if export_to_drive:
            task = ee.batch.Export.image.toDrive(
                image=composite,
                description="S2_composite_Red_River",
                folder=drive_folder,
                region=aoi,
                scale=10,
                crs="EPSG:32648",
                maxPixels=1e13,
                fileFormat="GeoTIFF",
            )
            task.start()
            logger.info(
                "Export task started → Google Drive/%s/S2_composite_Red_River.tif",
                drive_folder,
            )
            logger.info(
                "Monitor at: https://code.earthengine.google.com/tasks"
            )
            return task.id

        else:
            # Direct download for small areas (< ~100 MB)
            import requests

            url = composite.getDownloadURL({
                "scale": 10,
                "crs": "EPSG:32648",
                "region": aoi,
                "format": "GEO_TIFF",
            })
            out_path = self.output_dir / "s2_composite.tif"
            logger.info("Downloading composite directly...")
            response = requests.get(url, stream=True)
            with open(out_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Saved to %s", out_path)
            return str(out_path)

    def download_sar_flood(
        self,
        flood_date: str,
        pre_days: int = 30,
    ) -> str:
        """Download Sentinel-1 SAR pre/post flood images for data assimilation.

        This is a convenience method; the main SAR processing is in
        ``phase2_groundtruth/sar_flood_extractor.py``.

        Parameters
        ----------
        flood_date : str
            Date of the flood event (YYYY-MM-DD).
        pre_days : int
            Number of days before the flood for the "dry" reference.

        Returns
        -------
        str
            Google Drive task ID.
        """
        import ee
        from datetime import datetime, timedelta

        try:
            ee.Initialize()
        except Exception:
            ee.Authenticate()
            ee.Initialize()

        west, south, east, north = self.bbox
        aoi = ee.Geometry.Rectangle([west, south, east, north])

        flood_dt = datetime.strptime(flood_date, "%Y-%m-%d")
        pre_start = (flood_dt - timedelta(days=pre_days)).strftime("%Y-%m-%d")
        pre_end = (flood_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        post_start = flood_date
        post_end = (flood_dt + timedelta(days=5)).strftime("%Y-%m-%d")

        s1 = ee.ImageCollection("COPERNICUS/S1_GRD").filter(
            ee.Filter.listContains("transmitterReceiverPolarisation", "VV")
        ).filter(
            ee.Filter.eq("instrumentMode", "IW")
        ).filterBounds(aoi)

        pre_image = s1.filterDate(pre_start, pre_end).median().select("VV")
        post_image = s1.filterDate(post_start, post_end).median().select("VV")

        # Difference map (post - pre); negative values = flood
        diff = post_image.subtract(pre_image).rename("flood_diff")

        task = ee.batch.Export.image.toDrive(
            image=diff.clip(aoi),
            description=f"SAR_flood_diff_{flood_date}",
            folder="KLTN_SAR",
            region=aoi,
            scale=10,
            crs="EPSG:32648",
            maxPixels=1e13,
        )
        task.start()
        logger.info("SAR flood diff export started for %s", flood_date)
        return task.id
