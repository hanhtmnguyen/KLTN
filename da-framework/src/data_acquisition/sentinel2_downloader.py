"""
Step ② — Sentinel-2 spectral band download via Google Earth Engine.

Downloads a cloud-free composite of B2 (Blue), B3 (Green), B4 (Red),
B8 (NIR) at 10 m and B11 (SWIR1) at 20 m — resampled to 10 m — for
training the ANN DEM corrector (Step ④) and computing MNDWI.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from src.utils.logging_config import get_logger
from src.utils.geo_utils import load_config

logger = get_logger("da.data.sentinel2")


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
        self.output_dir = Path(
            output_dir or Path(config["paths"]["data_raw"]) / "sentinel2"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _mask_clouds(image):
        """Apply cloud mask using QA60 band."""
        import ee

        qa = image.select("QA60")
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

        try:
            ee.Initialize()
        except Exception:
            ee.Authenticate()
            ee.Initialize()

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

        # B11 is 20 m native; export at 10 m forces bilinear resampling
        # which is handled automatically by GEE when scale=10.
        logger.info("Created median composite from %d images (bands: %s)",
                     collection.size().getInfo(), self.bands)

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
            return task.id

        else:
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
