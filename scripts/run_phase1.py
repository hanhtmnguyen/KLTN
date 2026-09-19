"""
CLI entry point for Phase 1: DTM Generation & Pre-processing.

Usage:
    python scripts/run_phase1.py --config configs/base_config.yaml
"""

import sys
import click
from pathlib import Path

# Ensure project root is in sys.path so 'from src...' imports work
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.logging_config import setup_logger
from src.utils.geo_utils import load_config


@click.command()
@click.option(
    "--config", default="configs/base_config.yaml",
    help="Path to base configuration YAML.",
)
@click.option("--skip-download", is_flag=True, help="Skip data download steps.")
@click.option("--extract-only", is_flag=True, help="Skip downloading ICESat-2 and extract photons from existing files.")
@click.option("--skip-ann", is_flag=True, help="Skip ANN training (use existing).")
@click.option(
    "--step", "-s", type=click.IntRange(1, 7), default=None,
    help="Run only a specific step number (1-7): 1=ICESat-2, 2=Sentinel-2, 3=FabDEM, 4=ANN, 5=OSM, 6=Roughness, 7=Terrain.",
)
@click.option(
    "--only", type=click.Choice(["icesat2", "sentinel2", "fabdem", "ann", "osm", "roughness", "terrain"], case_sensitive=False), default=None,
    help="Run only a specific step by name.",
)
def main(config: str, skip_download: bool, extract_only: bool, skip_ann: bool, step: int | None, only: str | None):
    """Run Phase 1: Download data, train ANN, generate 10m DTM & features."""
    logger = setup_logger("pigan.phase1")
    cfg = load_config(config)

    def should_run(step_num: int, step_name: str) -> bool:
        if step is not None:
            return step_num == step
        if only is not None:
            return step_name.lower() == only.lower()
        return True

    # ---- Step ①: ICESat-2 ----
    if should_run(1, "icesat2") and not skip_download:
        from src.phase1_dtm.icesat2_downloader import ICESat2Downloader
        logger.info("=" * 60)
        logger.info("Step ①: ICESat-2 ATL03 Download")
        logger.info("=" * 60)
        icesat = ICESat2Downloader(cfg)
        photons = icesat.process_all(extract_only=extract_only)
        logger.info("Ground photons: %d", len(photons))

    # ---- Step ②: Sentinel-2 ----
    if should_run(2, "sentinel2") and not skip_download:
        from src.phase1_dtm.sentinel2_downloader import Sentinel2Downloader
        logger.info("=" * 60)
        logger.info("Step ②: Sentinel-2 Composite Download")
        logger.info("=" * 60)
        s2 = Sentinel2Downloader(cfg)
        s2.download_composite()

    # ---- Step ③: FabDEM ----
    if should_run(3, "fabdem") and not skip_download:
        from src.phase1_dtm.fabdem_downloader import FabDEMDownloader
        logger.info("=" * 60)
        logger.info("Step ③: FabDEM 30m Download")
        logger.info("=" * 60)
        fabdem = FabDEMDownloader(cfg)
        fabdem.download_and_reproject()

    # ---- Step ④: ANN DEM Correction ----
    if should_run(4, "ann") and not skip_ann:
        from src.phase1_dtm.ann_dtm_corrector import ANNDTMCorrector
        logger.info("=" * 60)
        logger.info("Step ④: ANN DEM Correction Training")
        logger.info("=" * 60)

        ann = ANNDTMCorrector(cfg)

        raw_dir = Path(cfg["paths"]["data_raw"])
        X, y = ann.prepare_training_data(
            photons_path=raw_dir / "icesat2" / "ground_photons.gpkg",
            sentinel2_path=raw_dir / "sentinel2" / "s2_composite.tif",
            fabdem_path=raw_dir / "fabdem" / "fabdem_utm.tif",
        )
        ann.train(X, y)
        ann.predict_dtm(
            sentinel2_path=raw_dir / "sentinel2" / "s2_composite.tif",
            fabdem_path=raw_dir / "fabdem" / "fabdem_utm.tif",
            output_path=Path(cfg["paths"]["data_processed"]) / "dtm_10m" / "dtm_10m.tif",
        )

    # ---- Step ⑤: OSM Buildings ----
    if should_run(5, "osm"):
        from src.phase1_dtm.osm_buildings import OSMBuildingExtractor
        logger.info("=" * 60)
        logger.info("Step ⑤: OSM Building Extraction")
        logger.info("=" * 60)
        osm = OSMBuildingExtractor(cfg)
        osm.process()

    # ---- Step ⑥: Manning's Roughness ----
    if should_run(6, "roughness"):
        from src.phase1_dtm.roughness_matrix import RoughnessMatrixGenerator
        logger.info("=" * 60)
        logger.info("Step ⑥: Manning's Roughness Matrix")
        logger.info("=" * 60)
        roughness = RoughnessMatrixGenerator(cfg)
        roughness.generate(
            buildings_path=Path(cfg["paths"]["data_raw"]) / "osm" / "buildings.gpkg",
            dtm_path=Path(cfg["paths"]["data_processed"]) / "dtm_10m" / "dtm_10m.tif",
            sentinel2_path=Path(cfg["paths"]["data_raw"]) / "sentinel2" / "s2_composite.tif",
        )

    # ---- Step ⑦: Terrain Features ----
    if should_run(7, "terrain"):
        from src.phase1_dtm.terrain_features import TerrainFeatureGenerator
        logger.info("=" * 60)
        logger.info("Step ⑦: Terrain Feature Engineering")
        logger.info("=" * 60)
        terrain = TerrainFeatureGenerator(cfg)
        terrain.generate_all(
            dtm_path=Path(cfg["paths"]["data_processed"]) / "dtm_10m" / "dtm_10m.tif",
        )

    logger.info("=" * 60)
    logger.info("Phase 1 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
