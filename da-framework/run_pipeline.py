import sys
from pathlib import Path
import argparse

from src.utils.geo_utils import load_config
from src.utils.logging_config import setup_logger, get_logger
from src.data_acquisition.icesat2_atl03 import ICESat2Downloader

def main():
    parser = argparse.ArgumentParser(description="GeoAI DEM Correction Pipeline")
    parser.add_argument(
        "step", 
        choices=["download-icesat2", "download-sentinel2", "download-fabdem", "extract-photons", "build-features", "train-models"], 
        help="The pipeline step to run."
    )
    parser.add_argument(
        "--products", 
        nargs="+", 
        help="Override which ICESat-2 products to download (e.g., --products ATL08)"
    )
    args = parser.parse_args()

    setup_logger("da", level=20)
    logger = get_logger("da.main")
    
    config = load_config()

    if args.step == "download-icesat2":
        if args.products:
            config["data_sources"]["icesat2"]["products"] = args.products
            
        logger.info(f"Starting ICESat-2 download & photon extraction for {config['data_sources']['icesat2']['products']}...")
        downloader = ICESat2Downloader(config)
        photons_gdf = downloader.process_all(extract_only=False)
        logger.info(f"Process complete! Saved to {downloader.output_dir / 'ground_photons.gpkg'}")
        
    elif args.step == "download-sentinel2":
        from src.data_acquisition.sentinel2_downloader import Sentinel2Downloader
        logger.info("Starting Sentinel-2 GEE download...")
        s2_downloader = Sentinel2Downloader(config)
        s2_path = s2_downloader.download_composite()
        logger.info(f"Sentinel-2 download complete! Saved to {s2_path}")

    elif args.step == "download-fabdem":
        from src.data_acquisition.fabdem_downloader import FabDEMDownloader
        logger.info("Starting FabDEM download...")
        fabdem_downloader = FabDEMDownloader(config)
        fabdem_path = fabdem_downloader.download()
        logger.info(f"FabDEM download complete! Saved to {fabdem_path}")
        
    elif args.step == "extract-photons":
        logger.info("Starting ICESat-2 photon extraction (skipping download)...")
        downloader = ICESat2Downloader(config)
        photons_gdf = downloader.process_all(extract_only=True)
        logger.info(f"Extraction complete! Saved to {downloader.output_dir / 'ground_photons.gpkg'}")

    elif args.step == "build-features":
        from src.preprocessing.feature_engineering import FeatureEngineer
        from src.preprocessing.terrain_features import TerrainFeatureGenerator
        import geopandas as gpd
        
        logger.info("Starting Feature Engineering...")
        
        # Paths
        photons_path = Path(config["paths"]["data_raw"]) / "icesat2_atl03" / "ground_photons_32648.gpkg"
        fabdem_path = Path(config["paths"]["data_raw"]) / "fabdem" / "fabdem_32648_10m.tif"
        s2_path = Path(config["paths"]["data_raw"]) / "sentinel2" / "S2_composite_Red_River.tif"
        river_shp_path = Path(r"d:\KLTN\bound\Mangluoi_Thuyvan_diss2.shp")
        
        if not fabdem_path.exists():
            raise FileNotFoundError(f"FabDEM missing at {fabdem_path}. Run download-fabdem first.")
        if not s2_path.exists():
            raise FileNotFoundError(f"Sentinel-2 missing at {s2_path}. Ensure it is downloaded from Drive.")
            
        logger.info("Step 1: Generating terrain features from FabDEM using WhiteboxTools...")
        terrain_gen = TerrainFeatureGenerator(config)
        # Force regeneration to pick up new river shapefile
        dtm_cache = Path(config["paths"]["data_processed"]) / "terrain_features" / "dtm.tif"
        if dtm_cache.exists():
            dtm_cache.unlink()
        terrain_gen.generate_all(dtm_path=fabdem_path, river_shp_path=river_shp_path)
        terrain_paths = terrain_gen.get_feature_paths()
        
        logger.info(f"Step 2: Loading 100% of ground photons from {photons_path}")
        # Load all data; bbox filtering will be done later
        sql = "SELECT * FROM ground_photons_32648"
        photons = gpd.read_file(photons_path, engine="pyogrio", sql=sql)
        
        # Apply geoid correction: h_ortho = h_ph - geoid_undulation
        # ICESat-2 h_ph is WGS84 ellipsoidal; FabDEM is EGM2008 orthometric.
        # Measured geoid undulation for the Red River study area: ~-28.1 m (very uniform over 30 km).
        # The raw H5 geoid_h column was not preserved in the GPKG, so we apply a constant correction.
        GEOID_UNDULATION = -28.1  # metres (EGM2008 at ~105.85E, 21.0N)
        logger.info("  Applying EGM2008 geoid correction: elevation += %.1f m ...", -GEOID_UNDULATION)
        photons["elevation"] = photons["elevation"] - GEOID_UNDULATION
        
        logger.info("Step 3: Sampling all features at photon locations...")
        fe = FeatureEngineer(config)
        training_data = fe.sample_features_at_photons(
            photons=photons,
            sentinel2_path=s2_path,
            fabdem_path=fabdem_path,
            slope_path=terrain_paths["slope"],
            aspect_path=terrain_paths["aspect"],
            dist2river_path=terrain_paths["distance_to_river"]
        )
        
        out_path = Path(config["paths"]["training"]) / "training_data_v2.gpkg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        training_data.to_file(out_path, driver="GPKG", engine="pyogrio")
        logger.info(f"Feature Engineering complete! Training data saved to {out_path}")
        
    elif args.step == "train-models":
        from src.preprocessing.ann_dtm_corrector import ANNDTMCorrector
        from src.preprocessing.tree_dtm_corrector import TreeDTMCorrector
        import geopandas as gpd
        
        logger.info("Starting Model Training Phase (GroupKFold Spatial CV)...")
        training_path = Path(config["paths"]["training"]) / "training_data_v2.gpkg"
        if not training_path.exists():
            raise FileNotFoundError(f"Training data missing at {training_path}. Run build-features first.")
            
        logger.info(f"Loading strictly filtered training dataset from {training_path}")
        training_data = gpd.read_file(training_path, engine="pyogrio")
        
        # Train Artificial Neural Network (Residual MLP)
        logger.info("Initializing ANNDTMCorrector (Residual MLP with Huber Loss)...")
        ann = ANNDTMCorrector(config, model_variant="residual")
        X, y, groups = ann.prepare_training_data(training_data)
        if groups is None:
            raise ValueError("track_id column missing. Cannot perform GroupKFold.")
            
        # logger.info("Running Spatial K-Fold (GroupKFold) for ANN...")
        # ann_results = ann.train_spatial_kfold(X, y, groups=groups, n_splits=5)
        
        logger.info("Training final ANN on full dataset...")
        ann.train(X, y, epochs=100)
        
        # Train Tree-based Model (XGBoost)
        logger.info("Initializing TreeDTMCorrector (XGBoost)...")
        tree = TreeDTMCorrector(config, backend="xgboost")
        X_t, y_t, groups_t = tree.prepare_training_data(training_data)
        
        logger.info("Running Spatial K-Fold (GroupKFold) for XGBoost...")
        tree_results = tree.train_spatial_kfold(X_t, y_t, groups=groups_t, n_splits=5)
        
        logger.info("Training final XGBoost on full dataset...")
        tree.train(X_t, y_t)
        
        logger.info("Model Training Phase Complete!")
        
if __name__ == "__main__":
    main()
