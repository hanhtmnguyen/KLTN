import argparse
import sys
import pickle
from pathlib import Path
import numpy as np
import rasterio

sys.path.append(str(Path("d:/KLTN/da-framework").resolve()))
from src.utils.geo_utils import load_config
from src.utils.logging_config import setup_logger, get_logger
from src.preprocessing.feature_engineering import FeatureEngineer
from src.preprocessing.tree_dtm_corrector import TreeDTMCorrector

def generate_full_dem():
    setup_logger("da", level=20)
    logger = get_logger("da.predict")
    config = load_config()
    
    logger.info("Initializing FeatureEngineer to build feature stack...")
    fe = FeatureEngineer(config)
    
    s2_path = Path("data/raw/sentinel2/S2_composite_Red_River.tif")
    fabdem_path = Path("data/raw/fabdem/fabdem_32648_10m.tif")
    slope_path = Path("data/processed/terrain_features/slope.tif")
    aspect_path = Path("data/processed/terrain_features/aspect.tif")
    dist2river_path = Path("data/processed/terrain_features/distance_to_river.tif")
    
    feature_stack, profile = fe.build_feature_stack(
        sentinel2_path=s2_path,
        fabdem_path=fabdem_path,
        slope_path=slope_path,
        aspect_path=aspect_path,
        dist2river_path=dist2river_path
    )
    
    logger.info(f"Feature stack built with shape {feature_stack.shape}")
    
    logger.info("Loading XGBoost Model...")
    tree = TreeDTMCorrector(config, backend="xgboost")
    model_path = Path("models/tree_dtm/xgboost_dtm_best.pkl")
    
    with open(model_path, "rb") as f:
        tree.model = pickle.load(f)
        
    logger.info("Predicting DTM...")
    output_path = Path("data/processed/dtm/xgboost_corrected_dtm.tif")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read fabdem again since predict_dtm needs it to add residuals
    with rasterio.open(fabdem_path) as src:
        # In case dimensions differ, read with the target profile shape
        fabdem = src.read(1, out_shape=(profile['height'], profile['width']), resampling=rasterio.enums.Resampling.bilinear)
    
    tree.predict_dtm(feature_stack, fabdem, profile, output_path)
    logger.info(f"Successfully generated full corrected DTM at {output_path}")

if __name__ == "__main__":
    generate_full_dem()
