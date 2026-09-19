import sys
from pathlib import Path
import geopandas as gpd
import numpy as np

sys.path.append(str(Path("d:/KLTN/da-framework").resolve()))
from src.utils.geo_utils import load_config
from src.utils.logging_config import setup_logger, get_logger
from src.preprocessing.tree_dtm_corrector import TreeDTMCorrector

def generate_shap():
    setup_logger("da", level=20)
    logger = get_logger("da.shap")
    config = load_config()
    
    logger.info("Loading training data...")
    data_path = Path(config["paths"]["training"]) / "training_data_v2.gpkg"
    photons = gpd.read_file(data_path)
    
    logger.info("Initializing TreeDTMCorrector to load model...")
    tree = TreeDTMCorrector(config, backend="xgboost")
    
    # Load model
    model_path = Path("models/tree_dtm/xgboost_dtm_best.pkl")
    import pickle
    with open(model_path, "rb") as f:
        tree.model = pickle.load(f)
        
    X, _, _ = tree.prepare_training_data(photons)
    
    # Scale X
    X = tree.scaler.fit_transform(X)
    
    logger.info("Generating SHAP summary...")
    tree.generate_shap_summary(X)

if __name__ == "__main__":
    generate_shap()
