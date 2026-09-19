import argparse
import sys
import geopandas as gpd
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle

sys.path.append(str(Path("d:/KLTN/da-framework").resolve()))
from src.utils.geo_utils import load_config
from src.utils.logging_config import setup_logger, get_logger
from src.preprocessing.tree_dtm_corrector import TreeDTMCorrector

def get_feature_importance():
    setup_logger("da", level=20)
    config = load_config()
    
    # Load Tree Corrector
    tree = TreeDTMCorrector(config, backend="xgboost")
    model_path = Path("models/tree_dtm/xgboost_dtm_best.pkl")
    
    with open(model_path, "rb") as f:
        tree.model = pickle.load(f)
        
    booster = tree.model.get_booster()
    importance = booster.get_score(importance_type='gain')
    
    features = config["ann_dtm"]["input_features"]
    # map f0, f1... to real names
    feature_names = []
    for f in features:
        if f.startswith("sentinel2_"):
            feature_names.append(f.replace("sentinel2_", ""))
        elif f.startswith("fabdem_"):
            feature_names.append(f.replace("fabdem_", ""))
        else:
            feature_names.append(f)
            
    mapped_importance = {}
    for key, val in importance.items():
        # key is f0, f1, etc.
        idx = int(key.replace('f', ''))
        mapped_importance[feature_names[idx]] = val
        
    sorted_imp = dict(sorted(mapped_importance.items(), key=lambda item: item[1]))
    
    plt.figure(figsize=(10, 6))
    plt.barh(list(sorted_imp.keys()), list(sorted_imp.values()), color='skyblue')
    plt.xlabel('F-Score (Gain)')
    plt.title('XGBoost Feature Importance (Gain)')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    print("Saved feature_importance.png")

if __name__ == "__main__":
    get_feature_importance()
