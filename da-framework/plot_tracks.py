import sys
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import folium

from src.utils.geo_utils import load_config
from src.preprocessing.ann_dtm_corrector import ANNDTMCorrector
from src.preprocessing.tree_dtm_corrector import TreeDTMCorrector

def main():
    print("Loading configuration...")
    config = load_config()
    
    training_path = Path(config["paths"]["training"]) / "training_data_v2.gpkg"
    if not training_path.exists():
        print(f"Error: Training data missing at {training_path}")
        sys.exit(1)
        
    print(f"Loading training data from {training_path}...")
    gdf = gpd.read_file(training_path, engine="pyogrio")
    
    output_dir = Path("outputs") / "track_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Initializing models...")
    # Load ANN
    ann = ANNDTMCorrector(config, model_variant="residual")
    ann._load_checkpoint("best")
    
    # Load Tree
    tree = TreeDTMCorrector(config, backend="xgboost")
    tree._load_model()
    
    print("Preparing features...")
    # Prepare X
    feature_cols = ann.feature_cols
    X = gdf[feature_cols].values.astype(np.float32)
    
    # Standardize
    try:
        X_ann = ann.scaler.transform(X)
    except:
        print("Warning: ANN scaler not fit/loaded properly.")
        X_ann = X
        
    try:
        X_tree = tree.scaler.transform(X)
    except:
        print("Warning: Tree scaler not fit/loaded properly.")
        X_tree = X

    print("Generating predictions...")
    # ANN Predictions
    import torch
    ann.model.eval()
    ann_preds = []
    with torch.no_grad():
        test_t = torch.from_numpy(X_ann)
        batch_size = 65536
        for i in range(0, len(test_t), batch_size):
            batch = test_t[i:i + batch_size].to(ann.device)
            ann_preds.append(ann.model(batch).cpu().numpy())
    ann_delta = np.concatenate(ann_preds)
    
    # Tree Predictions
    tree_delta = tree.model.predict(X_tree)
    
    # Calculate corrected elevations
    # If residual learning is True, predicted delta_h is FabDEM - ICESat2
    # So Corrected H = FabDEM - delta_h
    if ann.residual_learning:
        gdf["ann_corrected"] = gdf["fabdem_elev"] - ann_delta
    else:
        gdf["ann_corrected"] = ann_delta
        
    if tree.residual_learning:
        gdf["xgb_corrected"] = gdf["fabdem_elev"] - tree_delta
    else:
        gdf["xgb_corrected"] = tree_delta

    # Group by track and plot
    tracks = gdf["track_id"].unique()
    print(f"Found {len(tracks)} unique tracks. Generating plots...")
    
    sns.set_theme(style="whitegrid")
    
    for track in tracks:
        track_data = gdf[gdf["track_id"] == track].copy()
        
        # Sort by latitude for a continuous profile
        track_data = track_data.sort_values(by="lat")
        
        plt.figure(figsize=(15, 6))
        
        # Plot FabDEM (Baseline)
        plt.plot(track_data["lat"], track_data["fabdem_elev"], label="FabDEM (Uncorrected)", 
                 color="blue", alpha=0.6, linewidth=1.5, linestyle="--")
        
        # Plot ANN Corrected
        plt.plot(track_data["lat"], track_data["ann_corrected"], label="ANN Corrected", 
                 color="orange", alpha=0.8, linewidth=1.5)
                 
        # Plot XGB Corrected
        plt.plot(track_data["lat"], track_data["xgb_corrected"], label="XGBoost Corrected", 
                 color="green", alpha=0.8, linewidth=1.5)
                 
        # Plot ICESat-2 (Ground Truth)
        plt.scatter(track_data["lat"], track_data["elevation"], label="ICESat-2 (Ground Truth)", 
                    color="red", s=5, zorder=5)
        
        plt.title(f"Elevation Profile - Track {track}")
        plt.xlabel("Latitude")
        plt.ylabel("Elevation (m)")
        plt.legend()
        plt.tight_layout()
        
        out_file = output_dir / f"track_{track}_profile.png"
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f"Saved plot: {out_file}")
        
        # ── Generate Interactive HTML Map ──
        # Generate an interactive HTML map instead of static image to bypass local Python network blocks
        # Downsample to 5,000 points max to prevent generating massive HTML files and exhausting disk space
        track_sample = track_data.sample(n=min(len(track_data), 5000), random_state=42)
        
        m = track_sample.explore(
            column="elevation",
            cmap="plasma",
            tiles="OpenStreetMap", # Default normal map
            marker_kwds={"radius": 4, "fill": True},
            name="ICESat-2 Photons (Sampled)"
        )
        
        # Add a Satellite imagery layer option
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Add layer control to let user toggle between Normal Map and Satellite
        folium.LayerControl().add_to(m)
        
        map_file = output_dir / f"track_{track}_map.html"
        m.save(str(map_file))
        print(f"Saved interactive map: {map_file}")

    print("All plots generated successfully!")

if __name__ == "__main__":
    main()
