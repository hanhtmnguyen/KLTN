"""
Preprocessing modules for the GeoAI DEM Correction pipeline.

Includes:
- Feature engineering (10-feature spatial stack)
- ANN DTM corrector (Residual MLP with Huber Loss — Model B)
- Tree-based DTM corrector (XGBoost/LightGBM — Model C)
- Terrain feature generation (Slope, Aspect, TWI, HAND, Dist2River)
- HAND flood model (dual-DEM comparison)
- Roughness matrix and OSM building processing
"""
