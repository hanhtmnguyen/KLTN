# GeoAI DEM Correction & Flood Impact Assessment Framework
"""
Improves vertical accuracy of Digital Elevation Models (DEM) using
ICESat-2 LiDAR altimetry and evaluates cascading impact on HAND-based
flood inundation modelling for the Red River floodplain, Hanoi.

Pipeline:
  1. Data Acquisition  — ICESat-2 ATL03, FabDEM 30m, Sentinel-2
  2. Feature Engineering — 10-feature spatial stack (spectral + terrain)
  3. DEM Correction    — Residual MLP & XGBoost/LightGBM correctors
  4. Flood Simulation  — HAND on FabDEM 30m vs ANN-DTM 10m
  5. Impact Assessment — Cascading metrics (CSI, F1, FAR, depth MAE)
  6. Risk Mapping      — Red River Landscape Corridor overlay
"""

__version__ = "0.2.0"
