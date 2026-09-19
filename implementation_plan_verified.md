# Pipeline Restructure: GeoAI DEM Correction + HAND Flood Impact Assessment

Restructure the `da-framework` from the old PI-GAN-centric pipeline to the new 6-phase methodology: **ICESat-2 DEM correction via Residual Learning → HAND-based flood simulation → Cascading impact evaluation for Red River planning corridor**.

---

## User Review Required

> [!IMPORTANT]
> **Major architectural shift**: The PI-GAN model and its associated modules (`src/model/`) will be **deprecated/removed**. The new pipeline has two pillars: (1) ANN-based DEM correction with residual learning and (2) HAND flood comparison — no GAN or physics-informed loss is needed.

> [!WARNING]  
> **Reference repo (`manyoarcane/HAND`)**: I was able to read the GitHub repo directly. However, it only contains a single file [`HAND workflow.py`](https://github.com/manyoarcane/HAND/blob/main/HAND%20workflow.py) — the raw content was **truncated** at ~140 lines by the HTTP fetcher (the file appears to use GDAL + WhiteboxTools for DEM → depression filling → flow direction → stream extraction → HAND computation → flood inundation thresholding). I captured the utility functions (`find_dem_files`, `read_raster_info`, `calculate_representative_pixel_area`) but the main `process_dem()` function body was cut off. **I recommend you clone it locally** (`git clone https://github.com/manyoarcane/HAND.git`) so I can read the full file. Alternatively, I can proceed based on standard HAND workflow knowledge + your existing [`hand_model.py`](file:///d:/KLTN/da-framework/src/preprocessing/hand_model.py) and [`terrain_features.py`](file:///d:/KLTN/da-framework/src/preprocessing/terrain_features.py), which already implement the core HAND logic.

## Open Questions

> [!IMPORTANT]
> 1. **B11 (SWIR) band for MNDWI**: Your plan says `MNDWI = (B3 − B11) / (B3 + B11)`, but the current config only downloads `B2, B3, B4, B8`. Should I add **B11** to the Sentinel-2 download pipeline? (B11 is 20 m resolution, so it will need resampling to 10 m.)
> 2. **XGBoost/LightGBM (Model C)**: Should I implement this as a separate class in the same file, or as a standalone module (e.g., `tree_dtm_corrector.py`)? This is a non-neural approach and uses different libraries (`xgboost`/`lightgbm`).
> 3. **Spatial K-Fold**: Your plan mentions ICESat-2 track-based spatial splits. Should I implement this using `sklearn.model_selection.GroupKFold` (grouping by `track_id`) or a custom spatial block approach?
> 4. **HEC-RAS 2D**: The plan mentions HEC-RAS 2D alongside HAND. Do you want me to keep the existing [`hecras_runner.py`](file:///d:/KLTN/da-framework/src/groundtruth/hecras_runner.py) module, or is HAND the sole flood model going forward?
> 5. **PI-GAN model directory** (`src/model/`): Should I delete all PI-GAN files (generator, discriminator, physics_loss, trainer, etc.), or keep them archived?

---

## Proposed Changes

### Overview of Changes (Old → New)

| Aspect | Old Pipeline | New Pipeline |
|--------|-------------|-------------|
| **Core ML** | PI-GAN (generator + discriminator + physics loss) | Residual MLP + XGBoost DEM correctors |
| **Target variable** | Direct elevation `H` | Residual `ΔH = FabDEM − ICESat2` |
| **Loss function** | MSELoss | **Huber Loss** (robust to outliers) |
| **Features** | 5 (B2,B3,B4,B8,FabDEM) | **10** (+NDVI,MNDWI,Slope,Aspect,Dist2River) |
| **Flood model** | HAND → PI-GAN refiner | **HAND only** (direct comparison FabDEM vs ANN-DTM) |
| **Evaluation** | PI-GAN vs HEC-RAS | **Cascading impact**: DTM accuracy → flood extent accuracy |
| **Assessment** | Single scenario | **Dual DEM comparison** + SAR validation |

---

### Component 1: Feature Engineering

#### [MODIFY] [sentinel2_downloader.py](file:///d:/KLTN/da-framework/src/data_acquisition/sentinel2_downloader.py)
- Add **B11 (SWIR1)** to the band list for MNDWI computation.

#### [NEW] `src/preprocessing/feature_engineering.py`
New module to compute the expanded 10-feature set from raw inputs:
1. Extract B2, B3, B4, B8 from Sentinel-2
2. Compute **NDVI** = (B8 − B4) / (B8 + B4)
3. Compute **MNDWI** = (B3 − B11) / (B3 + B11)
4. Extract **FabDEM elevation** (resampled to 10 m)
5. Compute **Slope** and **Aspect** from FabDEM (via WhiteboxTools or numpy gradient)
6. Compute **Dist2River** (Euclidean distance to nearest stream cell)

These are partially already in [`terrain_features.py`](file:///d:/KLTN/da-framework/src/preprocessing/terrain_features.py) — we'll refactor to create a unified feature matrix builder.

---

### Component 2: ANN DEM Corrector (Major Rewrite)

#### [MODIFY] [ann_dtm_corrector.py](file:///d:/KLTN/da-framework/src/preprocessing/ann_dtm_corrector.py)

**Key changes:**

1. **Residual learning** — target becomes `ΔH = FabDEM_elev − ICESat2_elev` instead of absolute elevation.
   ```python
   # Old: y = photons["elevation"]  (absolute ICESat-2 height)
   # New: y = photons["fabdem_elev"] - photons["elevation"]  (residual error)
   ```
   At prediction: `H_corrected = FabDEM − ΔH_predicted`

2. **Huber Loss** — replace `nn.MSELoss()` with `nn.SmoothL1Loss()` (Huber) for outlier robustness.

3. **10-feature input** — expand `input_dim` from 5 → 10: `[B2, B3, B4, B8, NDVI, MNDWI, FabDEM, Slope, Aspect, Dist2River]`.

4. **Architecture** — update `DEMCorrectorANN`:
   ```
   Input(10) → 256 → ReLU → BN → Drop(0.3)
            → 128 → ReLU → BN → Drop(0.2)
            → 64  → ReLU → BN → Drop(0.1)
            → 32  → ReLU → BN
            → 1   (predicted ΔH)
   ```

5. **Model B naming** — rename as `ResidualMLP` to distinguish from the old baseline.

6. **Spatial K-Fold** — add `train_spatial_kfold()` method for track-based or block-based cross-validation.

7. **Stratified evaluation** — report MAE/RMSE/R² broken down by:
   - Floodplain (Dist2River < 1200 m) vs. Agricultural vs. Urban
   - Flat terrain (Slope < 2°) vs. Sloped (Slope ≥ 2°)

#### [NEW] `src/preprocessing/tree_dtm_corrector.py`
- **Model C**: XGBoost / LightGBM regressor with same 10-feature input and residual target.
- Implements same `prepare_training_data()` → `train()` → `predict_dtm()` interface.
- Hyperparameter tuning via `optuna` or manual grid search.

---

### Component 3: HAND Flood Model (Enhance for Dual-DEM Comparison)

#### [MODIFY] [hand_model.py](file:///d:/KLTN/da-framework/src/preprocessing/hand_model.py)
- Add `run_dual_comparison()` method that runs HAND flood simulation on **both** FabDEM 30 m and ANN-DTM 10 m, returning paired results for cascading impact analysis.
- Integrate concepts from the reference `HAND workflow.py` repo (physical stream threshold in km², multi-threshold inundation maps).

#### [MODIFY] [terrain_features.py](file:///d:/KLTN/da-framework/src/preprocessing/terrain_features.py)
- Keep as-is (already computes Slope, Aspect, TWI, HAND, Dist2River via WhiteboxTools).
- Add a method to return features as a stacked numpy array ready for the corrector.

---

### Component 4: Cascading Impact Evaluation

#### [MODIFY] [metrics.py](file:///d:/KLTN/da-framework/src/assessment/metrics.py)
- Already has CSI, F1, POD, FAR, RMSE, MAE — these are correct.
- Add **R² score** and **percentage improvement** metrics for DTM accuracy evaluation.
- Add **depth MAE/RMSE** for flood depth comparison between the two DEM scenarios.

#### [MODIFY] [scenario_runner.py](file:///d:/KLTN/da-framework/src/assessment/scenario_runner.py)
- **Replace PI-GAN references** with dual-DEM HAND comparison logic.
- Remove `PIGANInference` import and usage.
- Add `run_cascading_impact()` method that:
  1. Runs HAND on FabDEM 30 m → baseline flood map
  2. Runs HAND on ANN-DTM 10 m → improved flood map
  3. Compares both against SAR-derived flood extent (Sentinel-1)
  4. Reports CSI, F1, FAR deltas (improvement).

#### [MODIFY] [risk_mapper.py](file:///d:/KLTN/da-framework/src/assessment/risk_mapper.py)
- Update to overlay 10 m flood maps on the Red River landscape corridor boundaries (12 BT plots, monorail route, 16 wards).

---

### Component 5: Configuration & Pipeline Wiring

#### [MODIFY] [base_config.yaml](file:///d:/KLTN/da-framework/configs/base_config.yaml)
- Remove PI-GAN loss weights section.
- Add `ann_dtm` section with hyperparams (epochs, lr, batch_size, features list).
- Add `tree_model` section for XGBoost/LightGBM params.
- Add `flood_scenarios` section with water level thresholds.
- Update `data_sources.sentinel2.bands` to include B11.

#### [MODIFY] [src/__init__.py](file:///d:/KLTN/da-framework/src/__init__.py)
- Update module docstring to reflect new pipeline purpose.

---

### Component 6: Files to Deprecate/Remove

#### [DELETE] `src/model/generator.py` — PI-GAN generator
#### [DELETE] `src/model/discriminator.py` — PI-GAN discriminator
#### [DELETE] `src/model/physics_loss.py` — SWE physics loss
#### [DELETE] `src/model/boundary_condition_loss.py` — BC loss
#### [DELETE] `src/model/fourier_features.py` — Fourier feature encoding
#### [DELETE] `src/model/trainer.py` — PI-GAN trainer
#### [DELETE] `src/model/inference.py` — PI-GAN inference
#### [DELETE] `src/model/dataset.py` — PI-GAN dataset
#### [DELETE] `src/model/data_assimilation.py` — DA constraints

> [!CAUTION]
> All 9 files in `src/model/` are PI-GAN specific and have **no role** in the new pipeline. Confirm whether to delete or archive before I proceed.

---

## Verification Plan

### Automated Tests
```bash
# Verify imports and module structure
python -c "from src.preprocessing.ann_dtm_corrector import ANNDTMCorrector; print('OK')"
python -c "from src.preprocessing.feature_engineering import FeatureEngineer; print('OK')"
python -c "from src.preprocessing.tree_dtm_corrector import TreeDTMCorrector; print('OK')"
python -c "from src.assessment.scenario_runner import ScenarioRunner; print('OK')"

# Verify config loads correctly
python -c "from src.utils.geo_utils import load_config; c = load_config(); print(c.keys())"
```

### Manual Verification
- Run the full pipeline on sample data (small subset of study area)
- Verify that `ann_dtm_corrector.py` produces residual predictions (ΔH) and correctly applies `H_corrected = FabDEM − ΔH`
- Verify HAND dual-comparison produces flood maps from both DEMs
- Check metrics output includes CSI/F1/FAR improvement deltas
