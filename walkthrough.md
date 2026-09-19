# PI-GAN Flood Super-Resolution Pipeline — Walkthrough

## Summary

Built the complete Python codebase for a **Physics-Informed GAN** pipeline that super-resolves coarse HAND flood maps to HEC-RAS-quality 10m flood maps for the Red River, Hanoi. The architecture: **HAND (fast input) → PI-GAN → HEC-RAS quality (target)**.

## Files Created (27 files)

### Foundation (4 files)

| File | Purpose |
|:---|:---|
| [requirements.txt](file:///d:/KLTN/requirements.txt) | All Python dependencies (geospatial, ML, satellite data) |
| [base_config.yaml](file:///d:/KLTN/configs/base_config.yaml) | Global settings: study area bbox, CRS, resolutions, Manning's n values, gauge stations |
| [geo_utils.py](file:///d:/KLTN/src/utils/geo_utils.py) | Raster I/O, reprojection, resampling, point sampling, vector rasterization |
| [logging_config.py](file:///d:/KLTN/src/utils/logging_config.py) | Structured logging with ANSI colour console + rotating file handler |
| [visualization.py](file:///d:/KLTN/src/utils/visualization.py) | Flood depth, risk classification, comparison, and training curve plots |

---

### Phase 1: DTM Generation (7 files)

| File | Step | Purpose |
|:---|:---|:---|
| [icesat2_downloader.py](file:///d:/KLTN/src/phase1_dtm/icesat2_downloader.py) | ① | Download ATL03 via `icepyx`, extract ground photons (confidence ≥ 3) |
| [sentinel2_downloader.py](file:///d:/KLTN/src/phase1_dtm/sentinel2_downloader.py) | ② | GEE-based cloud-free composite of B2/B3/B4/B8 at 10m |
| [fabdem_downloader.py](file:///d:/KLTN/src/phase1_dtm/fabdem_downloader.py) | ③ | Download + reproject FabDEM 30m tiles |
| [ann_dtm_corrector.py](file:///d:/KLTN/src/phase1_dtm/ann_dtm_corrector.py) | ④ | ANN: `[B2,B3,B4,B8,FabDEM] → corrected elevation`. Includes train/predict/checkpoint |
| [osm_buildings.py](file:///d:/KLTN/src/phase1_dtm/osm_buildings.py) | ⑤ | Extract + classify OSM building footprints via `osmnx` |
| [roughness_matrix.py](file:///d:/KLTN/src/phase1_dtm/roughness_matrix.py) | ⑥ | NDVI/NDWI + building density → Manning's n 10m raster |
| [terrain_features.py](file:///d:/KLTN/src/phase1_dtm/terrain_features.py) | ⑦ | WhiteboxTools: slope, curvature, TWI, HAND, distance-to-river, aspect |

---

### Phase 2: Ground Truth Generation (3 files)

| File | Step | Purpose |
|:---|:---|:---|
| [hand_model.py](file:///d:/KLTN/src/phase2_groundtruth/hand_model.py) | ⑩ | HAND flood model: `depth = max(0, WL - HAND)`, Manning velocity decomposition |
| [hecras_runner.py](file:///d:/KLTN/src/phase2_groundtruth/hecras_runner.py) | ⑧⑨ | HEC-RAS COM automation + HDF5 result extraction with mesh-to-grid interpolation |
| [training_data_builder.py](file:///d:/KLTN/src/phase2_groundtruth/training_data_builder.py) | ⑪ | Aligns HAND/HEC-RAS/terrain to same grid, creates 256×256 tiled `.npz` pairs |

---

### Phase 3: PI-GAN (8 files)

| File | Purpose |
|:---|:---|
| [fourier_features.py](file:///d:/KLTN/src/phase3_pigan/fourier_features.py) | `γ(v) = [cos(2πBv), sin(2πBv)]` with frozen Gaussian B matrix |
| [generator.py](file:///d:/KLTN/src/phase3_pigan/generator.py) | U-Net: 520ch input (8 physical + 512 Fourier) → 3ch output (h, u, v) |
| [discriminator.py](file:///d:/KLTN/src/phase3_pigan/discriminator.py) | PatchGAN: classifies local patches as real/fake |
| [physics_loss.py](file:///d:/KLTN/src/phase3_pigan/physics_loss.py) | 2D SWE residuals: mass + momentum conservation with Manning friction |
| [data_assimilation.py](file:///d:/KLTN/src/phase3_pigan/data_assimilation.py) | Gauge station point loss + SAR flood extent BCE loss |
| [dataset.py](file:///d:/KLTN/src/phase3_pigan/dataset.py) | PyTorch Dataset with flip/rotation augmentation |
| [trainer.py](file:///d:/KLTN/src/phase3_pigan/trainer.py) | 3-phase curriculum (warm-up → physics → full GAN), TensorBoard, checkpoints |
| [inference.py](file:///d:/KLTN/src/phase3_pigan/inference.py) | Tiled prediction with overlap-average stitching for full-domain output |

---

### Phase 4: Risk Assessment (3 files)

| File | Purpose |
|:---|:---|
| [metrics.py](file:///d:/KLTN/src/phase4_assessment/metrics.py) | CSI, F1, RMSE, MAE, POD, FAR with wet/dry threshold classification |
| [scenario_runner.py](file:///d:/KLTN/src/phase4_assessment/scenario_runner.py) | Runs HAND→PI-GAN for multiple scenarios, computes timing + metrics |
| [risk_mapper.py](file:///d:/KLTN/src/phase4_assessment/risk_mapper.py) | Depth → risk classification (Low/Moderate/High/Very High), scenario diff maps |

---

### CLI Scripts (2 files)

| File | Purpose |
|:---|:---|
| [run_phase1.py](file:///d:/KLTN/scripts/run_phase1.py) | Click-based CLI: runs all 7 Phase 1 steps with skip options |
| [run_phase3.py](file:///d:/KLTN/scripts/run_phase3.py) | Click-based CLI: PI-GAN training with resume support |

---

## Key Design Decisions

1. **HAND as input, HEC-RAS as target** — PI-GAN bridges the speed/accuracy gap
2. **Fourier Feature Embeddings** — prevent spectral bias, enable sharp urban edge learning
3. **3-phase curriculum training** — stabilises GAN training by gradually adding loss components
4. **Event-level train/test split** — prevents data leakage between tiles from the same flood event
5. **Overlap-average tiling** — eliminates seam artifacts at inference time
6. **Manning friction in physics loss** — makes the SWE residual terrain-aware via roughness coefficient

## Next Steps

To actually run the pipeline you need to:
1. Install dependencies: `pip install -r requirements.txt`
2. Set up NASA Earthdata credentials (`.netrc` file)
3. Authenticate Google Earth Engine: `earthengine authenticate`
4. Run Phase 1: `python scripts/run_phase1.py`
5. Set up HEC-RAS model manually in RAS Mapper GUI
6. Run Phase 3: `python scripts/run_phase3.py`
