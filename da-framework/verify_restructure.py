"""Quick import verification for all restructured modules."""

# Phase 1: Data acquisition
from src.data_acquisition.sentinel2_downloader import Sentinel2Downloader
print("sentinel2_downloader: OK")

# Phase 2: Feature engineering
from src.preprocessing.feature_engineering import FeatureEngineer, compute_ndvi, compute_mndwi
print("feature_engineering: OK")

from src.preprocessing.terrain_features import TerrainFeatureGenerator
print("terrain_features: OK")

# Phase 3: DEM correctors
from src.preprocessing.ann_dtm_corrector import ANNDTMCorrector, ResidualMLP, BaselineMLP
print("ann_dtm_corrector: OK")

from src.preprocessing.tree_dtm_corrector import TreeDTMCorrector
print("tree_dtm_corrector: OK")

# Phase 4: Flood simulation & evaluation
from src.preprocessing.hand_model import HANDFloodModel, DualDEMComparison
print("hand_model: OK")

from src.assessment.metrics import (
    compute_flood_metrics, compute_dtm_metrics,
    compute_cascading_impact, CascadingImpact, DTMMetrics, FloodMetrics,
)
print("metrics: OK")

from src.assessment.scenario_runner import ScenarioRunner
print("scenario_runner: OK")

from src.assessment.risk_mapper import RiskMapper
print("risk_mapper: OK")

# Config verification
from src.utils.geo_utils import load_config
c = load_config()
print(f"config keys: {list(c.keys())}")
print(f"ann_dtm features: {c['ann_dtm']['input_features']}")
print(f"tree_model backend: {c['tree_model']['backend']}")
print(f"sentinel2 bands: {c['data_sources']['sentinel2']['bands']}")
print(f"flood thresholds: {c['flood_simulation']['hand_thresholds']}")
print(f"spatial_cv method: {c['spatial_cv']['method']}")

print("\n=== ALL IMPORTS AND CONFIG OK ===")
