"""
Step ④c — Tree-Based DEM Correction (Model C).

An XGBoost / LightGBM ensemble regressor for DEM vertical error
correction, using the same 10 spatial features and residual learning
target as the Residual MLP (Model B).

    ΔH = H_FabDEM − H_ICESat-2
    H_corrected = H_FabDEM − ΔH_predicted

Provides a non-neural benchmark alongside the MLP models.
"""

from __future__ import annotations

import numpy as np
import geopandas as gpd
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

from src.utils.logging_config import get_logger
from src.utils.geo_utils import save_raster

logger = get_logger("da.prep.tree")


# ── Feature list (shared with ann_dtm_corrector) ─────────────────────────

FEATURE_NAMES = [
    "B2", "B3", "B4", "B8",
    "NDVI", "MNDWI",
    "fabdem_elev",
    "slope", "aspect", "dist2river",
]


class TreeDTMCorrector:
    """XGBoost / LightGBM DEM corrector with residual learning.

    Parameters
    ----------
    config : dict
        Full pipeline configuration.
    backend : str
        ``"xgboost"`` or ``"lightgbm"``.
    """

    def __init__(self, config: dict, backend: str | None = None):
        self.config = config
        tree_cfg = config.get("tree_model", {})
        self.backend = backend or tree_cfg.get("backend", "xgboost")
        self.residual_learning = tree_cfg.get("residual_learning", True)
        self.params = dict(tree_cfg.get("params", {}))
        self.outlier_threshold = config.get("ann_dtm", {}).get("outlier_threshold_m", 50)

        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = list(FEATURE_NAMES)

        self.model_dir = Path(config["paths"]["models"]) / "tree_dtm"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Tree DEM Corrector initialised: backend=%s, residual=%s",
            self.backend, self.residual_learning,
        )

    def _create_model(self):
        """Instantiate the tree model based on backend choice."""
        if self.backend == "xgboost":
            import xgboost as xgb
            self.model = xgb.XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                random_state=42,
                **self.params,
            )
        elif self.backend == "lightgbm":
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(
                objective="regression",
                random_state=42,
                verbose=-1,
                **self.params,
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

    # ── Data preparation (same interface as ANNDTMCorrector) ──────────

    def prepare_training_data(
        self,
        photons: gpd.GeoDataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Create feature matrix X, target vector y, and optional group IDs.

        Parameters
        ----------
        photons : gpd.GeoDataFrame
            Must contain all feature columns and ``elevation``,
            ``fabdem_elev``.  Optionally ``track_id``.

        Returns
        -------
        X, y, groups
        """
        logger.info("Preparing training data...")

        required_cols = self.feature_cols + ["elevation", "fabdem_elev"]
        photons = photons.dropna(subset=required_cols)

        elev_diff = np.abs(photons["elevation"] - photons["fabdem_elev"])
        photons = photons[elev_diff < self.outlier_threshold]

        X = photons[self.feature_cols].values.astype(np.float32)

        if self.residual_learning:
            y = (photons["fabdem_elev"] - photons["elevation"]).values.astype(np.float32)
            logger.info("  Residual target: dH = FabDEM - ICESat-2")
        else:
            y = photons["elevation"].values.astype(np.float32)

        groups = None
        if "track_id" in photons.columns:
            groups = photons["track_id"].values

        logger.info("  Data: %d samples, %d features", X.shape[0], X.shape[1])
        return X, y, groups

    # ── Training ─────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None = None,
        test_fraction: float = 0.3,
        tune_hyperparams: bool = True,
    ) -> dict:
        """Train the tree model.

        Returns
        -------
        dict
            Test metrics: RMSE, MAE, R².
        """
        self._create_model()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_fraction, random_state=42,
        )

        # Scaling (trees don't strictly need it, but keeps interface consistent)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        logger.info("Training %s on %d samples...", self.backend, len(X_train))

        if self.backend == "xgboost":
            if tune_hyperparams:
                logger.info("Tuning hyperparameters for XGBoost...")
                from sklearn.model_selection import RandomizedSearchCV
                param_dist = {
                    "max_depth": [3, 5, 7, 9],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "n_estimators": [100, 300, 500],
                    "subsample": [0.6, 0.8, 1.0],
                    "colsample_bytree": [0.6, 0.8, 1.0]
                }
                search = RandomizedSearchCV(
                    self.model, param_distributions=param_dist, n_iter=10, 
                    scoring="neg_mean_squared_error", cv=3, verbose=1, random_state=42, n_jobs=-1
                )
                search.fit(X_train, y_train)
                self.model = search.best_estimator_
                logger.info("Best parameters found: %s", search.best_params_)
            
            # Final refit on train set with eval set (useful for early stopping or logging)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=50,
            )
        elif self.backend == "lightgbm":
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[],
            )

        preds = self.model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        
        # Calculate NMAD
        errors = y_test - preds
        nmad = float(1.4826 * np.median(np.abs(errors - np.median(errors))))

        logger.info("  Test RMSE=%.4f m, MAE=%.4f m, NMAD=%.4f m, R²=%.4f", rmse, mae, nmad, r2)

        self._save_model()
        
        # Generate SHAP summary plot if possible
        if self.backend == "xgboost":
            try:
                self.generate_shap_summary(X_train)
            except Exception as e:
                logger.warning(f"Failed to generate SHAP summary: {e}")

        return {"RMSE": rmse, "MAE": mae, "NMAD": nmad, "R2": r2}

    def generate_shap_summary(self, X_train: np.ndarray):
        """Generate and save a SHAP summary plot for the trained model."""
        import shap
        import matplotlib.pyplot as plt
        
        logger.info("Generating SHAP summary plot...")
        import json
        booster = self.model.get_booster()
        try:
            b_config = json.loads(booster.save_config())
            base_score = b_config["learner"]["learner_model_param"]["base_score"]
            if isinstance(base_score, str) and base_score.startswith("["):
                # Extract the first float from the array string e.g. "[-1.2924102E-1]"
                b_config["learner"]["learner_model_param"]["base_score"] = base_score.strip("[]")
                booster.load_config(json.dumps(b_config))
        except Exception as e:
            logger.warning(f"Failed to patch XGBoost base_score for SHAP: {e}")
            
        explainer = shap.TreeExplainer(booster)
        
        # We need the feature names. Assuming X_train matches config features.
        # But we'll just use generic names or try to map them.
        feature_names = [f.replace("sentinel2_", "").replace("fabdem_", "") 
                         for f in self.config["ann_dtm"]["input_features"]]
        
        # Sample for SHAP calculation to save time
        sample_size = min(5000, len(X_train))
        X_sample = X_train[:sample_size]
        shap_values = explainer.shap_values(X_sample)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        
        out_path = Path(self.config["paths"]["outputs"]) / "shap_summary.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP summary plot saved to {out_path}")

    # ── Spatial K-Fold ───────────────────────────────────────────────

    def train_spatial_kfold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        n_splits: int = 5,
    ) -> list[dict]:
        """Train and evaluate with GroupKFold by ICESat-2 track_id."""
        gkf = GroupKFold(n_splits=n_splits)
        fold_results = []

        for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
            logger.info("-- Spatial K-Fold %d/%d --", fold_i, n_splits)

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            self._create_model()
            self.model.fit(X_train, y_train)

            preds = self.model.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))
            errors = y_test - preds
            nmad = float(1.4826 * np.median(np.abs(errors - np.median(errors))))

            fold_results.append({"fold": fold_i, "RMSE": rmse, "MAE": mae, "NMAD": nmad, "R2": r2})
            logger.info("  Fold %d: RMSE=%.4f m, MAE=%.4f m, NMAD=%.4f m, R²=%.4f", fold_i, rmse, mae, nmad, r2)

        avg_rmse = float(np.mean([f["RMSE"] for f in fold_results]))
        avg_mae = float(np.mean([f["MAE"] for f in fold_results]))
        avg_nmad = float(np.mean([f["NMAD"] for f in fold_results]))
        avg_r2 = float(np.mean([f["R2"] for f in fold_results]))
        logger.info(
            "Spatial K-Fold summary: RMSE=%.4f, MAE=%.4f, NMAD=%.4f, R²=%.4f",
            avg_rmse, avg_mae, avg_nmad, avg_r2,
        )
        return fold_results

    # ── Prediction ───────────────────────────────────────────────────

    def predict_dtm(
        self,
        feature_stack: np.ndarray,
        fabdem: np.ndarray,
        profile: dict,
        output_path: str | Path,
    ) -> Path:
        """Apply the trained tree model to produce a corrected 10 m DTM.

        Parameters
        ----------
        feature_stack : np.ndarray
            Shape ``(10, H, W)``.
        fabdem : np.ndarray
            Shape ``(H, W)`` — FabDEM elevation resampled to 10 m.
        profile : dict
            Rasterio profile.
        output_path : str | Path

        Returns
        -------
        Path
        """
        self._load_model()

        height, width = feature_stack.shape[1], feature_stack.shape[2]
        n_features = feature_stack.shape[0]
        logger.info("Predicting corrected DTM (%dx%d) with %s...", width, height, self.backend)

        features_flat = feature_stack.reshape(n_features, -1).T
        features_flat = self.scaler.transform(features_flat)

        delta_h = self.model.predict(features_flat).reshape(height, width)

        if self.residual_learning:
            dtm = fabdem - delta_h
            logger.info("  Applied residual correction: H = FabDEM − ΔH")
        else:
            dtm = delta_h

        dtm = dtm.astype(np.float32)
        output_path = Path(output_path)
        profile.update(count=1, dtype="float32")
        save_raster(output_path, dtm, profile, nodata=-9999.0)

        logger.info("Corrected 10m DTM saved → %s", output_path)
        return output_path

    # ── Feature importance ───────────────────────────────────────────

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importances from the trained model."""
        if self.model is None:
            self._load_model()
        importances = self.model.feature_importances_
        return dict(zip(self.feature_cols, importances.tolist()))

    # ── Persistence ──────────────────────────────────────────────────

    def _save_model(self) -> None:
        model_path = self.model_dir / f"{self.backend}_dtm_best.pkl"
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, self.model_dir / "scaler_tree_best.pkl")
        logger.info("Saved %s model → %s", self.backend, model_path)

    def _load_model(self) -> None:
        model_path = self.model_dir / f"{self.backend}_dtm_best.pkl"
        if not model_path.exists():
            logger.warning("Model not found: %s", model_path)
            return
        self.model = joblib.load(model_path)
        scaler_path = self.model_dir / "scaler_tree_best.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        logger.info("Loaded %s model from %s", self.backend, model_path)
