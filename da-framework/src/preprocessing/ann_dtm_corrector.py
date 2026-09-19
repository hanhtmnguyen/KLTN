"""
Step ④ — ANN-based DEM Correction via Residual Learning.

Trains a feedforward neural network (Residual MLP) to predict the
**vertical error** (residual) of FabDEM using 10 spatial features
and ICESat-2 ground photon elevations as ground truth:

    ΔH = H_FabDEM − H_ICESat-2
    H_corrected = H_FabDEM − ΔH_predicted

Uses Huber Loss (SmoothL1Loss) for robustness against photon outliers,
and supports Spatial K-Fold (GroupKFold by ICESat-2 track_id) to
prevent spatial data leakage during cross-validation.

Model B architecture (Residual MLP):
    Input(10) → 256 → ReLU → BN → Drop(0.3)
             → 128 → ReLU → BN → Drop(0.2)
             →  64 → ReLU → BN → Drop(0.1)
             →  32 → ReLU → BN
             →   1  (predicted ΔH)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import geopandas as gpd
import joblib

from src.utils.logging_config import get_logger
from src.utils.geo_utils import (
    load_raster,
    save_raster,
    sample_raster_at_points,
)

logger = get_logger("da.prep.ann")


# ── Feature list ─────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "B2", "B3", "B4", "B8",
    "NDVI", "MNDWI",
    "fabdem_elev",
    "slope", "aspect", "dist2river",
]


# ── Network architecture ─────────────────────────────────────────────────

class ResidualMLP(nn.Module):
    """Residual MLP for DEM elevation error prediction (Model B).

    Input features (10):
        B2, B3, B4, B8, NDVI, MNDWI, FabDEM_elev, Slope, Aspect, Dist2River.

    Output (1):
        Predicted residual ΔH = FabDEM − ICESat-2 (metres).
    """

    def __init__(self, input_dim: int = 10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


# ── Legacy architecture (Model A — Baseline MLP) ─────────────────────────

class BaselineMLP(nn.Module):
    """Original 5-feature MLP that directly predicts absolute elevation.

    Kept for benchmarking against the Residual MLP (Model B).
    """

    def __init__(self, input_dim: int = 5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


# ── Main corrector workflow ──────────────────────────────────────────────

class ANNDTMCorrector:
    """End-to-end workflow for ANN-based DEM correction using Residual Learning.

    Parameters
    ----------
    config : dict
        Full pipeline configuration.
    device : str
        ``"cuda"``, ``"cpu"``, or ``"auto"``.
    model_variant : str
        ``"residual"`` for Model B (default) or ``"baseline"`` for Model A.
    """

    def __init__(
        self,
        config: dict,
        device: str = "auto",
        model_variant: str = "residual",
    ):
        self.config = config
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_variant = model_variant
        ann_cfg = config.get("ann_dtm", {})
        self.residual_learning = ann_cfg.get("training", {}).get("residual_learning", True)
        self.outlier_threshold = ann_cfg.get("outlier_threshold_m", 50)

        if model_variant == "baseline":
            self.model = BaselineMLP(input_dim=5).to(self.device)
            self.feature_cols = ["B2", "B3", "B4", "B8", "fabdem_elev"]
        else:
            input_dim = len(FEATURE_NAMES)
            self.model = ResidualMLP(input_dim=input_dim).to(self.device)
            self.feature_cols = list(FEATURE_NAMES)

        self.scaler = StandardScaler()

        self.model_dir = Path(config["paths"]["models"]) / "ann_dtm"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "ANN DEM Corrector initialised: variant=%s, residual=%s, device=%s",
            model_variant, self.residual_learning, self.device,
        )

    # ── Data preparation ─────────────────────────────────────────────

    def prepare_training_data(
        self,
        photons: gpd.GeoDataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Create feature matrix X, target vector y, and optional group IDs.

        Parameters
        ----------
        photons : gpd.GeoDataFrame
            Must contain all feature columns and ``elevation`` (ICESat-2)
            and ``fabdem_elev`` columns.  Optionally ``track_id`` for
            spatial cross-validation.

        Returns
        -------
        X : np.ndarray  shape (N, n_features)
        y : np.ndarray  shape (N,)
        groups : np.ndarray | None  shape (N,) — track IDs for GroupKFold
        """
        logger.info("Preparing training data...")

        # Drop rows with missing values
        required_cols = self.feature_cols + ["elevation", "fabdem_elev"]
        photons = photons.dropna(subset=required_cols)

        # Outlier filter
        elev_diff = np.abs(photons["elevation"] - photons["fabdem_elev"])
        photons = photons[elev_diff < self.outlier_threshold]

        X = photons[self.feature_cols].values.astype(np.float32)

        if self.residual_learning:
            # Target: ΔH = FabDEM − ICESat-2  (the error to predict)
            y = (photons["fabdem_elev"] - photons["elevation"]).values.astype(np.float32)
            logger.info("  Residual target: dH = FabDEM - ICESat-2")
        else:
            # Direct elevation (legacy)
            y = photons["elevation"].values.astype(np.float32)
            logger.info("  Direct target: absolute ICESat-2 elevation")

        groups = None
        if "track_id" in photons.columns:
            groups = photons["track_id"].values
            n_tracks = len(np.unique(groups))
            logger.info("  Track IDs found: %d unique tracks for spatial CV", n_tracks)

        logger.info(
            "  Training data: %d samples, %d features", X.shape[0], X.shape[1],
        )
        return X, y, groups

    # ── Training ─────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None = None,
        epochs: int = 300,
        batch_size: int = 256,
        lr: float = 1e-3,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        patience: int = 25,
    ) -> dict[str, list[float]]:
        """Train the ANN with Huber Loss and early stopping.

        Returns
        -------
        dict
            Training history with ``train_loss``, ``val_loss`` lists and
            ``test_metrics`` dict.
        """
        # ── Split data ────────────────────────────────────────────────
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_fraction, random_state=42,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_fraction / (1 - test_fraction),
            random_state=42,
        )

        logger.info(
            "Split: train=%d, val=%d, test=%d",
            len(X_train), len(X_val), len(X_test),
        )

        # ── Standardize ──────────────────────────────────────────────
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        # ── Data loaders ─────────────────────────────────────────────
        train_ds = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train),
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(y_val),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # ── Optimizer, scheduler, loss ────────────────────────────────
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10,
        )
        criterion = nn.SmoothL1Loss()  # Huber Loss — robust to outliers

        # ── Training loop ────────────────────────────────────────────
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = self.model(xb)
                    val_losses.append(criterion(pred, yb).item())

            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            scheduler.step(val_loss)

            if epoch % 20 == 0 or epoch == 1:
                logger.info(
                    "  Epoch %3d/%d — train_loss=%.4f, val_loss=%.4f, lr=%.2e",
                    epoch, epochs, train_loss, val_loss,
                    optimizer.param_groups[0]["lr"],
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self._save_checkpoint("best")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info("  Early stopping at epoch %d", epoch)
                    break

        # ── Test evaluation ──────────────────────────────────────────
        self._load_checkpoint("best")
        test_metrics = self._evaluate_test(X_test, y_test, batch_size)
        history["test_metrics"] = test_metrics

        return history

    # ── Spatial K-Fold Cross-Validation ──────────────────────────────

    def train_spatial_kfold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        n_splits: int = 5,
        epochs: int = 300,
        batch_size: int = 256,
        lr: float = 1e-3,
        patience: int = 25,
    ) -> list[dict]:
        """Train and evaluate with GroupKFold (grouped by ICESat-2 track_id).

        Returns
        -------
        list[dict]
            Per-fold metrics: RMSE, MAE, R².
        """
        gkf = GroupKFold(n_splits=n_splits)
        fold_results = []

        for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
            logger.info("-- Spatial K-Fold %d/%d --", fold_i, n_splits)

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Further split train into train/val
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.15, random_state=42,
            )

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test)

            # Fresh model for each fold
            if self.model_variant == "baseline":
                model = BaselineMLP(input_dim=5).to(self.device)
            else:
                model = ResidualMLP(input_dim=len(self.feature_cols)).to(self.device)

            train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
            val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10,
            )
            criterion = nn.SmoothL1Loss()

            best_val = float("inf")
            no_improve = 0
            best_state = None

            for epoch in range(1, epochs + 1):
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    loss = criterion(model(xb), yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                model.eval()
                vl = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        vl.append(criterion(model(xb), yb).item())
                val_loss = float(np.mean(vl))
                scheduler.step(val_loss)

                if val_loss < best_val:
                    best_val = val_loss
                    no_improve = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break

            model.load_state_dict(best_state)
            model.eval()

            # Evaluate on test fold
            preds = []
            with torch.no_grad():
                test_t = torch.from_numpy(X_test_s)
                for i in range(0, len(test_t), batch_size):
                    batch = test_t[i:i + batch_size].to(self.device)
                    preds.append(model(batch).cpu().numpy())
            preds = np.concatenate(preds)

            rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
            mae = float(np.mean(np.abs(preds - y_test)))
            r2 = float(r2_score(y_test, preds))
            
            errors = y_test - preds
            nmad = float(1.4826 * np.median(np.abs(errors - np.median(errors))))

            fold_results.append({"fold": fold_i, "RMSE": rmse, "MAE": mae, "NMAD": nmad, "R2": r2})
            logger.info("  Fold %d: RMSE=%.4f m, MAE=%.4f m, NMAD=%.4f m, R²=%.4f", fold_i, rmse, mae, nmad, r2)

        # Summary
        avg_rmse = float(np.mean([f["RMSE"] for f in fold_results]))
        avg_mae = float(np.mean([f["MAE"] for f in fold_results]))
        avg_nmad = float(np.mean([f["NMAD"] for f in fold_results]))
        avg_r2 = float(np.mean([f["R2"] for f in fold_results]))
        logger.info(
            "Spatial K-Fold summary: RMSE=%.4f, MAE=%.4f, NMAD=%.4f, R²=%.4f",
            avg_rmse, avg_mae, avg_nmad, avg_r2,
        )
        return fold_results

    # ── Stratified evaluation ────────────────────────────────────────

    def evaluate_stratified(
        self,
        photons: gpd.GeoDataFrame,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> dict:
        """Report MAE/RMSE/R² stratified by terrain type and slope.

        Parameters
        ----------
        photons : gpd.GeoDataFrame
            Must have ``dist2river`` and ``slope`` columns.
        predictions, targets : np.ndarray
            Model predictions and ground-truth values.

        Returns
        -------
        dict
            Nested dict with per-stratum metrics.
        """
        eval_cfg = self.config.get("evaluation", {}).get("stratification", {})
        floodplain_dist = eval_cfg.get("floodplain_dist_m", 1200)
        slope_threshold = eval_cfg.get("slope_threshold_deg", 2.0)

        results = {}

        # ── By distance to river ──
        if "dist2river" in photons.columns:
            d2r = photons["dist2river"].values
            for label, mask in [
                ("floodplain", d2r < floodplain_dist),
                ("non_floodplain", d2r >= floodplain_dist),
            ]:
                if mask.sum() > 0:
                    p, t = predictions[mask], targets[mask]
                    results[label] = {
                        "n_samples": int(mask.sum()),
                        "RMSE": float(np.sqrt(np.mean((p - t) ** 2))),
                        "MAE": float(np.mean(np.abs(p - t))),
                        "R2": float(r2_score(t, p)) if len(t) > 1 else 0.0,
                    }

        # ── By slope class ──
        if "slope" in photons.columns:
            s = photons["slope"].values
            for label, mask in [
                ("flat", s < slope_threshold),
                ("sloped", s >= slope_threshold),
            ]:
                if mask.sum() > 0:
                    p, t = predictions[mask], targets[mask]
                    results[label] = {
                        "n_samples": int(mask.sum()),
                        "RMSE": float(np.sqrt(np.mean((p - t) ** 2))),
                        "MAE": float(np.mean(np.abs(p - t))),
                        "R2": float(r2_score(t, p)) if len(t) > 1 else 0.0,
                    }

        for stratum, m in results.items():
            logger.info(
                "  [%s] n=%d  RMSE=%.4f  MAE=%.4f  R²=%.4f",
                stratum, m["n_samples"], m["RMSE"], m["MAE"], m["R2"],
            )
        return results

    # ── Prediction ───────────────────────────────────────────────────

    def predict_dtm(
        self,
        feature_stack: np.ndarray,
        fabdem: np.ndarray,
        profile: dict,
        output_path: str | Path,
    ) -> Path:
        """Apply the trained ANN to produce a corrected 10 m DTM.

        Parameters
        ----------
        feature_stack : np.ndarray
            Shape ``(10, H, W)`` from ``FeatureEngineer.build_feature_stack()``.
        fabdem : np.ndarray
            Shape ``(H, W)`` — FabDEM elevation (resampled to 10 m).
        profile : dict
            Rasterio profile for the output GeoTIFF.
        output_path : str | Path
            Destination file path.

        Returns
        -------
        Path
            Path to the saved corrected DTM.
        """
        self._load_checkpoint("best")
        self.model.eval()

        height, width = feature_stack.shape[1], feature_stack.shape[2]
        logger.info("Predicting corrected DTM (%dx%d)...", width, height)

        # Flatten to (N, n_features)
        n_features = feature_stack.shape[0]
        features_flat = feature_stack.reshape(n_features, -1).T
        features_flat = self.scaler.transform(features_flat)

        # Predict in batches
        batch_size = 65536
        predictions = []
        with torch.no_grad():
            for i in range(0, len(features_flat), batch_size):
                batch = torch.from_numpy(
                    features_flat[i:i + batch_size]
                ).to(self.device)
                pred = self.model(batch).cpu().numpy()
                predictions.append(pred)

        delta_h = np.concatenate(predictions).reshape(height, width)

        if self.residual_learning:
            # H_corrected = FabDEM − ΔH_predicted
            dtm = fabdem - delta_h
            logger.info("  Applied residual correction: H = FabDEM − ΔH")
        else:
            dtm = delta_h
            logger.info("  Direct elevation prediction")

        dtm = dtm.astype(np.float32)
        output_path = Path(output_path)
        profile.update(count=1, dtype="float32")
        save_raster(output_path, dtm, profile, nodata=-9999.0)

        logger.info("Corrected 10m DTM saved → %s  (%dx%d)", output_path, width, height)
        return output_path

    # ── Internal helpers ─────────────────────────────────────────────

    def _evaluate_test(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 256,
    ) -> dict:
        """Evaluate on held-out test set."""
        self.model.eval()
        test_ds = TensorDataset(
            torch.from_numpy(X_test), torch.from_numpy(y_test),
        )
        test_loader = DataLoader(test_ds, batch_size=batch_size)
        test_preds, test_targets = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(self.device)
                test_preds.append(self.model(xb).cpu().numpy())
                test_targets.append(yb.numpy())
        test_preds = np.concatenate(test_preds)
        test_targets = np.concatenate(test_targets)

        rmse = float(np.sqrt(np.mean((test_preds - test_targets) ** 2)))
        mae = float(np.mean(np.abs(test_preds - test_targets)))
        r2 = float(r2_score(test_targets, test_preds))
        
        errors = test_targets - test_preds
        nmad = float(1.4826 * np.median(np.abs(errors - np.median(errors))))

        logger.info("  Test RMSE = %.4f m, MAE = %.4f m, NMAD = %.4f m, R² = %.4f", rmse, mae, nmad, r2)
        return {"RMSE": rmse, "MAE": mae, "NMAD": nmad, "R2": r2}

    def _save_checkpoint(self, tag: str = "best") -> None:
        ckpt = {
            "model_state": self.model.state_dict(),
            "scaler_mean": self.scaler.mean_,
            "scaler_scale": self.scaler.scale_,
            "model_variant": self.model_variant,
            "residual_learning": self.residual_learning,
            "feature_cols": self.feature_cols,
        }
        path = self.model_dir / f"ann_dtm_{tag}.pt"
        torch.save(ckpt, path)
        joblib.dump(self.scaler, self.model_dir / f"scaler_{tag}.pkl")

    def _load_checkpoint(self, tag: str = "best") -> None:
        path = self.model_dir / f"ann_dtm_{tag}.pt"
        if not path.exists():
            logger.warning("Checkpoint not found: %s", path)
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        scaler_path = self.model_dir / f"scaler_{tag}.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        logger.info("Loaded checkpoint: %s", path)
