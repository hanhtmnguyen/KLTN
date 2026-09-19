"""
Step ④ — ANN-based DEM Correction (the FIRST ML model).

Trains a simple feedforward neural network to correct FabDEM (30 m)
elevation errors using Sentinel-2 spectral bands (B2, B3, B4, B8) and
ICESat-2 ground photon elevations as ground truth labels.

The trained model predicts corrected elevation at every 10 m pixel,
producing the high-resolution bare-earth DTM used throughout the pipeline.

Reference
---------
"Improving 2D hydraulic modelling in floodplain areas with ICESat-2 data:
A case study in the Upstream Yellow River"
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
import joblib

from src.utils.logging_config import get_logger
from src.utils.geo_utils import (
    load_raster,
    save_raster,
    sample_raster_at_points,
)

logger = get_logger("pigan.phase1.ann")


# ---------------------------------------------------------------------------
# ANN architecture
# ---------------------------------------------------------------------------

class DEMCorrectorANN(nn.Module):
    """Simple feedforward ANN for DEM elevation correction.

    Input features (5):
        B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), FabDEM elevation.

    Output (1):
        Corrected elevation (metres above reference).

    Architecture matches the Yellow River reference paper.
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

            nn.Linear(32, 1),  # Linear activation — elevation output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ANNDTMCorrector:
    """End-to-end workflow for ANN DEM correction.

    Parameters
    ----------
    config : dict
        Full pipeline configuration.
    device : str
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(self, config: dict, device: str = "auto"):
        self.config = config
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = DEMCorrectorANN().to(self.device)
        self.scaler = StandardScaler()

        self.model_dir = Path(config["paths"]["models"]) / "ann_dtm"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info("ANN DEM Corrector initialised on %s", self.device)

    def prepare_training_data(
        self,
        photons_path: str | Path,
        sentinel2_path: str | Path,
        fabdem_path: str | Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create feature matrix X and target vector y from data sources.

        Parameters
        ----------
        photons_path : str | Path
            GeoPackage with ICESat-2 ground photons (columns: lon, lat, elevation).
        sentinel2_path : str | Path
            Multi-band Sentinel-2 GeoTIFF (bands: B2, B3, B4, B8).
        fabdem_path : str | Path
            FabDEM GeoTIFF (single band).

        Returns
        -------
        X : np.ndarray, shape (n_samples, 5)
            Features: [B2, B3, B4, B8, FabDEM_elev].
        y : np.ndarray, shape (n_samples,)
            Target: ICESat-2 elevation.
        """
        logger.info("Preparing training data...")

        # Load photon points
        photons = gpd.read_file(photons_path)
        logger.info("  Loaded %d ground photons", len(photons))

        # Sample Sentinel-2 bands at photon locations
        for band_idx, band_name in enumerate(["B2", "B3", "B4", "B8"], start=1):
            photons = sample_raster_at_points(
                sentinel2_path, photons, band=band_idx, column_name=band_name
            )

        # Sample FabDEM at photon locations
        photons = sample_raster_at_points(
            fabdem_path, photons, band=1, column_name="fabdem_elev"
        )

        # Drop rows with NoData
        feature_cols = ["B2", "B3", "B4", "B8", "fabdem_elev"]
        photons = photons.dropna(subset=feature_cols + ["elevation"])

        # Remove outliers (elevation difference > 50m likely erroneous)
        elev_diff = np.abs(photons["elevation"] - photons["fabdem_elev"])
        photons = photons[elev_diff < 50]

        X = photons[feature_cols].values.astype(np.float32)
        y = photons["elevation"].values.astype(np.float32)

        logger.info(
            "  Training data: %d samples, %d features", X.shape[0], X.shape[1]
        )
        return X, y

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        batch_size: int = 256,
        lr: float = 1e-3,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        patience: int = 20,
    ) -> dict[str, list[float]]:
        """Train the ANN.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, 5).
        y : np.ndarray
            Target elevations (n_samples,).
        epochs : int
        batch_size : int
        lr : float
        val_fraction : float
        test_fraction : float
        patience : int
            Early stopping patience (epochs without improvement).

        Returns
        -------
        dict
            Training history: ``{"train_loss": [...], "val_loss": [...]}``.
        """
        # Split data
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_fraction, random_state=42
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

        # Normalise features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        # To tensors
        train_ds = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(y_val)
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # Optimiser + scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        criterion = nn.MSELoss()

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            # ---- Train ----
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

            # ---- Validate ----
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = self.model(xb)
                    val_losses.append(criterion(pred, yb).item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            scheduler.step(val_loss)

            if epoch % 20 == 0 or epoch == 1:
                logger.info(
                    "  Epoch %3d/%d — train_loss=%.4f, val_loss=%.4f, lr=%.2e",
                    epoch, epochs, train_loss, val_loss,
                    optimizer.param_groups[0]["lr"],
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self._save_checkpoint("best")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info("  Early stopping at epoch %d", epoch)
                    break

        # ---- Test evaluation ----
        self._load_checkpoint("best")
        self.model.eval()
        test_ds = TensorDataset(
            torch.from_numpy(X_test), torch.from_numpy(y_test)
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
        rmse = np.sqrt(np.mean((test_preds - test_targets) ** 2))
        mae = np.mean(np.abs(test_preds - test_targets))
        logger.info("  Test RMSE = %.4f m, MAE = %.4f m", rmse, mae)

        return history

    def predict_dtm(
        self,
        sentinel2_path: str | Path,
        fabdem_path: str | Path,
        output_path: str | Path,
        target_resolution: float = 10.0,
    ) -> Path:
        """Apply the trained ANN to produce a corrected 10 m DTM.

        The model predicts elevation for every pixel in the study area
        using the 5 input features (B2, B3, B4, B8, FabDEM_elev).

        Parameters
        ----------
        sentinel2_path : str | Path
            4-band Sentinel-2 composite (B2, B3, B4, B8) at 10 m.
        fabdem_path : str | Path
            FabDEM GeoTIFF (resampled to 10 m to match Sentinel-2 grid).
        output_path : str | Path
            Where to save the corrected DTM.
        target_resolution : float
            Target pixel size (metres).

        Returns
        -------
        Path
        """
        import rasterio
        from rasterio.enums import Resampling

        self._load_checkpoint("best")
        self.model.eval()

        logger.info("Predicting corrected DTM over full study area...")

        # Load Sentinel-2 bands (C, H, W)
        with rasterio.open(sentinel2_path) as src:
            s2_data = src.read().astype(np.float32)  # (4, H, W)
            profile = dict(src.profile)
            height, width = s2_data.shape[1], s2_data.shape[2]

        # Load and resample FabDEM to match Sentinel-2 grid
        with rasterio.open(fabdem_path) as src:
            fabdem = src.read(
                1,
                out_shape=(height, width),
                resampling=Resampling.bilinear,
            ).astype(np.float32)

        # Stack features: (5, H, W) → reshape to (H*W, 5)
        features = np.stack([
            s2_data[0],  # B2
            s2_data[1],  # B3
            s2_data[2],  # B4
            s2_data[3],  # B8
            fabdem,
        ], axis=0)  # (5, H, W)

        features_flat = features.reshape(5, -1).T  # (H*W, 5)

        # Normalise
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

        dtm = np.concatenate(predictions).reshape(height, width)

        # Save
        output_path = Path(output_path)
        profile.update(count=1, dtype="float32")
        save_raster(output_path, dtm, profile, nodata=-9999.0)

        logger.info("Corrected 10m DTM saved → %s  (%dx%d)", output_path, width, height)
        return output_path

    # --- Checkpoint helpers ---

    def _save_checkpoint(self, tag: str = "best") -> None:
        ckpt = {
            "model_state": self.model.state_dict(),
            "scaler_mean": self.scaler.mean_,
            "scaler_scale": self.scaler.scale_,
        }
        path = self.model_dir / f"ann_dtm_{tag}.pt"
        torch.save(ckpt, path)
        # Also save scaler separately for sklearn compatibility
        joblib.dump(self.scaler, self.model_dir / f"scaler_{tag}.pkl")

    def _load_checkpoint(self, tag: str = "best") -> None:
        path = self.model_dir / f"ann_dtm_{tag}.pt"
        if not path.exists():
            logger.warning("Checkpoint not found: %s", path)
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        scaler_path = self.model_dir / f"scaler_{tag}.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        logger.info("Loaded checkpoint: %s", path)
