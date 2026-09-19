"""
Accuracy metrics for DTM and flood model evaluation.

Computes CSI, F1-score, RMSE, MAE, POD, FAR, R², bias, and cascading
impact improvement deltas for comparing flood results between
FabDEM 30 m (baseline) and ANN-DTM 10 m (improved).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from sklearn.metrics import r2_score as _r2_score

from src.utils.logging_config import get_logger

logger = get_logger("da.assessment.metrics")


# ── DTM accuracy metrics ────────────────────────────────────────────────

@dataclass
class DTMMetrics:
    """Container for DTM vertical accuracy evaluation."""
    rmse: float
    mae: float
    r2: float
    mean_error: float       # bias
    std_error: float
    n_samples: int

    def __str__(self) -> str:
        return (
            f"RMSE={self.rmse:.4f}m  MAE={self.mae:.4f}m  R²={self.r2:.4f}  "
            f"Bias={self.mean_error:.4f}m  n={self.n_samples}"
        )

    def to_dict(self) -> dict:
        return {
            "RMSE_m": round(self.rmse, 4),
            "MAE_m": round(self.mae, 4),
            "R2": round(self.r2, 4),
            "mean_error_m": round(self.mean_error, 4),
            "std_error_m": round(self.std_error, 4),
            "n_samples": self.n_samples,
        }


def compute_dtm_metrics(
    predicted_elev: np.ndarray,
    reference_elev: np.ndarray,
) -> DTMMetrics:
    """Compute vertical accuracy metrics for a corrected DTM.

    Parameters
    ----------
    predicted_elev : np.ndarray
        Corrected DTM elevations (or residuals).
    reference_elev : np.ndarray
        ICESat-2 / GPS reference elevations.
    """
    error = predicted_elev - reference_elev
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    r2 = float(_r2_score(reference_elev, predicted_elev)) if len(error) > 1 else 0.0
    mean_err = float(np.mean(error))
    std_err = float(np.std(error))

    return DTMMetrics(
        rmse=rmse, mae=mae, r2=r2,
        mean_error=mean_err, std_error=std_err,
        n_samples=len(error),
    )


# ── Flood accuracy metrics ──────────────────────────────────────────────

@dataclass
class FloodMetrics:
    """Container for flood model evaluation metrics."""
    csi: float
    f1: float
    rmse: float
    mae: float
    pod: float
    far: float
    bias: float
    n_tp: int
    n_fp: int
    n_fn: int
    n_tn: int

    def __str__(self) -> str:
        return (
            f"CSI={self.csi:.4f}  F1={self.f1:.4f}  RMSE={self.rmse:.4f}m  "
            f"MAE={self.mae:.4f}m  POD={self.pod:.4f}  FAR={self.far:.4f}"
        )

    def to_dict(self) -> dict:
        return {
            "CSI": round(self.csi, 4),
            "F1": round(self.f1, 4),
            "RMSE_m": round(self.rmse, 4),
            "MAE_m": round(self.mae, 4),
            "POD": round(self.pod, 4),
            "FAR": round(self.far, 4),
            "Bias": round(self.bias, 4),
        }


def compute_flood_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    depth_threshold: float = 0.05,
) -> FloodMetrics:
    """Compute comprehensive flood accuracy metrics."""
    pred_wet = predicted > depth_threshold
    obs_wet = observed > depth_threshold

    tp = np.sum(pred_wet & obs_wet)
    fp = np.sum(pred_wet & ~obs_wet)
    fn = np.sum(~pred_wet & obs_wet)
    tn = np.sum(~pred_wet & ~obs_wet)

    eps = 1e-10

    csi = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    pod = tp / (tp + fn + eps)
    far = fp / (tp + fp + eps)
    bias = (tp + fp) / (tp + fn + eps)

    wet_mask = obs_wet | pred_wet
    if wet_mask.sum() > 0:
        rmse = np.sqrt(np.mean((predicted[wet_mask] - observed[wet_mask]) ** 2))
        mae = np.mean(np.abs(predicted[wet_mask] - observed[wet_mask]))
    else:
        rmse = 0.0
        mae = 0.0

    return FloodMetrics(
        csi=float(csi), f1=float(f1), rmse=float(rmse), mae=float(mae),
        pod=float(pod), far=float(far), bias=float(bias),
        n_tp=int(tp), n_fp=int(fp), n_fn=int(fn), n_tn=int(tn),
    )


# ── Cascading impact improvement ────────────────────────────────────────

@dataclass
class CascadingImpact:
    """Improvement deltas between baseline (FabDEM) and improved (ANN-DTM) floods."""
    baseline_metrics: FloodMetrics
    improved_metrics: FloodMetrics
    delta_csi: float       # improved − baseline (positive = better)
    delta_f1: float
    delta_far: float       # baseline − improved (positive = fewer false alarms)
    delta_depth_mae: float # baseline − improved (positive = less error)
    delta_depth_rmse: float

    def __str__(self) -> str:
        return (
            f"ΔCSI={self.delta_csi:+.4f}  ΔF1={self.delta_f1:+.4f}  "
            f"ΔFAR={self.delta_far:+.4f}  ΔMAE={self.delta_depth_mae:+.4f}m  "
            f"ΔRMSE={self.delta_depth_rmse:+.4f}m"
        )

    def to_dict(self) -> dict:
        return {
            "baseline": self.baseline_metrics.to_dict(),
            "improved": self.improved_metrics.to_dict(),
            "delta_CSI": round(self.delta_csi, 4),
            "delta_F1": round(self.delta_f1, 4),
            "delta_FAR": round(self.delta_far, 4),
            "delta_depth_MAE_m": round(self.delta_depth_mae, 4),
            "delta_depth_RMSE_m": round(self.delta_depth_rmse, 4),
        }


def compute_cascading_impact(
    baseline_flood: np.ndarray,
    improved_flood: np.ndarray,
    reference_flood: np.ndarray,
    depth_threshold: float = 0.05,
) -> CascadingImpact:
    """Evaluate cascading impact: how DTM improvement affects flood accuracy.

    Parameters
    ----------
    baseline_flood : np.ndarray
        Flood depth from HAND on FabDEM 30 m.
    improved_flood : np.ndarray
        Flood depth from HAND on ANN-DTM 10 m.
    reference_flood : np.ndarray
        Reference flood extent (e.g. from SAR or HEC-RAS).
    depth_threshold : float
        Wet/dry threshold in metres.

    Returns
    -------
    CascadingImpact
    """
    baseline_m = compute_flood_metrics(baseline_flood, reference_flood, depth_threshold)
    improved_m = compute_flood_metrics(improved_flood, reference_flood, depth_threshold)

    impact = CascadingImpact(
        baseline_metrics=baseline_m,
        improved_metrics=improved_m,
        delta_csi=improved_m.csi - baseline_m.csi,
        delta_f1=improved_m.f1 - baseline_m.f1,
        delta_far=baseline_m.far - improved_m.far,    # positive = improvement
        delta_depth_mae=baseline_m.mae - improved_m.mae,
        delta_depth_rmse=baseline_m.rmse - improved_m.rmse,
    )

    logger.info("Cascading impact evaluation:")
    logger.info("  Baseline (FabDEM 30m): %s", baseline_m)
    logger.info("  Improved (ANN-DTM 10m): %s", improved_m)
    logger.info("  Impact: %s", impact)

    return impact


# ── Spatiotemporal metrics (retained for time-series analysis) ───────────

def compute_spatiotemporal_metrics(
    predicted_series: list[np.ndarray],
    observed_series: list[np.ndarray],
    depth_threshold: float = 0.05,
) -> dict[str, dict | float]:
    """Compute metrics across an entire time series of flood frames."""
    n_steps = min(len(predicted_series), len(observed_series))
    step_metrics = []

    for t in range(n_steps):
        m = compute_flood_metrics(predicted_series[t], observed_series[t], depth_threshold)
        step_metrics.append(m.to_dict())

    avg_csi = float(np.mean([m["CSI"] for m in step_metrics]))
    avg_rmse = float(np.mean([m["RMSE_m"] for m in step_metrics]))

    logger.info("Spatiotemporal evaluation across %d steps: Mean CSI=%.4f, Mean RMSE=%.4f m",
                n_steps, avg_csi, avg_rmse)

    return {
        "mean_csi": avg_csi,
        "mean_rmse_m": avg_rmse,
        "step_metrics": step_metrics,
    }
