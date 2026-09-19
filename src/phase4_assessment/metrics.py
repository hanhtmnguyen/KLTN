"""
Accuracy metrics for flood model evaluation.

Computes CSI, F1-score, RMSE, POD, FAR, and computational speedup
for comparing PI-GAN output against HEC-RAS ground truth.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class FloodMetrics:
    """Container for flood model evaluation metrics."""
    csi: float          # Critical Success Index
    f1: float           # F1 score
    rmse: float         # Root Mean Square Error (depth)
    mae: float          # Mean Absolute Error (depth)
    pod: float          # Probability of Detection
    far: float          # False Alarm Ratio
    bias: float         # Frequency Bias
    n_tp: int           # True positives
    n_fp: int           # False positives
    n_fn: int           # False negatives
    n_tn: int           # True negatives

    def __str__(self) -> str:
        return (
            f"CSI={self.csi:.4f}  F1={self.f1:.4f}  RMSE={self.rmse:.4f}m  "
            f"MAE={self.mae:.4f}m  POD={self.pod:.4f}  FAR={self.far:.4f}"
        )

    def to_dict(self) -> dict:
        """Convert to dict for serialisation."""
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
    """Compute comprehensive flood accuracy metrics.

    Parameters
    ----------
    predicted : np.ndarray
        Predicted flood depth (H, W), in metres.
    observed : np.ndarray
        Ground-truth flood depth (H, W), in metres (from HEC-RAS).
    depth_threshold : float
        Depth threshold to classify wet/dry cells.

    Returns
    -------
    FloodMetrics
    """
    pred_wet = predicted > depth_threshold
    obs_wet = observed > depth_threshold

    tp = np.sum(pred_wet & obs_wet)
    fp = np.sum(pred_wet & ~obs_wet)
    fn = np.sum(~pred_wet & obs_wet)
    tn = np.sum(~pred_wet & ~obs_wet)

    eps = 1e-10

    # Critical Success Index
    csi = tp / (tp + fp + fn + eps)

    # F1 score
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    # Probability of Detection
    pod = tp / (tp + fn + eps)

    # False Alarm Ratio
    far = fp / (tp + fp + eps)

    # Frequency Bias
    bias = (tp + fp) / (tp + fn + eps)

    # Depth-based errors (wet cells only)
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


def compute_speedup(
    time_physical: float,
    time_pigan: float,
) -> float:
    """Compute computational speedup factor.

    Parameters
    ----------
    time_physical : float
        Wall-clock time for the physics model (HEC-RAS), in seconds.
    time_pigan : float
        Wall-clock time for HAND + PI-GAN, in seconds.

    Returns
    -------
    float
        Speedup factor (e.g., 100 means 100× faster).
    """
    if time_pigan <= 0:
        return float("inf")
    return time_physical / time_pigan
