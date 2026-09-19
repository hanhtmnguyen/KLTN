"""
Visualization utilities for flood maps, terrain, and model results.

Provides publication-quality matplotlib/folium maps used throughout
Phases 1-4.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from pathlib import Path
from typing import Sequence
import rasterio
from rasterio.plot import show as rio_show

from src.utils.logging_config import get_logger

logger = get_logger("da.viz")


# ---------------------------------------------------------------------------
# Colour maps for flood visualization
# ---------------------------------------------------------------------------

# Flood depth colour map: blue gradient from light to dark
FLOOD_DEPTH_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "flood_depth",
    ["#ffffff", "#cce5ff", "#66b3ff", "#3399ff", "#0066cc", "#003366"],
)

# Risk level colour map: green → yellow → orange → red
RISK_CMAP = mcolors.ListedColormap(["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"])
RISK_BOUNDS = [0, 0.3, 1.0, 2.0, 10.0]
RISK_NORM = mcolors.BoundaryNorm(RISK_BOUNDS, RISK_CMAP.N)

# DTM terrain colour map
TERRAIN_CMAP = "terrain"


def plot_raster(
    data: np.ndarray,
    title: str = "",
    cmap: str | mcolors.Colormap = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str = "",
    figsize: tuple[int, int] = (10, 8),
    save_path: str | Path | None = None,
) -> Figure:
    """Plot a 2-D raster array with colourbar."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved figure → %s", save_path)

    return fig


def plot_flood_depth(
    depth: np.ndarray,
    title: str = "Flood Depth",
    vmax: float = 5.0,
    figsize: tuple[int, int] = (10, 8),
    save_path: str | Path | None = None,
) -> Figure:
    """Plot a flood depth map with the custom blue gradient colour map."""
    masked = np.ma.masked_where(depth <= 0.01, depth)
    return plot_raster(
        masked,
        title=title,
        cmap=FLOOD_DEPTH_CMAP,
        vmin=0,
        vmax=vmax,
        colorbar_label="Depth (m)",
        figsize=figsize,
        save_path=save_path,
    )


def plot_risk_map(
    depth: np.ndarray,
    title: str = "Flood Risk Classification",
    figsize: tuple[int, int] = (10, 8),
    save_path: str | Path | None = None,
) -> Figure:
    """Plot flood risk classification (Low/Moderate/High/Very High)."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(depth, cmap=RISK_CMAP, norm=RISK_NORM, origin="upper")
    ax.set_title(title, fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        ticks=[0.15, 0.65, 1.5, 3.0])
    cbar.ax.set_yticklabels(["Low\n(<0.3m)", "Moderate\n(0.3-1.0m)",
                              "High\n(1.0-2.0m)", "Very High\n(>2.0m)"])
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_comparison(
    data_left: np.ndarray,
    data_right: np.ndarray,
    title_left: str = "HAND (Input)",
    title_right: str = "PI-GAN (Output)",
    cmap: str | mcolors.Colormap = "viridis",
    vmin: float = 0,
    vmax: float = 5.0,
    colorbar_label: str = "Depth (m)",
    figsize: tuple[int, int] = (16, 7),
    save_path: str | Path | None = None,
) -> Figure:
    """Side-by-side comparison of two rasters."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    left_masked = np.ma.masked_where(data_left <= 0.01, data_left)
    right_masked = np.ma.masked_where(data_right <= 0.01, data_right)

    im1 = ax1.imshow(left_masked, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax1.set_title(title_left, fontsize=13, fontweight="bold")

    im2 = ax2.imshow(right_masked, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax2.set_title(title_right, fontsize=13, fontweight="bold")

    fig.colorbar(im2, ax=[ax1, ax2], fraction=0.02, pad=0.04, label=colorbar_label)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_curves(
    losses: dict[str, list[float]],
    title: str = "Training Loss Curves",
    figsize: tuple[int, int] = (12, 6),
    save_path: str | Path | None = None,
) -> Figure:
    """Plot training loss curves (one line per loss component)."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for name, values in losses.items():
        ax.plot(values, label=name, linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
