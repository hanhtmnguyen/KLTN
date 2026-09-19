import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot_dem_comparison():
    xgboost_path = Path("da-framework/data/processed/dtm/xgboost_corrected_dtm.tif")
    fabdem_path = Path("da-framework/data/raw/fabdem/fabdem_32648_10m.tif")
    output_path = Path("da-framework/outputs/dem_comparison.png")

    with rasterio.open(fabdem_path) as src_fab:
        fab_data = src_fab.read(1)
        fab_nodata = src_fab.nodata if src_fab.nodata is not None else -9999
        fab_data = np.ma.masked_where(fab_data == fab_nodata, fab_data)
        
        # Open XGBoost DEM and align to FabDEM using VRT
        with rasterio.open(xgboost_path) as src_xgb:
            vrt_options = {
                'resampling': Resampling.bilinear,
                'crs': src_fab.crs,
                'transform': src_fab.transform,
                'height': src_fab.height,
                'width': src_fab.width,
            }
            with WarpedVRT(src_xgb, **vrt_options) as vrt:
                xgb_data = vrt.read(1)
                xgb_nodata = vrt.nodata if vrt.nodata is not None else -9999
                xgb_data = np.ma.masked_where(xgb_data == xgb_nodata, xgb_data)
        
        # Calculate diff
        diff_data = xgb_data - fab_data

        # Determine common scale (vmin, vmax) for better visual comparison
        vmin = max(np.nanpercentile(fab_data.compressed(), 1), 0)
        vmax = np.nanpercentile(fab_data.compressed(), 99)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. FabDEM
        im1 = axes[0].imshow(fab_data, cmap='terrain', vmin=vmin, vmax=vmax)
        axes[0].set_title('Original FabDEM (10m)')
        axes[0].axis('off')
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='Elevation (m)')

        # 2. XGBoost DEM
        im2 = axes[1].imshow(xgb_data, cmap='terrain', vmin=vmin, vmax=vmax)
        axes[1].set_title('XGBoost Corrected DTM (10m)')
        axes[1].axis('off')
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Elevation (m)')

        # 3. Difference
        diff_vmax = max(abs(np.nanpercentile(diff_data.compressed(), 1)), 
                        abs(np.nanpercentile(diff_data.compressed(), 99)))
        im3 = axes[2].imshow(diff_data, cmap='RdBu', vmin=-diff_vmax, vmax=diff_vmax)
        axes[2].set_title('Difference (XGBoost - FabDEM)')
        axes[2].axis('off')
        fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label='Elevation Difference (m)')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to {output_path}")

if __name__ == "__main__":
    plot_dem_comparison()
