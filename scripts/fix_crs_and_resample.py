import sys
import geopandas as gpd
from pyogrio import write_dataframe

# Add da-framework to path to import utils
sys.path.append(r'D:\KLTN\da-framework')

from src.utils.geo_utils import reproject_raster, sample_raster_at_points

def main():
    roi_path = r'D:\KLTN\bound\study_area_buffer_5km.shp'
    fabdem_raw = r'D:\KLTN\da-framework\data\raw\fabdem\fabdem_raw.tif'
    fabdem_32648 = r'D:\KLTN\da-framework\data\raw\fabdem\fabdem_32648.tif'
    training_data_path = r'D:\KLTN\da-framework\data\training\training_data.gpkg'
    training_data_fixed_path = r'D:\KLTN\da-framework\data\training\training_data_fixed.gpkg'
    
    print("Loading ROI and converting to EPSG:32648...")
    roi = gpd.read_file(roi_path)
    roi_32648 = roi.to_crs("EPSG:32648")
    
    print("Loading existing training data...")
    df = gpd.read_file(training_data_path, engine='pyogrio')
    
    print(f"Total points before clipping: {len(df)}")
    df_clipped = gpd.clip(df, roi_32648)
    print(f"Total points after clipping: {len(df_clipped)}")
    
    if len(df_clipped) == 0:
        print("ERROR: No points found inside the ROI!")
        return
        
    print("Reprojecting FabDEM to EPSG:32648...")
    reproject_raster(
        src_path=fabdem_raw,
        dst_path=fabdem_32648,
        dst_crs="EPSG:32648"
    )
    
    print("Sampling new FabDEM elevations...")
    df_clipped = sample_raster_at_points(
        raster_path=fabdem_32648,
        points=df_clipped,
        column_name='fabdem_elev_new'
    )
    
    print("Updating values...")
    df_clipped['fabdem_elev'] = df_clipped['fabdem_elev_new']
    df_clipped = df_clipped.drop(columns=['fabdem_elev_new'])

    print(f"Saving fixed dataset to {training_data_fixed_path}...")
    write_dataframe(df_clipped, training_data_fixed_path, driver="GPKG")
    print("Done!")

if __name__ == '__main__':
    main()
