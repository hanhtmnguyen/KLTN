"""
Extract study area boundary by:
1. Filtering Sông Hồng (Red River) from the river network shapefile
2. Clipping it with the Hà Nội administrative boundary
3. Buffering the clipped river by 5 km
4. Saving the result as the study area boundary
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import geopandas as gpd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BOUND_DIR = Path(r"D:\KLTN\bound")
HANOI_SHP = BOUND_DIR / "Hà Nội (tỉnh thành) - 34.shp"
RIVER_SHP = BOUND_DIR / "Mangluoi_Thuyvan_diss2.shp"
OUTPUT_SHP = BOUND_DIR / "study_area_buffer_5km.shp"

BUFFER_KM = 5

# ── 1. Load data ──────────────────────────────────────────────────────
print("Loading Hà Nội boundary …")
hanoi = gpd.read_file(HANOI_SHP)

print("Loading river network …")
rivers = gpd.read_file(RIVER_SHP)

# ── 2. Extract Sông Hồng ─────────────────────────────────────────────
song_hong = rivers[rivers["name"] == "Sông Hồng"].copy()
print(f"Sông Hồng features: {len(song_hong)}")

if song_hong.empty:
    raise ValueError("No feature named 'Sông Hồng' found in the river shapefile!")

# ── 3. Clip Sông Hồng to Hà Nội boundary ─────────────────────────────
print("Clipping Sông Hồng to Hà Nội …")
song_hong_hanoi = gpd.clip(song_hong, hanoi)
print(f"Clipped features: {len(song_hong_hanoi)}")

# ── 4. Reproject to a metric CRS for accurate buffering ──────────────
#    UTM zone 48N (EPSG:32648) covers Hà Nội well.
METRIC_CRS = "EPSG:32648"
song_hong_proj = song_hong_hanoi.to_crs(METRIC_CRS)
hanoi_proj = hanoi.to_crs(METRIC_CRS)

# ── 5. Buffer by 5 km ────────────────────────────────────────────────
print(f"Buffering by {BUFFER_KM} km …")
buffer_m = BUFFER_KM * 1000
song_hong_proj["geometry"] = song_hong_proj.geometry.buffer(buffer_m)

# Dissolve into a single polygon (in case multiple segments)
study_area = song_hong_proj.dissolve()
study_area = study_area[["geometry"]].copy()
study_area["name"] = "Study Area – Sông Hồng 5 km buffer"

# ── 6. Reproject back to WGS 84 ──────────────────────────────────────
study_area = study_area.to_crs("EPSG:4326")

# ── 7. Save ───────────────────────────────────────────────────────────
study_area.to_file(OUTPUT_SHP, encoding="utf-8")
print(f"\n✓ Study area saved to: {OUTPUT_SHP}")
print(f"  CRS : {study_area.crs}")
print(f"  Bounds: {study_area.total_bounds}")
