import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

db_path = r'D:\KLTN\da-framework\data\training\training_data.gpkg'
conn = sqlite3.connect(db_path)

# Select a specific track to show a cross-section.
# Track 0903 has about 30k points. Let's take a consecutive segment of 3000 points.
track_id = '0903'
query = f"""
    SELECT lat, lon, elevation, fabdem_elev 
    FROM training_data 
    WHERE track_id = '{track_id}' 
      AND elevation IS NOT NULL 
      AND fabdem_elev IS NOT NULL
    ORDER BY lat DESC
    LIMIT 3000 OFFSET 5000
"""
df = pd.read_sql_query(query, conn)
conn.close()

# Calculate approximate distance along track from the first point in the segment
# 1 degree of latitude is approx 111 km. 
df['dist_km'] = np.sqrt(((df['lat'] - df['lat'].iloc[0]) * 111)**2 + 
                        ((df['lon'] - df['lon'].iloc[0]) * 111 * np.cos(np.radians(df['lat'].iloc[0])))**2)

plt.figure(figsize=(15, 6))

# Plot lines for cross section
plt.plot(df['dist_km'], df['elevation'], label='ICESat-2 Elevation', color='blue', linewidth=1.5, alpha=0.8)
plt.plot(df['dist_km'], df['fabdem_elev'], label='FabDEM Elevation', color='darkorange', linewidth=1.5, alpha=0.8, linestyle='--')

plt.title(f'Elevation Cross-Section Profile (Track {track_id} Segment)')
plt.xlabel('Distance along track (km)')
plt.ylabel('Elevation (m)')
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()

out_path = r'D:\KLTN\da-framework\data\training\icesat2_vs_dem_cross_section.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {out_path}")
