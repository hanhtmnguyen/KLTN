import sqlite3

db_path = r'D:\KLTN\da-framework\data\training\training_data.gpkg'
conn = sqlite3.connect(db_path)
cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall() if not row[0].startswith('gpkg_') and not row[0].startswith('rtree_') and not row[0].startswith('sqlite_')]
print('Tables:', tables)

for t in tables:
    cursor = conn.execute(f"PRAGMA table_info({t})")
    cols = [c[1] for c in cursor.fetchall()]
    print(f'Columns in {t}: {cols}')
