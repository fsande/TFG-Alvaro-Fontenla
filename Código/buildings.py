import rasterio
import geopandas as gpd
import osmnx as ox
import rasterio.features
import numpy as np
from shapely.geometry import mapping
import matplotlib.pyplot as plt
from pyproj import Transformer

# 📌 1. Cargar imagen Sentinel-2
sentinel_path = "Sentinel2/TOC_V2/2025/03/05/S2B_20250305T115219_28RCS_TOC_V210/S2B_20250305T115219_28RCS_TOC-B04_10M_V210.tif"
with rasterio.open(sentinel_path) as src:
    sentinel_img = src.read(1)  # Leer primera banda
    sentinel_meta = src.meta
    bounds = src.bounds  # Extensión geográfica

# 📌 2. Convertir las coordenadas de Sentinel-2 a WGS84
transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
minx, miny = transformer.transform(bounds.left, bounds.bottom)
maxx, maxy = transformer.transform(bounds.right, bounds.top)
bbox = (minx, miny, maxx, maxy)  # (izquierda, abajo, derecha, arriba)

# 📌 3. Descargar edificios desde OSM (Evitar consultas grandes)
ox.settings.max_query_area_size = int(1e13)  # 100 millones de metros cuadrados

try:
    buildings = ox.features_from_bbox(maxy, miny, maxx, minx, tags={"building": True})

    if buildings.empty:
        raise ValueError("No se encontraron edificios en el área especificada.")
except Exception as e:
    print(f"❌ Error al descargar edificios: {e}")
    buildings = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

# 📌 4. Convertir a CRS de Sentinel-2 (si hay edificios)
if not buildings.empty:
    buildings = buildings.to_crs(src.crs)

# 📌 5. Convertir edificios a raster
building_mask = np.zeros(sentinel_img.shape, dtype=np.uint8)
if not buildings.empty:
    shapes = [(mapping(geom), 1) for geom in buildings.geometry if geom is not None]
    building_mask = rasterio.features.rasterize(shapes, out_shape=sentinel_img.shape,
                                                transform=src.transform, fill=0, default_value=255)

# 📌 6. Guardar la máscara de edificios
building_mask_path = "buildings_mask.tif"
building_meta = sentinel_meta.copy()
building_meta.update({"count": 1, "dtype": "uint8", "nodata": 0})

with rasterio.open(building_mask_path, "w", **building_meta) as dst:
    dst.write(building_mask, 1)

print("✅ Máscara de edificios guardada en 'buildings_mask.tif'")
