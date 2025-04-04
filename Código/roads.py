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


# 📌 3. Descargar carreteras desde OSM (Evitar consultas grandes)
ox.settings.max_query_area_size = int(1e13)  # 100 millones de metros cuadrados

road_types = ["motorway", "trunk", "primary", "secondary", "tertiary"]

try:
    roads_graph = ox.graph_from_bbox(maxy, miny, maxx, minx, network_type="all")
    roads = ox.graph_to_gdfs(roads_graph, nodes=False)

    if roads.empty:
        raise ValueError("No se encontraron carreteras en el área especificada.")
except Exception as e:
    print(f"❌ Error al descargar carreteras: {e}")
    roads = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

# 📌 4. Convertir a CRS de Sentinel-2 (si hay carreteras)
if not roads.empty:
    roads = roads.to_crs(src.crs)

# 📌 5. Convertir carreteras a raster
road_mask = np.zeros(sentinel_img.shape, dtype=np.uint8)
if not roads.empty:
    shapes = [(mapping(geom), 1) for geom in roads.geometry if geom is not None]
    road_mask = rasterio.features.rasterize(shapes, out_shape=sentinel_img.shape,
                                            transform=src.transform, fill=0, default_value=255)

# 📌 6. Guardar la máscara de carreteras
road_mask_path = "roads_mask.tif"
road_meta = sentinel_meta.copy()
road_meta.update({"count": 1, "dtype": "uint8", "nodata": 0})

with rasterio.open(road_mask_path, "w", **road_meta) as dst:
    dst.write(road_mask, 1)

print("✅ Máscara de carreteras guardada en 'roads_mask.tif'")