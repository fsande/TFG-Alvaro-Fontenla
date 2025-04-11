import rasterio
import geopandas as gpd
import osmnx as ox
import rasterio.features
import numpy as np
from shapely.geometry import mapping
from pyproj import Transformer

# 📌 1. Cargar imagen Sentinel-2
sentinel_path = "Sentinel2/TOC_V2/2025/03/05/S2B_20250305T115219_28RCS_TOC_V210/S2B_20250305T115219_28RCS_TOC-B04_10M_V210.tif"
with rasterio.open(sentinel_path) as src:
    sentinel_img = src.read(1)
    sentinel_meta = src.meta
    bounds = src.bounds

# 📌 2. Convertir coordenadas a WGS84
transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
minx, miny = transformer.transform(bounds.left, bounds.bottom)
maxx, maxy = transformer.transform(bounds.right, bounds.top)

# 📌 3. Descargar todas las carreteras desde OSM
ox.settings.max_query_area_size = int(1e13)
try:
    roads_graph = ox.graph_from_bbox(maxy, miny, maxx, minx, network_type="all")
    roads = ox.graph_to_gdfs(roads_graph, nodes=False)
    if roads.empty:
        raise ValueError("No se encontraron carreteras en el área.")
except Exception as e:
    print(f"❌ Error al descargar carreteras: {e}")
    roads = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

# 📌 4. Convertir CRS a imagen Sentinel-2
if not roads.empty:
    roads = roads.to_crs(src.crs)

# 📌 5. Definir los tipos de carretera y su valor en la máscara
road_class_map = {
    "motorway": 1,
    "trunk": 2,
    "primary": 3,
    "secondary": 4,
    "residential": 6,
}

# 📌 6. Crear máscara por tipo de carretera
road_mask = np.zeros(sentinel_img.shape, dtype=np.uint8)  # Crear una máscara vacía
if not roads.empty:
    roads["highway"] = roads["highway"].fillna("other")  # Rellenar valores nulos

    for road_type, value in road_class_map.items():
        filtered = roads[roads["highway"] == road_type]
        if filtered.empty:
            continue

        shapes = [(mapping(geom), value) for geom in filtered.geometry if geom is not None]
        mask_layer = rasterio.features.rasterize(
            shapes,
            out_shape=sentinel_img.shape,
            transform=src.transform,
            fill=0,  # Rellenar con 0, no queremos valores no deseados en otras áreas
            default_value=value,  # Asignar el valor correcto para cada tipo de carretera
            dtype=np.uint8
        )
        road_mask = np.maximum(road_mask, mask_layer)  # Aseguramos que no sobrescribimos las clases existentes

# 📌 7. Guardar máscara multiclase
road_mask_path = "roads_multiclass_mask.tif"
road_meta = sentinel_meta.copy()
road_meta.update({"count": 1, "dtype": "uint8", "nodata": 0})  # Definir nodata como 0

with rasterio.open(road_mask_path, "w", **road_meta) as dst:
    dst.write(road_mask, 1)

print("✅ Máscara multiclase de carreteras guardada en 'roads_multiclass_mask.tif'")
