import matplotlib.pyplot as plt
import numpy as np
import rasterio

# Cargar la imagen Sentinel-2
with rasterio.open("Sentinel2/TOC_V2/2025/03/05/S2B_20250305T115219_28RCS_TOC_V210/S2B_20250305T115219_28RCS_TOC-B04_10M_V210.tif") as src:
    sentinel_img = src.read(1).astype(float)

# Cargar las máscaras
with rasterio.open("buildings_mask.tif") as mask_src:
    building_mask = mask_src.read(1)

with rasterio.open("roads_mask.tif") as mask_src:
    road_mask = mask_src.read(1)

# Asegurar que las máscaras sean binarias (0 o 1)
building_mask = (building_mask > 0).astype(np.uint8)  # Convertir a 0 o 1
road_mask = (road_mask > 0).astype(np.uint8)  # Convertir a 0 o 1

# Normalizar Sentinel-2 correctamente
sentinel_img = (sentinel_img - np.min(sentinel_img)) / (np.max(sentinel_img) - np.min(sentinel_img))
sentinel_img = np.uint8(255 * sentinel_img)

# Crear una imagen RGB con Sentinel-2 como base
rgb_img = np.stack([sentinel_img] * 3, axis=-1)  # Copiar en R, G y B

# Aplicar colores con transparencia
alpha_buildings = 0.5  # Transparencia edificios
alpha_roads = 0.5  # Transparencia carreteras

# Amarillo para edificios (rojo + verde)
rgb_img[..., 0] = np.where(building_mask == 1, (1 - alpha_buildings) * rgb_img[..., 0] + alpha_buildings * 255, rgb_img[..., 0])
rgb_img[..., 1] = np.where(building_mask == 1, (1 - alpha_buildings) * rgb_img[..., 1] + alpha_buildings * 255, rgb_img[..., 1])

# Azul para carreteras
rgb_img[..., 2] = np.where(road_mask == 1, (1 - alpha_roads) * rgb_img[..., 2] + alpha_roads * 255, rgb_img[..., 2])

# Mostrar la imagen con las máscaras superpuestas
plt.figure(figsize=(10, 8))
plt.imshow(rgb_img)
plt.title("Imagen Sentinel-2 con Máscaras de Edificios y Carreteras")
plt.axis("off")
plt.show()
