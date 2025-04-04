import matplotlib.pyplot as plt
import rasterio

# Cargar la imagen Sentinel-2
with rasterio.open("Sentinel2/TOC_V2/2025/03/05/S2B_20250305T115219_28RCS_TOC_V210/S2B_20250305T115219_28RCS_TOC-B04_10M_V210.tif") as src:
    sentinel_img = src.read(1)

# Cargar la máscara de carreteras
with rasterio.open("roads_mask.tif") as mask_src:
    road_mask = mask_src.read(1)

# Visualizar Sentinel-2 + carreteras
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Imagen Sentinel-2")
plt.imshow(sentinel_img, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Máscara de Carreteras")
plt.imshow(road_mask, cmap="gray")

plt.show()
