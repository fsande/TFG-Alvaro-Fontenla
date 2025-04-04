import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Rutas de las bandas RGB
banda_roja = "Sentinel2/TOC_V2/2025/03/05/S2B_20250305T115219_28RCS_TOC_V210/S2B_20250305T115219_28RCS_TOC-B04_10M_V210.tif"
banda_verde = "Sentinel2/TOC_V2/2025/03/05/S2B_20250305T115219_28RCS_TOC_V210/S2B_20250305T115219_28RCS_TOC-B03_10M_V210.tif"
banda_azul = "Sentinel2/TOC_V2/2025/03/05/S2B_20250305T115219_28RCS_TOC_V210/S2B_20250305T115219_28RCS_TOC-B02_10M_V210.tif"

# Cargar las bandas
with rasterio.open(banda_roja) as src:
    banda_roja_img = src.read(1)  # Leer la banda 4 (rojo)
    
with rasterio.open(banda_verde) as src:
    banda_verde_img = src.read(1)  # Leer la banda 3 (verde)
    
with rasterio.open(banda_azul) as src:
    banda_azul_img = src.read(1)  # Leer la banda 2 (azul)

# Normalizar las bandas a un rango de 0 a 255
def normalize_banda(banda):
    min_val = np.min(banda)
    max_val = np.max(banda)
    return ((banda - min_val) / (max_val - min_val)) * 255

# Normalizar las tres bandas
banda_roja_norm = normalize_banda(banda_roja_img)
banda_verde_norm = normalize_banda(banda_verde_img)
banda_azul_norm = normalize_banda(banda_azul_img)

# Crear una imagen RGB combinando las tres bandas normalizadas
rgb_img = np.dstack((banda_roja_norm, banda_verde_norm, banda_azul_norm))

# Visualizar la imagen RGB
plt.figure(figsize=(10, 8))
plt.imshow(rgb_img.astype(np.uint8))  # Convertir a enteros de 8 bits (0-255)
plt.title("Imagen RGB de Sentinel-2 Normalizada")
plt.axis('off')  # Opcional, para ocultar los ejes
plt.show()