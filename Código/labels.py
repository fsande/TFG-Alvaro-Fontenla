import rasterio
import numpy as np
import os

# Cargar el archivo de clasificación
with rasterio.open("Sentinel2/TOC_V2/2025/03/05/S2B_20250305T115219_28RCS_TOC_V210/S2B_20250305T115219_28RCS_SCENECLASSIFICATION_20M_V210.tif") as src:
    img = src.read(1)  # Leer la banda de clasificación

# Obtener valores únicos presentes en la imagen
unique_classes = np.unique(img)
print("Valores de clase en la imagen:", unique_classes)
