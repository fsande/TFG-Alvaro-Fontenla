import h5py
import numpy as np
import matplotlib.pyplot as plt

# Abrir el fichero HDF5
with h5py.File("sentinel2_dataset.h5", "r") as hdf:
    dataset_name = "S2B_20250305T115219_28RCS_TOC-B04_10M_V210.tif"  # Banda roja
    img = hdf[dataset_name][()]  # Leer la imagen

# Normalizar la imagen al rango 0-255
img = np.clip(img, 0, 20000)  # Recortar valores fuera de rango
img = (img / 20000) * 255  # Escalar a 0-255
img = img.astype(np.uint8)  # Convertir a uint8 para visualización

# Crear imagen RGB (inicialmente en escala de grises)
img_rgb = np.stack((img, img, img), axis=-1)  # Convertir a RGB

# Definir el umbral de valores bajos
umbral_bajo = 0     # Mínimo valor a considerar como "negro"
umbral_alto = 50    # Máximo valor que aún será afectado por el verde

# Máscara de valores bajos (donde img está entre umbral_bajo y umbral_alto)
mascara_baja = (img >= umbral_bajo) & (img <= umbral_alto)

# Generar un gradiente de verde para los valores bajos
verde_min = np.array([0, 50, 0], dtype=np.uint8)   # Verde oscuro
verde_max = np.array([0, 255, 0], dtype=np.uint8)  # Verde claro

# Crear un mapa de colores progresivo
factor_verde = (img[mascara_baja] - umbral_bajo) / (umbral_alto - umbral_bajo)  # Normalizar 0-1
verde_rango = (verde_min + (verde_max - verde_min) * factor_verde[:, None]).astype(np.uint8)

# Aplicar el gradiente de verde a los valores bajos
img_rgb[mascara_baja] = verde_rango

# Mostrar la imagen modificada
plt.imshow(img_rgb)
plt.title("Imagen con valores bajos en rango de verde")
plt.axis("off")  # Ocultar ejes
plt.show()