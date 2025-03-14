import h5py
import numpy as np
import matplotlib.pyplot as plt

# Función para recortar y normalizar las imágenes
def recortar_y_normalizar(imagen, max_val=10000):
    # Eliminar valores negativos
    imagen = np.clip(imagen, 0, None)  # Todos los valores negativos se convierten en 0
    
    # Recortar los valores demasiado altos (en base al valor máximo esperado, por ejemplo 10000)
    imagen = np.clip(imagen, 0, max_val)
    
    # Normalizar la imagen al rango [0, 255]
    imagen_normalizada = (imagen / max_val) * 255
    return np.clip(imagen_normalizada, 0, 255)

# Abrir el archivo HDF5
with h5py.File("sentinel2_dataset.h5", "r") as hdf:
    # Listar datasets disponibles
    print("Datasets disponibles:", list(hdf.keys()))

    # Definir los nombres de las bandas que vamos a usar para la composición RGB
    bandas_rgb = ["S2B_20250305T115219_28RCS_TOC-B04_10M_V210.tif",  # Banda Roja (Red)
                  "S2B_20250305T115219_28RCS_TOC-B03_10M_V210.tif",  # Banda Verde (Green)
                  "S2B_20250305T115219_28RCS_TOC-B02_10M_V210.tif"]  # Banda Azul (Blue)

    # Leer las bandas seleccionadas y almacenarlas
    img_roja = hdf[bandas_rgb[0]][()]
    img_verde = hdf[bandas_rgb[1]][()]
    img_azul = hdf[bandas_rgb[2]][()]

    # Comprobar las estadísticas de las bandas antes de la normalización
    print("\nEstadísticas antes de normalizar:")
    print("Banda roja (B04): Mínimo =", np.min(img_roja), "Máximo =", np.max(img_roja))
    print("Banda verde (B03): Mínimo =", np.min(img_verde), "Máximo =", np.max(img_verde))
    print("Banda azul (B02): Mínimo =", np.min(img_azul), "Máximo =", np.max(img_azul))

    # Normalizar las bandas con el rango recortado
    img_roja = recortar_y_normalizar(img_roja, max_val=20000)  # Ajustar max_val si es necesario
    img_verde = recortar_y_normalizar(img_verde, max_val=20000)
    img_azul = recortar_y_normalizar(img_azul, max_val=20000)

    # Crear la imagen RGB combinando las tres bandas
    img_rgb = np.stack((img_roja, img_verde, img_azul), axis=-1)

    # Mostrar la imagen RGB
    plt.imshow(img_rgb.astype(np.uint8))  # Convertir a uint8 para asegurar que los valores son de 0 a 255
    plt.title("Composición RGB de Sentinel-2")
    plt.axis("off")  # Desactivar los ejes
    plt.show()
