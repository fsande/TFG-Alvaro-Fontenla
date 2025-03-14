import os
import h5py
import rasterio
import numpy as np

# Directorio donde están las imágenes TIF
ruta_tifs = os.path.join(os.path.dirname(__file__), "Sentinel2", "TOC_V2", "2025", "03", "05", "S2B_20250305T115219_28RCS_TOC_V210")
fichero_hdf5 = "sentinel2_dataset.h5"

# Crear el fichero HDF5
with h5py.File(fichero_hdf5, "w") as hdf:
    for fichero in os.listdir(ruta_tifs):
        if fichero.endswith(".tif"):  # Solo ficheros TIF
            ruta_completa = os.path.join(ruta_tifs, fichero)
            
            # Leer la imagen TIF
            with rasterio.open(ruta_completa) as src:
                img_data = src.read(1)  # Leer la banda
                metadata = src.meta  # Guardar metadatos
            
            # Guardar en HDF5 con el nombre del fichero como dataset
            dataset = hdf.create_dataset(fichero, data=img_data, compression="gzip")
            
            # Guardar metadatos en atributos del dataset
            dataset.attrs["crs"] = str(metadata["crs"])  # Sistema de coordenadas
            dataset.attrs["transform"] = str(metadata["transform"])  # Matriz de transformación
            dataset.attrs["width"] = metadata["width"]  # Ancho de la imagen
            dataset.attrs["height"] = metadata["height"]  # Alto de la imagen
            dataset.attrs["count"] = metadata["count"]  # Número de bandas
            
            print(f"Guardado: {fichero} en {fichero_hdf5}")

print("Conversión completada")
