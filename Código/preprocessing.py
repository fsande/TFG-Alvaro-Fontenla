import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# 📌 Cargar las bandas de la imagen Sentinel-2 (el archivo .tif debe contener varias bandas)
def load_sentinel_image(sentinel_path, bands=[1]):  # Solo Banda 1 (ajustar según la imagen)
    with rasterio.open(sentinel_path) as src:
        # Leer la banda seleccionada
        img_bands = [src.read(band) for band in bands]
        
        # Convertir de (bandas, alto, ancho) a (alto, ancho, bandas)
        sentinel_img = np.moveaxis(np.array(img_bands), 0, -1)
        
        # Normalizar la banda a [0, 1]
        sentinel_img = sentinel_img / 10000.0  # Normalizar a rango [0,1] si es necesario
        return sentinel_img


# 📌 Cargar la máscara fusionada de carreteras y edificios
def load_mask(road_mask_path, building_mask_path):
    with rasterio.open(road_mask_path) as road_src:
        road_mask = road_src.read(1)  # Solo una capa de la máscara de carreteras
    
    with rasterio.open(building_mask_path) as building_src:
        building_mask = building_src.read(1)  # Solo una capa de la máscara de edificios
    
    # Fusionar las máscaras: usar valores diferentes para cada tipo de carretera
    fused_mask = np.zeros_like(road_mask)

    # Asignar valores para los diferentes tipos de carreteras
    fused_mask[road_mask == 1] = 1  # Motorway
    fused_mask[road_mask == 2] = 2  # Trunk
    fused_mask[road_mask == 3] = 3  # Primary
    fused_mask[road_mask == 4] = 4  # Secondary
    fused_mask[road_mask == 6] = 5  # Residential

    # Asignar valor para edificios
    fused_mask[building_mask == 1] = 6  # Edificio

    return fused_mask


# 📌 Procesar los datos (sin dividir en entrenamiento y validación)
def preprocess_data(image_paths, road_mask_paths, building_mask_paths):
    images = []
    masks = []
    
    for sentinel_path, road_mask_path, building_mask_path in zip(image_paths, road_mask_paths, building_mask_paths):
        sentinel_img = load_sentinel_image(sentinel_path)
        fused_mask = load_mask(road_mask_path, building_mask_path)
        images.append(sentinel_img)
        masks.append(fused_mask)
    
    # Convertir listas a arrays numpy
    images = np.array(images)
    masks = np.array(masks)
    
    return images, masks


# 📌 Rutas de las imágenes y máscaras
image_paths = [
    "Sentinel2/TOC_V2/2025/03/05/S2B_20250305T115219_28RCS_TOC_V210/S2B_20250305T115219_28RCS_TOC-B04_10M_V210.tif"
]  # Añadir todas las imágenes de Sentinel-2

road_mask_paths = [
    "roads_multiclass_mask.tif"
]  # Añadir las máscaras de carreteras

building_mask_paths = [
    "buildings_mask.tif"
]  # Añadir las máscaras de edificios

# Crear los datasets (sin dividir)
X, y = preprocess_data(image_paths, road_mask_paths, building_mask_paths)

# 📌 Definir clases y colores para la visualización de la máscara
class_labels = {
    0: "Fondo",            # 0 - Fondo
    1: "Motorway",         # 1 - Motorway
    2: "Trunk",            # 2 - Trunk
    3: "Primary",          # 3 - Primary
    4: "Secondary",        # 4 - Secondary
    5: "Residential",      # 5 - Residential
    6: "Edificio"          # 6 - Edificio
}

colors = [
    "#000000",  # 0 - Fondo
    "#ff0000",  # 1 - Motorway (Carretera principal)
    "#ffa500",  # 2 - Trunk (Carretera secundaria)
    "#ffff00",  # 3 - Primary (Carretera primaria)
    "#00ff00",  # 4 - Secondary (Carretera secundaria)
    "#00ffff",  # 5 - Residential (Residencial)
    "#0000ff",  # 6 - Edificio
]

# Asegurarse que el mapa tenga los índices correctos
cmap = ListedColormap(colors)
bounds = [0, 1, 2, 3, 4, 5, 6, 7]  # Para tener los límites de las clases
norm = BoundaryNorm(bounds, cmap.N)

# 📌 Visualización de una imagen y su máscara de ejemplo
plt.figure(figsize=(12, 6))

# Imagen de Sentinel-2
plt.subplot(1, 2, 1)
plt.title("Imagen de Sentinel-2")
plt.imshow(X[0])  # Mostrar la primera imagen procesada

# Máscara fusionada (con colores definidos para las clases)
plt.subplot(1, 2, 2)
plt.title("Máscara Fusionada")
im = plt.imshow(y[0], cmap=cmap, norm=norm)  # Mostrar la máscara fusionada con colores
cbar = plt.colorbar(im, ticks=list(class_labels.keys()))
cbar.ax.set_yticklabels([class_labels[k] for k in class_labels])  # Asignar etiquetas de clase
plt.tight_layout()
plt.show()