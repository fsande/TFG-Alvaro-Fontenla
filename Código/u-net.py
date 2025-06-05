import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from pathlib import Path
import random

# Configuración de TensorFlow para una mejor utilización de memoria
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error al configurar GPU: {e}")

# Configuración general
BATCH_SIZE = 8
EPOCHS = 100    # Aumentar para mejor entrenamiento
INPUT_SIZE = (128, 128) 
NUM_CLASSES = 3  # Fondo (0), Edificios (1), Carreteras (2)
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2

# Configuración de la subregión (ROI)
# Definir coordenadas de la región de interés (en píxeles)
ROI_START_X = 1000  # Columna inicial
ROI_START_Y = 1000  # Fila inicial
ROI_WIDTH = 2000    # Ancho de la región en píxeles
ROI_HEIGHT = 2000   # Alto de la región en píxeles

BAND_FILES = {
    'B02': 'S2B_20250305T115219_28RCS_TOC-B02_10M_V210.tif',  # Azul (10m)
    'B03': 'S2B_20250305T115219_28RCS_TOC-B03_10M_V210.tif',  # Verde (10m)
    'B04': 'S2B_20250305T115219_28RCS_TOC-B04_10M_V210.tif',  # Rojo (10m)
    'B08': 'S2B_20250305T115219_28RCS_TOC-B08_10M_V210.tif',  # NIR (10m)
    'B05': 'S2B_20250305T115219_28RCS_TOC-B05_20M_V210.tif',  # Red Edge 1 (20m)
    'B06': 'S2B_20250305T115219_28RCS_TOC-B06_20M_V210.tif',  # Red Edge 2 (20m)
    'B07': 'S2B_20250305T115219_28RCS_TOC-B07_20M_V210.tif',  # Red Edge 3 (20m)
    'B8A': 'S2B_20250305T115219_28RCS_TOC-B8A_20M_V210.tif',  # Red Edge 4 (20m)
    'B11': 'S2B_20250305T115219_28RCS_TOC-B11_20M_V210.tif',  # SWIR 1 (20m)
    'B12': 'S2B_20250305T115219_28RCS_TOC-B12_20M_V210.tif',  # SWIR 2 (20m)
}

# Rutas de archivos
SENTINEL_DIR = "Sentinel2/TOC_V2/2025/03/05/S2B_20250305T115219_28RCS_TOC_V210"
BUILDINGS_MASK = "masks/buildings_mask.tif"
ROADS_MASK = "masks/roads_mask.tif"
OUTPUT_MODEL = "masks/unet_sentinel_model.h5"
OUTPUT_DATASET_DIR = "masks/dataset"
OUTPUT_MULTICLASS_MASK = "masks/fused_multiclass_mask.tif"
OUTPUT_ROI_DIR = "masks/roi"  # Directorio para guardar la subregión

# Obtener las dimensiones de la imagen original
def get_image_dimensions(file_path):
    with rasterio.open(file_path) as src:
        return src.height, src.width

# Función para obtener el ROI centrado
def get_centered_roi(height, width, roi_height=2000, roi_width=2000):
    # Calcular el punto central
    center_y = height // 2
    center_x = width // 2
    
    # Calcular las coordenadas del ROI centrado
    roi_start_y = center_y - (roi_height // 2)
    roi_start_x = center_x - (roi_width // 2)
    
    roi_start_y = max(0, min(roi_start_y, height - roi_height))
    roi_start_x = max(0, min(roi_start_x, width - roi_width))
    
    return roi_start_x, roi_start_y, roi_width, roi_height

reference_file = os.path.join(SENTINEL_DIR, BAND_FILES['B02'])
img_height, img_width = get_image_dimensions(reference_file)
ROI_START_X, ROI_START_Y, ROI_WIDTH, ROI_HEIGHT = get_centered_roi(img_height, img_width)

print(f"Dimensiones de la imagen original: {img_width}x{img_height}")
print(f"ROI configurado: x={ROI_START_X}, y={ROI_START_Y}, ancho={ROI_WIDTH}, alto={ROI_HEIGHT}")

def visualize_masks():
    """
    Visualiza las máscaras para verificar que contienen datos
    """
    print("Verificando máscaras...")
    
    try:
        with rasterio.open(BUILDINGS_MASK) as src:
            buildings = src.read(1)
            buildings_count = np.sum(buildings > 0)
            print(f"Máscara de edificios: {buildings.shape}, Píxeles positivos: {buildings_count} ({buildings_count/buildings.size*100:.4f}%)")
        
        with rasterio.open(ROADS_MASK) as src:
            roads = src.read(1)
            roads_count = np.sum(roads > 0)
            print(f"Máscara de carreteras: {roads.shape}, Píxeles positivos: {roads_count} ({roads_count/roads.size*100:.4f}%)")
    
    except Exception as e:
        print(f"Error al verificar máscaras: {e}")

def extract_roi_from_image(image_path, roi_window, output_path=None):
    """
    Extrae una región de interés (ROI) de una imagen y la guarda opcionalmente
    
    Args:
        image_path: Ruta al archivo de imagen
        roi_window: Objeto Window de rasterio que define la ROI
        output_path: Ruta donde guardar la ROI (opcional)
    
    Returns:
        data: Array NumPy con los datos de la ROI
        transform: Transformación geoespacial actualizada
    """
    with rasterio.open(image_path) as src:
        # Leer ROI
        data = src.read(window=roi_window)
        
        # Actualizar transformación para la ROI
        transform = rasterio.windows.transform(roi_window, src.transform)
        
        # Si se proporciona una ruta de salida, guardar la ROI
        if output_path:
            # Actualizar metadatos
            meta = src.meta.copy()
            meta.update({
                'height': roi_window.height,
                'width': roi_window.width,
                'transform': transform
            })
            
            # Guardar ROI
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(data)
            
            print(f"ROI guardada en {output_path}")
        
        return data, transform

def create_multiclass_mask_roi():
    """
    Fusiona las máscaras de edificios y carreteras en una única máscara multiclase,
    extrayendo solo la región de interés
    """
    print("Creando máscara multiclase para la ROI...")
    
    # Crear directorio para la ROI si no existe
    os.makedirs(OUTPUT_ROI_DIR, exist_ok=True)
    
    # Definir ventana para la ROI
    roi_window = Window(ROI_START_X, ROI_START_Y, ROI_WIDTH, ROI_HEIGHT)
    
    # Cargar la máscara de edificios (solo la ROI)
    with rasterio.open(BUILDINGS_MASK) as src:
        buildings, transform = extract_roi_from_image(
            BUILDINGS_MASK, 
            roi_window, 
            os.path.join(OUTPUT_ROI_DIR, "buildings_roi.tif")
        )
        buildings = buildings[0]  # Obtener primera banda
        meta = src.meta.copy()
    
    # Cargar la máscara de carreteras (solo la ROI)
    with rasterio.open(ROADS_MASK) as src:
        roads, _ = extract_roi_from_image(
            ROADS_MASK, 
            roi_window, 
            os.path.join(OUTPUT_ROI_DIR, "roads_roi.tif")
        )
        roads = roads[0]  # Obtener primera banda
    
    # Asegurarse de que las máscaras son binarias
    buildings = (buildings > 0).astype(np.uint8)
    roads = (roads > 0).astype(np.uint8)
    
    # Crear máscara multiclase
    # Prioridad: 1. Carreteras, 2. Edificios, 0. Fondo
    multiclass_mask = np.zeros_like(buildings, dtype=np.uint8)
    multiclass_mask[buildings > 0] = 1  # Edificios
    multiclass_mask[roads > 0] = 2      # Carreteras (sobrescribe edificios en intersecciones)
    
    # Guardar la máscara multiclase
    roi_multiclass_mask_path = os.path.join(OUTPUT_ROI_DIR, "multiclass_mask_roi.tif")
    meta.update({
        'dtype': rasterio.uint8,
        'count': 1,
        'nodata': 255,
        'height': ROI_HEIGHT,
        'width': ROI_WIDTH,
        'transform': transform
    })
    
    with rasterio.open(roi_multiclass_mask_path, 'w', **meta) as dst:
        dst.write(multiclass_mask, 1)
    
    print(f"Máscara multiclase ROI guardada en {roi_multiclass_mask_path}")
    
    # Mostrar estadísticas
    pixels_buildings = np.sum(multiclass_mask == 1)
    pixels_roads = np.sum(multiclass_mask == 2)
    pixels_total = multiclass_mask.size
    
    print(f"Estadísticas de la máscara multiclase ROI:")
    print(f"  - Total píxeles: {pixels_total}")
    print(f"  - Edificios: {pixels_buildings} ({pixels_buildings/pixels_total*100:.2f}%)")
    print(f"  - Carreteras: {pixels_roads} ({pixels_roads/pixels_total*100:.2f}%)")
    print(f"  - Fondo: {pixels_total - pixels_buildings - pixels_roads} ({(pixels_total - pixels_buildings - pixels_roads)/pixels_total*100:.2f}%)")
    
    return multiclass_mask, meta

def load_and_resample_bands_roi():
    """
    Carga y remuestrea solo la región de interés de todas las bandas a 10m de resolución
    """
    print("Cargando y remuestreando bandas para la ROI...")
    
    # Crear directorio para la ROI si no existe
    os.makedirs(OUTPUT_ROI_DIR, exist_ok=True)
    
    # Definir ventana para la ROI en 10m de resolución
    roi_window_10m = Window(ROI_START_X, ROI_START_Y, ROI_WIDTH, ROI_HEIGHT)
    
    roi_window_20m = Window(
        ROI_START_X // 2,     # Dividir por 2 para pasar de 10m a 20m
        ROI_START_Y // 2,
        ROI_WIDTH // 2,
        ROI_HEIGHT // 2
    )
    
    # Cargamos primero una banda de 10m para obtener las dimensiones de referencia
    reference_band_path = os.path.join(SENTINEL_DIR, BAND_FILES['B02'])
    roi_ref_data, roi_ref_transform = extract_roi_from_image(
        reference_band_path,
        roi_window_10m,
        os.path.join(OUTPUT_ROI_DIR, "B02_roi.tif")
    )
    
    ref_shape = roi_ref_data.shape[1:]  # (altura, ancho)
    
    with rasterio.open(reference_band_path) as src:
        ref_crs = src.crs
    
    stacked_bands = []
    band_names = []
    
    # Procesar cada banda
    for band_name, band_file in BAND_FILES.items():
        band_path = os.path.join(SENTINEL_DIR, band_file)
        
        # Determinar si es banda de 10m o 20m
        is_20m_band = band_name in ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
        roi_window = roi_window_20m if is_20m_band else roi_window_10m
        
        # Extraer ROI
        roi_band_data, _ = extract_roi_from_image(
            band_path,
            roi_window,
            os.path.join(OUTPUT_ROI_DIR, f"{band_name}_roi.tif")
        )
        
        # Extrae la primera banda (las imágenes satelitales suelen tener solo una banda por archivo)
        band_data = roi_band_data[0]
        
        # Remuestrear si la resolución es diferente (20m)
        if is_20m_band:
            # Remuestrear usando el método más adecuado para imágenes satelitales
            from skimage.transform import resize
            band_data = resize(
                band_data, 
                ref_shape, 
                order=1,  # Interpolación bilineal
                preserve_range=True
            ).astype(np.float32)
        
        # Normalizar los valores
        band_data = band_data.astype(np.float32)
        if band_data.max() > 0:
            band_data = band_data / band_data.max()
        
        stacked_bands.append(band_data)
        band_names.append(band_name)
    
    # Apilar todas las bandas en un solo array
    stacked_array = np.stack(stacked_bands, axis=0)
    
    print(f"Bandas ROI cargadas y remuestreadas: {', '.join(band_names)}")
    print(f"Forma del array multibanda ROI: {stacked_array.shape}")
    
    return stacked_array, roi_ref_transform, ref_crs

def create_patches(image_array, mask_array, patch_size=INPUT_SIZE, stride=128, min_nonzero_percentage=0.01):
    """
    Crea parches de entrenamiento a partir de las imágenes y máscaras
    """
    print("Creando parches para entrenamiento...")
    num_bands, height, width = image_array.shape
    
    building_patches = []
    road_patches = []
    mixed_patches = []
    background_patches = []
    
    for y in range(0, height - patch_size[0] + 1, stride):
        for x in range(0, width - patch_size[1] + 1, stride):
            if y + patch_size[0] <= height and x + patch_size[1] <= width:
                mask_patch = mask_array[y:y + patch_size[0], x:x + patch_size[1]]
                image_patch = image_array[:, y:y + patch_size[0], x:x + patch_size[1]]
                image_patch = np.transpose(image_patch, (1, 2, 0))
                
                buildings = np.sum(mask_patch == 1) / mask_patch.size
                roads = np.sum(mask_patch == 2) / mask_patch.size
                
                if buildings > 0.05:  # >5% edificios
                    building_patches.append((image_patch, mask_patch, (y, x)))
                elif roads > 0.05:    # >5% carreteras
                    road_patches.append((image_patch, mask_patch, (y, x)))
                elif buildings > 0 or roads > 0:  # algunos edificios o carreteras
                    mixed_patches.append((image_patch, mask_patch, (y, x)))
                else:
                    background_patches.append((image_patch, mask_patch, (y, x)))
    
    print(f"Parches con edificios: {len(building_patches)}")
    print(f"Parches con carreteras: {len(road_patches)}")
    print(f"Parches mixtos: {len(mixed_patches)}")
    print(f"Parches de fondo: {len(background_patches)}")
    
    # Balancear el dataset: tomar todos los parches de interés y un subconjunto del fondo
    max_bg_patches = min(len(background_patches), max(len(building_patches) + len(road_patches), 50))
    selected_bg = random.sample(background_patches, max_bg_patches)
    
    all_patches = building_patches + road_patches + mixed_patches + selected_bg
    random.shuffle(all_patches)
    
    # Extraer los componentes
    image_patches = [p[0] for p in all_patches]
    mask_patches = [p[1] for p in all_patches]
    patch_coords = [p[2] for p in all_patches]
    
    # Coordenadas de los parches (para referencia)
    patch_coords = []
    
    # Contador de parches examinados
    total_examined = 0
    total_selected = 0
    
    # Para estadísticas
    building_counts = []
    road_counts = []
    
    # Generar parches con solapamiento
    for y in range(0, height - patch_size[0] + 1, stride):
        for x in range(0, width - patch_size[1] + 1, stride):
            if y + patch_size[0] <= height and x + patch_size[1] <= width:
                total_examined += 1
                
                # Extraer parche de máscara
                mask_patch = mask_array[y:y + patch_size[0], x:x + patch_size[1]]
                
                # Contar píxeles por clase
                building_pixels = np.sum(mask_patch == 1)
                road_pixels = np.sum(mask_patch == 2)
                total_pixels = patch_size[0] * patch_size[1]
                
                # Calcular porcentajes
                building_percentage = building_pixels / total_pixels
                road_percentage = road_pixels / total_pixels
                total_percentage = (building_pixels + road_pixels) / total_pixels
                
                # Guardar estadísticas
                building_counts.append(building_percentage)
                road_counts.append(road_percentage)
                
                # Comprobar si el parche contiene información relevante
                if total_percentage >= min_nonzero_percentage:
                    # Extraer parche de imagen
                    image_patch = image_array[:, y:y + patch_size[0], x:x + patch_size[1]]
                    
                    # Reorganizar dimensiones para TensorFlow: (altura, ancho, canales)
                    image_patch = np.transpose(image_patch, (1, 2, 0))
                    
                    image_patches.append(image_patch)
                    mask_patches.append(mask_patch)
                    patch_coords.append((y, x))
                    total_selected += 1
                    
                    # Imprimir información cada 50 parches seleccionados
                    if total_selected % 50 == 0:
                        print(f"  Seleccionado parche {total_selected}: edificios={building_percentage*100:.2f}%, carreteras={road_percentage*100:.2f}%")
    
    print(f"Parches examinados: {total_examined}, Parches seleccionados: {total_selected}")
    
    # Mostrar estadísticas generales
    if building_counts:
        print(f"Estadísticas de edificios en parches: min={min(building_counts)*100:.2f}%, max={max(building_counts)*100:.2f}%, media={np.mean(building_counts)*100:.2f}%")
        print(f"Estadísticas de carreteras en parches: min={min(road_counts)*100:.2f}%, max={max(road_counts)*100:.2f}%, media={np.mean(road_counts)*100:.2f}%")
    
    # Si no se encontraron suficientes parches, usar un porcentaje menor
    if len(image_patches) < 100:  # Ajustar este valor según sea necesario
        print(f"¡Advertencia! Se encontraron solo {len(image_patches)} parches con suficientes elementos de interés.")
        if min_nonzero_percentage > 0.001:  # Umbral mínimo
            new_threshold = min_nonzero_percentage / 2
            print(f"Intentando con un umbral menor: {new_threshold}")
            return create_patches(image_array, mask_array, patch_size, stride, new_threshold)
        else:
            print("El umbral ya es muy bajo. Asegurando un mínimo de parches...")
            
            # Ordenar todos los parches examinados por porcentaje total de elementos de interés
            all_patches = []
            for y in range(0, height - patch_size[0] + 1, stride):
                for x in range(0, width - patch_size[1] + 1, stride):
                    if y + patch_size[0] <= height and x + patch_size[1] <= width:
                        mask_patch = mask_array[y:y + patch_size[0], x:x + patch_size[1]]
                        interest_percentage = np.sum(mask_patch > 0) / (patch_size[0] * patch_size[1])
                        all_patches.append((y, x, interest_percentage))
            
            # Ordenar por porcentaje descendente
            all_patches.sort(key=lambda x: x[2], reverse=True)
            
            # Tomar los 200 mejores parches
            min_patches_to_take = min(200, len(all_patches))
            print(f"Tomando los {min_patches_to_take} mejores parches por contenido de interés")
            
            for i in range(min_patches_to_take):
                y, x, percentage = all_patches[i]
                
                # Extraer parche de imagen
                image_patch = image_array[:, y:y + patch_size[0], x:x + patch_size[1]]
                mask_patch = mask_array[y:y + patch_size[0], x:x + patch_size[1]]
                
                # Reorganizar dimensiones
                image_patch = np.transpose(image_patch, (1, 2, 0))
                
                # Añadir a las listas
                if not any((y == existing_y and x == existing_x) for existing_y, existing_x in patch_coords):
                    image_patches.append(image_patch)
                    mask_patches.append(mask_patch)
                    patch_coords.append((y, x))
                    
                    # Imprime porcentaje cada 20 parches
                    if len(image_patches) % 20 == 0:
                        print(f"  Añadido parche forzado {len(image_patches)}: {percentage*100:.2f}% de elementos")
            
    # Convertir a arrays numpy
    X = np.array(image_patches) if image_patches else np.array([])
    y = np.array(mask_patches) if mask_patches else np.array([])
    
    print(f"Total de parches creados: {len(X)}")
    if len(X) > 0:
        print(f"Forma de X: {X.shape}, Forma de y: {y.shape}")
    
    return X, y, patch_coords

def save_patches_as_tiff(X, y, patch_coords):
    """
    Guarda los parches como archivos TIFF individuales
    """
    print("Guardando parches como archivos TIFF...")
    
    # Crear directorios si no existen
    os.makedirs(f"{OUTPUT_DATASET_DIR}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DATASET_DIR}/masks", exist_ok=True)
    
    # Guardar una lista de parches
    patch_list = []
    
    # Guardar cada parche como un archivo TIFF
    for i, (image_patch, mask_patch, (y_coord, x_coord)) in enumerate(zip(X, y, patch_coords)):
        # Nombres de archivos
        image_filename = f"{OUTPUT_DATASET_DIR}/images/patch_{i:04d}_y{y_coord}_x{x_coord}.tif"
        mask_filename = f"{OUTPUT_DATASET_DIR}/masks/patch_{i:04d}_y{y_coord}_x{x_coord}.tif"
        
        # Guardar imagen
        with rasterio.open(
            image_filename,
            'w',
            driver='GTiff',
            height=image_patch.shape[0],
            width=image_patch.shape[1],
            count=image_patch.shape[2],
            dtype=image_patch.dtype
        ) as dst:
            # Reorganizar para rasterio (canales, altura, ancho)
            for j in range(image_patch.shape[2]):
                dst.write(image_patch[:, :, j], j+1)
        
        # Guardar máscara
        with rasterio.open(
            mask_filename,
            'w',
            driver='GTiff',
            height=mask_patch.shape[0],
            width=mask_patch.shape[1],
            count=1,
            dtype=mask_patch.dtype
        ) as dst:
            dst.write(mask_patch, 1)
        
        # Añadir a la lista
        patch_list.append({
            'id': i,
            'image': image_filename,
            'mask': mask_filename,
            'y_coord': y_coord,
            'x_coord': x_coord
        })
    
    # Guardar lista de parches como un archivo NPZ
    np.savez(f"{OUTPUT_DATASET_DIR}/patch_list.npz", patches=patch_list)
    
    print(f"Parches guardados en {OUTPUT_DATASET_DIR}")
    return patch_list

def load_patches():
    """
    Carga los parches desde los archivos TIFF
    """
    print("Cargando parches desde archivos TIFF...")
    
    # Cargar lista de parches
    patch_list = np.load(f"{OUTPUT_DATASET_DIR}/patch_list.npz", allow_pickle=True)['patches']
    
    # Listas para almacenar los parches
    X = []
    y = []
    patch_coords = []
    
    # Cargar cada parche
    for patch in patch_list:
        # Cargar imagen
        with rasterio.open(patch['image']) as src:
            # Leer todas las bandas
            image_patch = np.zeros((src.height, src.width, src.count), dtype=np.float32)
            for i in range(src.count):
                image_patch[:, :, i] = src.read(i+1)
        
        # Cargar máscara
        with rasterio.open(patch['mask']) as src:
            mask_patch = src.read(1)
        
        X.append(image_patch)
        y.append(mask_patch)
        patch_coords.append((patch['y_coord'], patch['x_coord']))
    
    # Convertir a arrays numpy
    X = np.array(X)
    y = np.array(y)
    
    print(f"Parches cargados: {len(X)}")
    print(f"Forma de X: {X.shape}, Forma de y: {y.shape}")
    
    return X, y, patch_coords

def build_unet_model(input_shape, num_classes):
    """
    Construye el modelo U-Net para segmentación de imágenes multiespectrales
    """
    print("Construyendo modelo U-Net...")
    
    # Función auxiliar para bloques de convolución
    def conv_block(inputs, filters, kernel_size=3, activation='relu', padding='same'):
        x = layers.Conv2D(filters, kernel_size, padding=padding)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        
        x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        return x
    
    # Input
    inputs = layers.Input(input_shape)
    
    # Encoder (downsampling)
    conv1 = conv_block(inputs, 64)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = conv_block(pool1, 128)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = conv_block(pool2, 256)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = conv_block(pool3, 512)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bridge
    conv5 = conv_block(pool4, 1024)
    
    # Decoder (upsampling)
    up6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    concat6 = layers.Concatenate()([up6, conv4])
    conv6 = conv_block(concat6, 512)
    
    up7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    concat7 = layers.Concatenate()([up7, conv3])
    conv7 = conv_block(concat7, 256)
    
    up8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    concat8 = layers.Concatenate()([up8, conv2])
    conv8 = conv_block(concat8, 128)
    
    up9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    concat9 = layers.Concatenate()([up9, conv1])
    conv9 = conv_block(concat9, 64)
    
    # Output
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(conv9)
    
    # Modelo
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compilar modelo
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.MeanIoU(num_classes=num_classes),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )
    
    print(model.summary())
    return model

def prepare_data_for_training(X, y):
    """
    Prepara los datos para el entrenamiento
    """
    print("Preparando datos para entrenamiento...")
    
    # Dividir en conjuntos de entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED
    )
    
    # Convertir máscaras a formato one-hot
    y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val_cat = to_categorical(y_val, num_classes=NUM_CLASSES)
    
    print(f"X_train: {X_train.shape}, y_train: {y_train_cat.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val_cat.shape}")
    
    return X_train, X_val, y_train_cat, y_val_cat

def train_model(model, X_train, y_train, X_val, y_val, class_weights):
    """
    Entrena el modelo U-Net
    """
    print("Iniciando entrenamiento del modelo...")
    
    # Callbacks
    model_checkpoint = callbacks.ModelCheckpoint(
        OUTPUT_MODEL,
        monitor='val_mean_io_u',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Entrenamiento
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[model_checkpoint, early_stopping, reduce_lr],
        class_weight={i: w for i, w in enumerate(class_weights)}
    )
    
    return model, history

def plot_training_history(history):
    """
    Visualiza el historial de entrenamiento
    """
    # Crear figura con 2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot de pérdida
    axes[0, 0].plot(history.history['loss'], label='Train')
    axes[0, 0].plot(history.history['val_loss'], label='Validation')
    axes[0, 0].set_title('Pérdida')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    
    # Plot de precisión
    axes[0, 1].plot(history.history['accuracy'], label='Train')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 1].set_title('Precisión')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    
    # Plot de IoU
    axes[1, 0].plot(history.history['mean_io_u'], label='Train')
    axes[1, 0].plot(history.history['val_mean_io_u'], label='Validation')
    axes[1, 0].set_title('Mean IoU')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    
    # Plot de recall
    axes[1, 1].plot(history.history['recall'], label='Train Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
    axes[1, 1].plot(history.history['precision'], label='Train Precision')
    axes[1, 1].plot(history.history['val_precision'], label='Val Precision')
    axes[1, 1].set_title('Recall y Precisión')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_ROI_DIR, "training_history.png"))
    plt.show()

def predict_full_roi(model, stacked_array, ref_transform, ref_crs, patch_size=INPUT_SIZE, overlap=0.5):
    """
    Predice la imagen ROI completa utilizando un enfoque de ventana deslizante
    """
    print("Prediciendo ROI completa...")
    
    num_bands, height, width = stacked_array.shape
    # stride = int(patch_size[0] * (1 - overlap))
    stride = 64  # Mayor solapamiento que 200
    
    # Crear un array vacío para la predicción final
    prediction = np.zeros((height, width, NUM_CLASSES), dtype=np.float32)
    counts = np.zeros((height, width, 1), dtype=np.float32)
    
    # Iterar sobre la imagen con ventana deslizante
    for y in range(0, height - patch_size[0] + 1, stride):
        for x in range(0, width - patch_size[1] + 1, stride):
            if y + patch_size[0] <= height and x + patch_size[1] <= width:
                # Extraer parche
                patch = stacked_array[:, y:y + patch_size[0], x:x + patch_size[1]]
                patch = np.transpose(patch, (1, 2, 0))
                
                # Preparar para la predicción
                patch = np.expand_dims(patch, axis=0)
                
                # Predecir
                pred = model.predict(patch, verbose=0)[0]
                
                # Agregar a la predicción final
                prediction[y:y + patch_size[0], x:x + patch_size[1]] += pred
                counts[y:y + patch_size[0], x:x + patch_size[1]] += 1
    
    # Promediar las predicciones
    prediction = np.divide(prediction, counts, out=np.zeros_like(prediction), where=counts != 0)
    
    # Convertir a etiquetas de clase
    prediction_class = np.argmax(prediction, axis=-1).astype(np.uint8)
    
    # Guardar resultado como GeoTIFF
    output_path = os.path.join(OUTPUT_ROI_DIR, "prediction_roi.tif")
    
    roi_mask_path = os.path.join(OUTPUT_ROI_DIR, "multiclass_mask_roi.tif")
    with rasterio.open(roi_mask_path) as src:
        meta = src.meta.copy()
    
    meta.update({
        'count': 1,
        'dtype': 'uint8',
        'nodata': 255
    })
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(prediction_class, 1)
    
    print(f"Predicción ROI guardada en {output_path}")
    
    # Visualizar resultados
    return prediction_class, meta

def visualize_results(prediction, multiclass_mask):
    """
    Visualiza los resultados de la predicción junto con la máscara real
    """
    print("Visualizando resultados...")
    
    # Crear un colormap personalizado
    colors = [(0.1, 0.1, 0.1),    # Negro/gris oscuro para fondo
              (0.2, 0.6, 1.0),    # Azul para edificios
              (1.0, 0.4, 0.4)]    # Rojo para carreteras
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    # Crear figura con 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Máscara real
    axes[0].imshow(multiclass_mask, cmap=cmap, vmin=0, vmax=2)
    axes[0].set_title('Máscara real')
    axes[0].axis('off')
    
    # Predicción
    axes[1].imshow(prediction, cmap=cmap, vmin=0, vmax=2)
    axes[1].set_title('Predicción')
    axes[1].axis('off')
    
    # Diferencias
    diff = np.zeros_like(prediction)
    diff[prediction == multiclass_mask] = 0  # Correcto
    diff[np.logical_and(prediction != multiclass_mask, prediction == 0)] = 1  # Falso negativo
    diff[np.logical_and(prediction != multiclass_mask, prediction > 0)] = 2   # Falso positivo
    
    diff_colors = [(0.0, 0.7, 0.0),    # Verde para correcto
                   (0.7, 0.0, 0.0),    # Rojo oscuro para falso negativo
                   (1.0, 0.6, 0.0)]    # Naranja para falso positivo
    diff_cmap = plt.matplotlib.colors.ListedColormap(diff_colors)
    
    axes[2].imshow(diff, cmap=diff_cmap, vmin=0, vmax=2)
    axes[2].set_title('Diferencias')
    axes[2].axis('off')
    
    # Leyenda para las diferencias
    patches = [
        plt.matplotlib.patches.Patch(color=diff_colors[0], label='Correcto'),
        plt.matplotlib.patches.Patch(color=diff_colors[1], label='Falso negativo'),
        plt.matplotlib.patches.Patch(color=diff_colors[2], label='Falso positivo')
    ]
    axes[2].legend(handles=patches, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_ROI_DIR, "prediction_visualization.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calcular estadísticas
    accuracy = np.sum(prediction == multiclass_mask) / multiclass_mask.size
    print(f"Precisión global: {accuracy:.4f}")
    
    for class_id, class_name in enumerate(['Fondo', 'Edificios', 'Carreteras']):
        # True positives, false positives, false negatives
        tp = np.sum(np.logical_and(prediction == class_id, multiclass_mask == class_id))
        fp = np.sum(np.logical_and(prediction == class_id, multiclass_mask != class_id))
        fn = np.sum(np.logical_and(prediction != class_id, multiclass_mask == class_id))
        
        # Métricas
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Clase {class_id} ({class_name}):")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall: {recall:.4f}")
        print(f"  - F1-Score: {f1:.4f}")

def calculate_class_weights(y):
    class_counts = np.bincount(y.flatten())
    total = np.sum(class_counts)
    
    weights = total / (len(class_counts) * class_counts)
    
    weights[0] *= 1.0  # Reducir mucho el peso del fondo
    weights[1] *= 5.0  # Aumentar significativamente el peso de edificios
    weights[2] *= 10.0  # Aumentar significativamente el peso de carreteras
    
    print(f"Pesos de clase ajustados: {weights}")
    return weights

def main():
    """
    Función principal que ejecuta todo el proceso
    """
    print("Iniciando procesamiento de ROI de Sentinel-2 para detección de edificios y carreteras")
    
    # Verificar máscaras
    visualize_masks()
    
    # Crear directorio de salida si no existe
    os.makedirs(OUTPUT_ROI_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)
    
    # 1. Crear máscara multiclase para la ROI
    multiclass_mask, mask_meta = create_multiclass_mask_roi()
    
    # 2. Cargar y remuestrear las bandas para la ROI
    stacked_array, ref_transform, ref_crs = load_and_resample_bands_roi()
    
    # Comprobar si el dataset de parches ya existe y tiene contenido
    patch_list_file = os.path.join(OUTPUT_DATASET_DIR, "patch_list.npz")
    dataset_exists = os.path.exists(patch_list_file)
    
    try:
        # Intentar cargar parches existentes
        if dataset_exists:
            X, y, patch_coords = load_patches()
            # Si no hay parches, forzar la creación de nuevos
            if len(X) == 0:
                print("No se encontraron parches en el dataset existente. Generando nuevos parches...")
                dataset_exists = False
                # Eliminar el archivo vacío para evitar problemas futuros
                os.remove(patch_list_file)
    except Exception as e:
        print(f"Error al cargar parches existentes: {e}")
        dataset_exists = False
    
    if not dataset_exists:
        # 3. Crear parches para entrenamiento
        X, y, patch_coords = create_patches(stacked_array, multiclass_mask)
        
        # Si no hay parches, terminar
        if len(X) == 0:
            print("No se pudieron generar parches. Verifica las máscaras y el ROI.")
            return
        
        # 4. Guardar parches como TIFF
        save_patches_as_tiff(X, y, patch_coords)
    
    print(f"Trabajando con {len(X)} parches para entrenamiento")
    
    # 5. Preparar datos para entrenamiento
    X_train, X_val, y_train, y_val = prepare_data_for_training(X, y)
    
    # Calcular pesos de clase para manejar desbalance
    class_weights = calculate_class_weights(y)
    
    # 6. Construir modelo U-Net
    input_shape = (INPUT_SIZE[0], INPUT_SIZE[1], stacked_array.shape[0])
    model = build_unet_model(input_shape, NUM_CLASSES)
    
    # Comprobar si existe un modelo guardado
    model_exists = os.path.exists(OUTPUT_MODEL)
    
    if model_exists:
        # Cargar modelo existente
        print(f"Cargando modelo existente desde {OUTPUT_MODEL}")
        model = tf.keras.models.load_model(OUTPUT_MODEL)
    else:
        # 7. Entrenar modelo
        model, history = train_model(model, X_train, y_train, X_val, y_val, class_weights)
        
        # 8. Visualizar historial de entrenamiento
        plot_training_history(history)
    
    # 9. Predecir ROI completa
    prediction, _ = predict_full_roi(model, stacked_array, ref_transform, ref_crs)
    
    # 10. Visualizar resultados
    visualize_results(prediction, multiclass_mask)
    
    print("Procesamiento completado con éxito")

if __name__ == "__main__":
    main()