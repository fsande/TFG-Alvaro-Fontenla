import numpy as np
import rasterio
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models

# 📌 Cargar las bandas de la imagen Sentinel-2 (el archivo .tif debe contener varias bandas)
def load_sentinel_image(sentinel_path, bands=[1], patch_size=512, overlap=128):  
    """Carga la imagen y la divide en parches"""
    with rasterio.open(sentinel_path) as src:
        img_bands = [src.read(band) for band in bands]
        sentinel_img = np.moveaxis(np.array(img_bands), 0, -1)
        sentinel_img = sentinel_img / 10000.0  # Normalizar a rango [0, 1] si es necesario
        
        patches = []
        h, w, c = sentinel_img.shape
        stride = patch_size - overlap
        
        # Dividir en parches con solapamiento
        for y in range(0, max(1, h-patch_size+1), stride):
            for x in range(0, max(1, w-patch_size+1), stride):
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                
                patch = sentinel_img[y:y_end, x:x_end]
                
                # Si el parche es más pequeño que el tamaño, rellenarlo con ceros
                if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                    temp_patch = np.zeros((patch_size, patch_size, c))
                    temp_patch[:patch.shape[0], :patch.shape[1], :] = patch
                    patch = temp_patch
                
                patches.append(patch)
                
    return np.array(patches)


# 📌 Cargar la máscara fusionada de carreteras y edificios
def load_mask(road_mask_path, building_mask_path, patch_size=512, overlap=128):
    """Cargar la máscara y dividirla en parches"""
    with rasterio.open(road_mask_path) as road_src:
        road_mask = road_src.read(1)  # Solo una capa de la máscara de carreteras
    
    with rasterio.open(building_mask_path) as building_src:
        building_mask = building_src.read(1)  # Solo una capa de la máscara de edificios
    
    # Fusionar las máscaras: 1 para carreteras, 2 para edificios
    fused_mask = np.zeros_like(road_mask)
    fused_mask[road_mask == 1] = 1  # Carretera
    fused_mask[building_mask == 1] = 2  # Edificio
    
    patches = []
    h, w = fused_mask.shape
    stride = patch_size - overlap
    
    # Dividir la máscara en parches con solapamiento
    for y in range(0, max(1, h-patch_size+1), stride):
        for x in range(0, max(1, w-patch_size+1), stride):
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)
            
            patch = fused_mask[y:y_end, x:x_end]
            
            # Si el parche es más pequeño que el tamaño, rellenarlo con ceros
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                temp_patch = np.zeros((patch_size, patch_size))
                temp_patch[:patch.shape[0], :patch.shape[1]] = patch
                patch = temp_patch
               
            patches.append(patch)
    
    return np.array(patches)


# 📌 Procesar los datos (sin dividir en entrenamiento y validación)
def preprocess_data(image_paths, road_mask_paths, building_mask_paths, patch_size=512, overlap=128):
    images = []
    masks = []
    
    for sentinel_path, road_mask_path, building_mask_path in zip(image_paths, road_mask_paths, building_mask_paths):
        sentinel_img = load_sentinel_image(sentinel_path, patch_size=patch_size, overlap=overlap)
        fused_mask = load_mask(road_mask_path, building_mask_path, patch_size=patch_size, overlap=overlap)
        
        sentinel_img = tf.image.resize(sentinel_img, (256, 256))
        fused_mask = tf.image.resize(fused_mask, (256, 256))
        
        images.extend(sentinel_img)
        masks.extend(fused_mask)
    
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

def unet_model(input_shape):
    inputs = layers.Input(input_shape)
    
    # Encoder path
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    # Bridge
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    # Decoder path with UpSampling2D to ensure size matching
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.UpSampling2D(size=(2, 2))(u6)  # Aumenta las dimensiones de u6
    u6_resized = layers.Lambda(lambda x: tf.image.resize(x, (c4.shape[1], c4.shape[2])))(u6)  # Redimensionar u6 para que coincida con c4
    u6 = layers.concatenate([u6_resized, c4])  # Ahora deberían tener las mismas dimensiones
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.UpSampling2D(size=(2, 2))(u7)  # Aumenta las dimensiones de u7
    u7_resized = layers.Lambda(lambda x: tf.image.resize(x, (c3.shape[1], c3.shape[2])))(u7)  # Redimensionar u7 para que coincida con c3
    u7 = layers.concatenate([u7_resized, c3])  # Concatenamos u7 con c3
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.UpSampling2D(size=(2, 2))(u8)  # Aumenta las dimensiones de u8
    u8_resized = layers.Lambda(lambda x: tf.image.resize(x, (c2.shape[1], c2.shape[2])))(u8)  # Redimensionar u8 para que coincida con c2
    u8 = layers.concatenate([u8_resized, c2])  # Ahora deberían tener las mismas dimensiones
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.UpSampling2D(size=(2, 2))(u9)  # Aumenta las dimensiones de u9
    u9_resized = layers.Lambda(lambda x: tf.image.resize(x, (c1.shape[1], c1.shape[2])))(u9)  # Redimensionar u9 para que coincida con c1
    u9 = layers.concatenate([u9_resized, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    
    # Output layer
    out = layers.Conv2D(3, (1, 1), activation='softmax')(u9)
    out_resized = layers.Lambda(lambda x: tf.image.resize(x, (512, 512)))(out)  # Asegura que la salida tenga el mismo tamaño que las etiquetas

    model = models.Model(inputs=inputs, outputs=out_resized)

    return model

# Crear los datasets (sin dividir en entrenamiento y validación)
X, y = preprocess_data(image_paths, road_mask_paths, building_mask_paths)

# 📌 Utilizar todo el conjunto de datos para entrenar (sin validación)
X_train = X
y_train = y

# 📌 Crear el modelo U-Net
input_shape = X_train[0].shape  # (alto, ancho, bandas)
model = unet_model(input_shape)

# 📌 Compilar el modelo
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 📌 Entrenar el modelo
history = model.fit(X_train, y_train, 
                    batch_size=1,  # Usar batch_size pequeño para imágenes grandes
                    epochs=20)

# 📌 Visualizar las predicciones de ejemplo
predictions = model.predict(X_train)

# Visualizar la primera imagen de entrenamiento y su predicción
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Imagen de Entrenamiento")
plt.imshow(X_train[0])

plt.subplot(1, 3, 2)
plt.title("Máscara Real")
plt.imshow(y_train[0], cmap='viridis')

plt.subplot(1, 3, 3)
plt.title("Predicción del Modelo")
plt.imshow(np.argmax(predictions[0], axis=-1), cmap='viridis')

plt.tight_layout()
plt.savefig('results.png')  # Guardar la figura por si no se puede mostrar en terminal
plt.show()

# 📌 Guardar el modelo entrenado
model.save('unet_sentinel_model.h5')
print("Modelo guardado como 'unet_sentinel_model.h5'")
