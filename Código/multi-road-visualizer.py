import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 📌 2. Cargar la máscara multiclase de carreteras
road_mask_path = "roads_multiclass_mask.tif"
with rasterio.open(road_mask_path) as mask_src:
    road_mask = mask_src.read(1)

# 📌 3. Definir los colores para cada tipo de carretera
road_colors = {
    1: [255, 0, 0],        # motorway (rojo)
    2: [255, 165, 0],      # trunk (naranja)
    3: [0, 255, 0],        # primary (verde)
    4: [0, 0, 255],        # secondary (azul)
    6: [255, 255, 0],      # residential (amarillo)
}

# Convertir los colores a un formato adecuado para matplotlib
road_colors_list = [road_colors[i] for i in sorted(road_colors.keys())]
road_colors_list = np.array(road_colors_list) / 255.0  # Normalizar a [0, 1]
road_cmap = ListedColormap(road_colors_list)

# 📌 4. Crear una imagen RGB a partir de la máscara
road_rgb = np.zeros((*road_mask.shape, 3), dtype=np.uint8)
for road_type, color in road_colors.items():
    road_rgb[road_mask == road_type] = color
road_rgb = road_rgb.astype(np.uint8)

leyenda = {
    1: "Motorway",
    2: "Trunk",
    3: "Primary",
    4: "Secondary",
    6: "Residential",
}

# 📌 5. Visualizar la imagen
# Crear leyenda
for road_type, color in road_colors.items():
    plt.scatter([], [], color=np.array(color) / 255.0, label=leyenda[road_type], s=100)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Tipos de Carretera", fontsize=8)
plt.tight_layout()
plt.subplots_adjust(right=0.75)  # Ajustar el espacio para la leyenda
# Mostrar la imagen
plt.imshow(road_rgb)
plt.title("Máscara Multiclase de Carreteras")
plt.axis("off")  # Ocultar ejes

plt.show()