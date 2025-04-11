import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap, BoundaryNorm

# 📌 Cargar la máscara fusionada
with rasterio.open("fused_multiclass_mask.tif") as src:
    fused_mask = src.read(1)

# 📌 Definir clases y colores
class_labels = {
    0: "Fondo",
    1: "Motorway",
    2: "Trunk",
    3: "Primary",
    4: "Secondary",
    6: "Residential",
    7: "Edificio"
}

colors = [
    "#000000",  # 0 - Fondo
    "#ff0000",  # 1 - Motorway
    "#ffa500",  # 2 - Trunk
    "#ffff00",  # 3 - Primary
    "#00ff00",  # 4 - Secondary
    "#00ffff",  # 5 (no usado, solo como relleno)
    "#0000ff",  # 6 - Residential
    "#800080",  # 7 - Edificio
]

# Asegurarse que el mapa tenga los índices correctos
cmap = ListedColormap(colors)
bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8]
norm = BoundaryNorm(bounds, cmap.N)

# 📌 Visualizar
plt.figure(figsize=(12, 10))
im = plt.imshow(fused_mask, cmap=cmap, norm=norm)
cbar = plt.colorbar(im, ticks=list(class_labels.keys()))
cbar.ax.set_yticklabels([class_labels[k] for k in class_labels])
plt.title("Máscara multiclase: Carreteras y Edificios", fontsize=15)
plt.axis('off')
plt.tight_layout()
plt.show()