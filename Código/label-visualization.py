import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Ruta de la imagen de clasificación de escena
file = "Sentinel2/TOC_V2/2025/03/05/S2B_20250305T115219_28RCS_TOC_V210/S2B_20250305T115219_28RCS_SCENECLASSIFICATION_20M_V210.tif"

# Cargar la imagen
with rasterio.open(file) as src:
    img = src.read(1)  # Leer la banda de clasificación

# Definir los colores para cada clase
class_colors = {
    0: "black",          # Sin datos
    1: "red",            # Saturado o defectuoso
    2: "dimgray",        # Zonas oscuras
    3: "saddlebrown",    # Sombras de nubes
    4: "forestgreen",    # Vegetación
    5: "yellow",         # Suelo desnudo
    6: "royalblue",      # Agua
    7: "gray",           # Sin clasificar
    8: "darkgray",       # Nube (media probabilidad)
    9: "white",          # Nube (alta probabilidad)
    10: "lightblue",     # Nubes delgadas
    11: "rebeccapurple", # Nieve o hielo
}

# Crear un mapa de colores basado en la clasificación
cmap = mcolors.ListedColormap([class_colors[i] for i in sorted(class_colors.keys())])
bounds = list(class_colors.keys())
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Graficar la imagen con la leyenda de colores
plt.figure(figsize=(10, 8))
plt.imshow(img, cmap=cmap, norm=norm)
plt.colorbar(label="Clase", ticks=bounds)
plt.title("Clasificación de escena Sentinel-2")
plt.show()
