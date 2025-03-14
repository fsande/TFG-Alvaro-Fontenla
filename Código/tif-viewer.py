import rasterio
import matplotlib.pyplot as plt

# Cargar imagen
with rasterio.open("imagen.tif") as src:
    img = src.read(1)  # Leer la única banda

# Mostrar la imagen en escala de grises
plt.imshow(img, cmap="gray")
plt.colorbar(label="Intensidad")
plt.title("Imagen TIF (Monobanda)")
plt.show()