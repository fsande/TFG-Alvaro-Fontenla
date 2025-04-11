import rasterio
import numpy as np

# 📌 Cargar máscara de carreteras
with rasterio.open("roads_multiclass_mask.tif") as road_src:
    road_mask = road_src.read(1)
    road_meta = road_src.meta

# 📌 Cargar máscara de edificios
with rasterio.open("buildings_mask.tif") as building_src:
    building_mask = building_src.read(1)

# 📌 Crear máscara combinada (multiclase)
fused_mask = road_mask.copy()

# Reasignar valor 1 de edificios a clase 7
fused_mask[building_mask == 1] = 7  # Si se superpone con carretera, la clase edificio sobrescribe

# 📌 Guardar máscara multiclase final
fused_mask_path = "fused_multiclass_mask.tif"
fused_meta = road_meta.copy()
fused_meta.update({"dtype": "uint8", "count": 1, "nodata": 0})

with rasterio.open(fused_mask_path, "w", **fused_meta) as dst:
    dst.write(fused_mask, 1)

print("✅ Máscara multiclase combinada guardada como 'fused_multiclass_mask.tif'")