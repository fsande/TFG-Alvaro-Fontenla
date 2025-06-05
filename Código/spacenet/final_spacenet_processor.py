#!/usr/bin/env python3

import os
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import LineString, mapping, box
from rasterio.features import rasterize
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import re

class FinalSpaceNetProcessor:
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.train_cities = ['AOI_2_Vegas', 'AOI_3_Paris', 'AOI_4_Shanghai', 'AOI_5_Khartoum']
        
    def find_corresponding_ms_file(self, rgb_file, ms_dir):
        """Encuentra el archivo MS correspondiente al RGB"""
        rgb_name = rgb_file.name
        
        # Reemplazo PS-RGB -> PS-MS
        ms_name_v1 = rgb_name.replace('PS-RGB', 'PS-MS')
        ms_file_v1 = ms_dir / ms_name_v1
        if ms_file_v1.exists():
            return ms_file_v1
            
        # Buscar por número de imagen
        match = re.search(r'img(\d+)', rgb_name)
        if match:
            img_num = match.group(1)
            ms_candidates = list(ms_dir.glob(f"*img{img_num}.tif"))
            if ms_candidates:
                return ms_candidates[0]
        
        # Buscar por patrón más general
        base_parts = rgb_name.replace('PS-RGB', '').replace('.tif', '')
        ms_candidates = list(ms_dir.glob(f"*{base_parts}*.tif"))
        if ms_candidates:
            return ms_candidates[0]
            
        return None
    
    def load_road_geojson(self, city_path):
        geojson_path = city_path / 'geojson_roads'
        roads = []
        
        if not geojson_path.exists():
            return roads
            
        print("Cargando geometrías de carreteras...")
        
        geojson_files = list(geojson_path.glob('*.geojson'))
        
        for geojson_file in tqdm(geojson_files, desc="Cargando GeoJSON"):
            try:
                gdf = gpd.read_file(geojson_file)
                if not gdf.empty:
                    # Filtrar geometrías válidas
                    valid_geoms = gdf[gdf.geometry.is_valid].geometry.tolist()
                    roads.extend(valid_geoms)
                    
            except Exception as e:
                continue
        
        print(f"Cargadas {len(roads)} geometrías de carreteras")
        return roads
    
    def create_road_mask_robust(self, rgb_file, roads_geom, buffer_width=0.0001):  # Buffer en grados
        """Crear máscara de carreteras"""
        try:
            with rasterio.open(rgb_file) as src:
                transform = src.transform
                shape = src.shape
                crs = src.crs
                bounds = src.bounds
                
            # Verificar si el transform es válido
            if abs(transform.a) < 1e-10 or abs(transform.e) < 1e-10:
                print("Transform inválido, calculando desde bounds")
                # Calcular transform desde bounds
                width, height = shape[1], shape[0]
                pixel_width = (bounds.right - bounds.left) / width
                pixel_height = (bounds.top - bounds.bottom) / height
                
                from rasterio.transform import from_bounds
                transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, width, height)
                
            print(f"   Imagen: {shape[0]}x{shape[1]}")
            print(f"   CRS: {crs}, Bounds: {bounds}")
            
            if not roads_geom:
                return np.zeros(shape, dtype=np.uint8)
            
            # Crear bounding box de la imagen
            image_bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            
            # Filtrar y procesar geometrías
            processed_roads = []
            intersect_count = 0
            
            for road in roads_geom:
                try:
                    if not road.is_valid:
                        continue
                        
                    # Verificar intersección
                    if road.intersects(image_bbox):
                        intersect_count += 1
                        
                        # Aplicar buffer (en unidades de CRS - grados para EPSG:4326)
                        if buffer_width > 0:
                            buffered = road.buffer(buffer_width)
                            if buffered.is_valid and not buffered.is_empty:
                                processed_roads.append(buffered)
                        else:
                            processed_roads.append(road)
                            
                except Exception:
                    continue
            
            print(f"   Carreteras que intersectan: {intersect_count}")
            print(f"   Geometrías procesadas: {len(processed_roads)}")
            
            if not processed_roads:
                return np.zeros(shape, dtype=np.uint8)
            
            # Crear máscara
            try:
                mask = rasterize(
                    [(mapping(road), 1) for road in processed_roads],
                    out_shape=shape,
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                    all_touched=True
                )
                
                road_pixels = np.sum(mask > 0)
                print(f"   Píxeles de carretera: {road_pixels}")
                
                return mask
                
            except Exception as e:
                print(f"Error en rasterize: {e}")
                return np.zeros(shape, dtype=np.uint8)
                
        except Exception as e:
            print(f"Error creando máscara: {e}")
            return None
    
    def extract_crops_efficiently(self, rgb_img, ms_img, mask, crop_size_rgb=512):
        h_rgb, w_rgb = rgb_img.shape[:2]
        h_ms, w_ms = ms_img.shape[:2]
        
        # Calcular crop size MS basado en la relación de aspectos
        scale_factor = min(h_ms / h_rgb, w_ms / w_rgb)
        crop_size_ms = int(crop_size_rgb * scale_factor)
        
        print(f"   Crop sizes - RGB: {crop_size_rgb}x{crop_size_rgb}, MS: {crop_size_ms}x{crop_size_ms}")
        
        crops = []
        
        # Grid regular con overlap
        step = crop_size_rgb // 2  # 50% overlap
        
        for i in range(0, h_rgb - crop_size_rgb + 1, step):
            for j in range(0, w_rgb - crop_size_rgb + 1, step):
                
                # Coordenadas MS proporcionales
                i_ms = int(i * h_ms / h_rgb)
                j_ms = int(j * w_ms / w_rgb)
                
                # Verificar límites
                if (i_ms + crop_size_ms <= h_ms and j_ms + crop_size_ms <= w_ms):
                    
                    # Extraer crops
                    rgb_crop = rgb_img[i:i+crop_size_rgb, j:j+crop_size_rgb]
                    ms_crop = ms_img[i_ms:i_ms+crop_size_ms, j_ms:j_ms+crop_size_ms]
                    mask_crop = mask[i:i+crop_size_rgb, j:j+crop_size_rgb]
                    
                    # Verificar contenido de carreteras
                    road_pixels = np.sum(mask_crop > 0)
                    
                    # Threshold adaptativo
                    min_road_pixels = 20
                    
                    if road_pixels >= min_road_pixels:
                        crops.append({
                            'rgb': rgb_crop,
                            'ms': ms_crop,
                            'mask': mask_crop,
                            'coords': (i, j),
                            'road_pixels': road_pixels
                        })
        
        print(f"   Crops generados: {len(crops)}")
        return crops
    
    def process_city_final(self, city_name, output_dir, max_images=10):
        print(f"\nProcesando {city_name}...")
        print("="*50)
        
        city_path = self.data_root / 'train' / city_name
        output_city_dir = Path(output_dir) / city_name
        output_city_dir.mkdir(parents=True, exist_ok=True)
        
        # Verificar directorios
        rgb_dir = city_path / 'PS-RGB'
        ms_dir = city_path / 'PS-MS'
        
        if not rgb_dir.exists() or not ms_dir.exists():
            print("Directorios no encontrados")
            return 0
        
        # Cargar geometrías
        roads_geom = self.load_road_geojson(city_path)
        if not roads_geom:
            print("No se cargaron geometrías")
            return 0
        
        # Obtener archivos RGB
        rgb_files = sorted(list(rgb_dir.glob('*.tif')))[:max_images]
        print(f"Procesando {len(rgb_files)} imágenes...")
        
        total_crops = 0
        
        for img_idx, rgb_file in enumerate(rgb_files):
            print(f"\n[{img_idx+1}/{len(rgb_files)}] {rgb_file.name}")
            
            # Encontrar archivo MS correspondiente
            ms_file = self.find_corresponding_ms_file(rgb_file, ms_dir)
            
            if ms_file is None:
                print("Archivo MS no encontrado")
                continue
                
            print(f"   MS correspondiente: {ms_file.name}")
            
            try:
                # Cargar imágenes
                with rasterio.open(rgb_file) as src:
                    rgb_img = src.read().transpose(1, 2, 0)  # HWC
                    
                with rasterio.open(ms_file) as src:
                    ms_img = src.read().transpose(1, 2, 0)  # HWC
                
                print(f"   RGB shape: {rgb_img.shape}, MS shape: {ms_img.shape}")
                
                # Normalizar imágenes
                if rgb_img.max() > 255:
                    rgb_img = (rgb_img / rgb_img.max() * 255).astype(np.uint8)
                if ms_img.max() > 255:
                    ms_img = (ms_img / ms_img.max() * 255).astype(np.uint8)
                
                # Crear máscara
                mask = self.create_road_mask_robust(rgb_file, roads_geom)
                
                if mask is None or mask.sum() == 0:
                    print("Máscara vacía")
                    continue
                
                # Extraer crops
                crops = self.extract_crops_efficiently(rgb_img, ms_img, mask)
                
                if not crops:
                    print("No se generaron crops")
                    continue
                
                # Limitar crops por imagen
                max_crops_per_image = 20
                selected_crops = crops[:max_crops_per_image]
                
                # Guardar crops
                for crop_idx, crop in enumerate(selected_crops):
                    crop_name = f"{city_name}_{img_idx:03d}_{crop_idx:03d}"
                    
                    try:
                        np.save(output_city_dir / f"{crop_name}_rgb.npy", crop['rgb'])
                        np.save(output_city_dir / f"{crop_name}_ms.npy", crop['ms'])
                        np.save(output_city_dir / f"{crop_name}_mask.npy", crop['mask'])
                        
                        # Metadata
                        metadata = {
                            'source_rgb': rgb_file.name,
                            'source_ms': ms_file.name,
                            'city': city_name,
                            'coords': crop['coords'],
                            'road_pixels': int(crop['road_pixels'])
                        }
                        
                        with open(output_city_dir / f"{crop_name}_meta.json", 'w') as f:
                            json.dump(metadata, f)
                        
                        total_crops += 1
                        
                    except Exception as e:
                        print(f"Error guardando crop: {e}")
                        continue
                
                print(f"{len(selected_crops)} crops guardados")
                
            except Exception as e:
                print(f"Error procesando imagen: {e}")
                continue
        
        print(f"\n{city_name} completado: {total_crops} crops generados")
        return total_crops

class WorkingSpaceNetDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.samples = self._load_samples()
        print(f"Dataset cargado: {len(self.samples)} muestras")
        
    def _load_samples(self):
        samples = []
        for city_dir in self.data_dir.iterdir():
            if city_dir.is_dir():
                rgb_files = list(city_dir.glob("*_rgb.npy"))
                for rgb_file in rgb_files:
                    base_name = rgb_file.name.replace("_rgb.npy", "")
                    ms_file = city_dir / f"{base_name}_ms.npy"
                    mask_file = city_dir / f"{base_name}_mask.npy"
                    
                    if ms_file.exists() and mask_file.exists():
                        samples.append({
                            'rgb': rgb_file,
                            'ms': ms_file,
                            'mask': mask_file
                        })
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Cargar y normalizar
        rgb = np.load(sample['rgb']).astype(np.float32) / 255.0
        ms = np.load(sample['ms']).astype(np.float32) / 255.0  
        mask = np.load(sample['mask']).astype(np.int64)
        
        # Convertir a tensores
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)  # CHW
        ms = torch.from_numpy(ms).permute(2, 0, 1)    # CHW
        mask = torch.from_numpy(mask)                 # HW
        
        return {'rgb': rgb, 'ms': ms, 'mask': mask}

def main():
    """Función principal"""
    print("PROCESADOR SPACENET FINAL")
    print("="*60)
    
    data_root = "data"
    output_dir = "processed_data"
    
    processor = FinalSpaceNetProcessor(data_root)
    
    # Test con Vegas primero
    total_crops = processor.process_city_final("AOI_2_Vegas", output_dir, max_images=3)
    
    if total_crops > 0:
        print(f"Generados {total_crops} crops")
        
        # Probar dataset
        dataset = WorkingSpaceNetDataset(output_dir)
        
        if len(dataset) > 0:
            loader = DataLoader(dataset, batch_size=2, shuffle=True)
            sample = next(iter(loader))
            
            print(f"DATASET VERIFICADO:")
            print(f"   Total samples: {len(dataset)}")
            print(f"   RGB batch shape: {sample['rgb'].shape}")
            print(f"   MS batch shape: {sample['ms'].shape}")
            print(f"   Mask batch shape: {sample['mask'].shape}")
            print(f"   Valores únicos en mask: {torch.unique(sample['mask'])}")
            
            # Verificar que tenemos ambas clases
            mask_values = torch.unique(sample['mask'])
            if len(mask_values) >= 2:
                print("Mask contiene carreteras y fondo")
            else:
                print("Mask solo contiene una clase")
            
            print("Listo para entrenar")
            
            # Ofrecer procesar más ciudades
            response = input("\n¿Procesar más ciudades? (y/n): ")
            if response.lower() == 'y':
                for city in processor.train_cities[1:]:
                    city_crops = processor.process_city_final(city, output_dir, max_images=5)
                    total_crops += city_crops
                    print(f"Total acumulado: {total_crops} crops")
        else:
            print("Dataset vacío")
    else:
        print("No se generaron crops")

if __name__ == "__main__":
    main()
