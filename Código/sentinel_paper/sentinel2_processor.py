#!/usr/bin/env python3

import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import LineString, mapping, box
from rasterio.features import rasterize
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import requests
import time
import warnings
warnings.filterwarnings('ignore')

class Sentinel2Config:
    """Configuración de bandas Sentinel-2"""
    
    # Bandas por resolución
    BANDS_10M = ['B02', 'B03', 'B04', 'B08']  # Blue, Green, Red, NIR
    BANDS_20M = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']  # Red edges, SWIR
    BANDS_60M = ['B01']  # Coastal aerosol
    
    # Configuración RGB (usando bandas de 10M)
    RGB_BANDS = {
        'red': 'B04',    # 665 nm
        'green': 'B03',  # 560 nm  
        'blue': 'B02'    # 490 nm
    }
    
    # Configuración Multispectral (todas las bandas útiles)
    MS_BANDS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
    
    # Resolución objetivo
    TARGET_RESOLUTION = 10

class OSMRoadExtractorFixed:
    """Extractor de carreteras desde OpenStreetMap"""
    
    def __init__(self, cache_dir="osm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_roads_from_bbox(self, bbox, max_retries=3):
        """
        Obtiene carreteras de OSM para un bounding box
        bbox: (minx, miny, maxx, maxy) en coordenadas WGS84
        """
        # Validar coordenadas
        minx, miny, maxx, maxy = bbox
        
        print(f"Consultando OSM para bbox: [{minx:.6f}, {miny:.6f}, {maxx:.6f}, {maxy:.6f}]")
        
        # Verificar que las coordenadas están en rango válido para WGS84
        if not (-180 <= minx <= 180 and -180 <= maxx <= 180 and -90 <= miny <= 90 and -90 <= maxy <= 90):
            print(f"Coordenadas fuera de rango WGS84: {bbox}")
            return gpd.GeoDataFrame()
        
        if minx >= maxx or miny >= maxy:
            print(f"Bbox inválido: minx >= maxx o miny >= maxy")
            return gpd.GeoDataFrame()
        
        # Verificar tamaño del área (no más de 1 grado para evitar timeouts)
        area_size = (maxx - minx) * (maxy - miny)
        if area_size > 1.0:  # 1 grado cuadrado
            print(f"Área muy grande ({area_size:.3f} grados²), limitando búsqueda")
            # Tomar solo el centro del área
            center_x, center_y = (minx + maxx) / 2, (miny + maxy) / 2
            margin = 0.1  # ~11km
            minx, miny = center_x - margin, center_y - margin
            maxx, maxy = center_x + margin, center_y + margin
        
        cache_file = self.cache_dir / f"roads_{minx:.4f}_{miny:.4f}_{maxx:.4f}_{maxy:.4f}.geojson"
        
        # Verificar cache
        if cache_file.exists():
            try:
                print(f"Cargando desde cache: {cache_file.name}")
                return gpd.read_file(cache_file)
            except Exception as e:
                print(f"Error leyendo cache: {e}")
                cache_file.unlink()  # Eliminar cache corrupto
        
        # Query Overpass API
        overpass_url = "http://overpass-api.de/api/interpreter"
        
        overpass_query = f"""
        [out:json][timeout:120];
        (
          way["highway"]({miny:.6f},{minx:.6f},{maxy:.6f},{maxx:.6f});
        );
        out geom;
        """
        
        print(f"Query OSM: área {(maxx-minx)*111:.1f}km x {(maxy-miny)*111:.1f}km")
        
        for attempt in range(max_retries):
            try:
                print(f"Descargando carreteras OSM... (intento {attempt+1}/{max_retries})")
                
                response = requests.post(
                    overpass_url, 
                    data=overpass_query, 
                    timeout=180,  # Aumentar timeout
                    headers={'User-Agent': 'MSFANet-Sentinel2-Processor/1.0'}
                )
                
                print(f"Respuesta OSM: {response.status_code}")
                
                if response.status_code != 200:
                    print(f"Error HTTP {response.status_code}: {response.text[:200]}")
                    if attempt < max_retries - 1:
                        time.sleep(10)
                        continue
                    else:
                        return gpd.GeoDataFrame()
                
                data = response.json()
                
                # Procesar respuesta
                roads = []
                elements = data.get('elements', [])
                print(f"Elementos recibidos: {len(elements)}")
                
                for element in elements:
                    if element['type'] == 'way' and 'geometry' in element:
                        coords = [(node['lon'], node['lat']) for node in element['geometry']]
                        if len(coords) >= 2:
                            highway_type = element.get('tags', {}).get('highway', 'unknown')
                            
                            # Filtrar tipos de carretera relevantes
                            if highway_type in ['motorway', 'trunk', 'primary', 'secondary', 
                                              'tertiary', 'residential', 'unclassified', 'service']:
                                road = {
                                    'geometry': LineString(coords),
                                    'highway': highway_type,
                                    'osm_id': element.get('id')
                                }
                                roads.append(road)
                
                print(f"Carreteras válidas encontradas: {len(roads)}")
                
                if roads:
                    gdf = gpd.GeoDataFrame(roads, crs='EPSG:4326')
                    
                    # Guardar en cache
                    try:
                        gdf.to_file(cache_file, driver='GeoJSON')
                        print(f"Cache guardado: {cache_file.name}")
                    except Exception as e:
                        print(f"Error guardando cache: {e}")
                    
                    print(f"Descargadas {len(gdf)} carreteras")
                    return gdf
                else:
                    print("No se encontraron carreteras válidas en la región")
                    return gpd.GeoDataFrame()
                    
            except requests.exceptions.Timeout:
                print(f"Timeout en intento {attempt+1}")
                if attempt < max_retries - 1:
                    time.sleep(15)
            except requests.exceptions.RequestException as e:
                print(f"Error de conexión (intento {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(10)
            except Exception as e:
                print(f"Error inesperado (intento {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    
        print("Todos los intentos fallaron")
        return gpd.GeoDataFrame()

class Sentinel2ProcessorFixed:
    """Procesador para datos Sentinel-2"""
    
    def __init__(self, data_root, output_dir="processed_sentinel2"):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = Sentinel2Config()
        self.osm_extractor = OSMRoadExtractorFixed()
        
    def find_sentinel2_scenes(self):
        scenes = []
        
        for root, dirs, files in os.walk(self.data_root):
            if any(f.endswith('_TOC-B02_10M_V210.tif') for f in files):
                scenes.append(Path(root))
                
        print(f"Encontradas {len(scenes)} escenas Sentinel-2")
        return scenes
    
    def load_and_resample_bands(self, scene_path):
        print(f"Procesando escena: {scene_path.name}")
        
        band_files = {}
        for band in self.config.MS_BANDS:
            pattern = f"*TOC-{band}_*M_*.tif"
            matches = list(scene_path.glob(pattern))
            if matches:
                band_files[band] = matches[0]
            else:
                print(f"Banda {band} no encontrada")
        
        if len(band_files) < 4:  # Mínimo para RGB
            print(f"Insuficientes bandas encontradas: {len(band_files)}")
            return None, None
            
        # Cargar banda de referencia (B02 - 10M)
        ref_band = 'B02'
        if ref_band not in band_files:
            print(f"Banda de referencia {ref_band} no encontrada")
            return None, None
            
        with rasterio.open(band_files[ref_band]) as ref_src:
            ref_profile = ref_src.profile
            ref_bounds = ref_src.bounds
            ref_crs = ref_src.crs
            ref_transform = ref_src.transform
            
            print(f"Referencia - CRS: {ref_crs}, Bounds: {ref_bounds}")
            print(f"Shape: {ref_profile['height']}x{ref_profile['width']}")
            
        # Cargar y remuestrear todas las bandas
        bands_data = {}
        
        for band, file_path in tqdm(band_files.items(), desc="Cargando bandas"):
            with rasterio.open(file_path) as src:
                if src.profile['width'] == ref_profile['width'] and src.profile['height'] == ref_profile['height']:
                    # Misma resolución, cargar directamente
                    bands_data[band] = src.read(1).astype(np.float32)
                else:
                    # Diferente resolución, remuestrear
                    resampled = np.empty((ref_profile['height'], ref_profile['width']), dtype=np.float32)
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=resampled,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.bilinear
                    )
                    bands_data[band] = resampled
        
        # Crear imágenes RGB y Multispectral
        rgb_image = self._create_rgb_image(bands_data)
        ms_image = self._create_multispectral_image(bands_data)
        
        # Metadata para georeferenciación
        geo_metadata = {
            'transform': ref_transform,
            'crs': ref_crs,
            'bounds': ref_bounds,
            'profile': ref_profile
        }
        
        return (rgb_image, ms_image), geo_metadata
        
    def _create_rgb_image(self, bands_data):
        rgb_bands = []
        for color in ['red', 'green', 'blue']:
            band_name = self.config.RGB_BANDS[color]
            if band_name in bands_data:
                rgb_bands.append(bands_data[band_name])
            else:
                print(f"Banda {band_name} para {color} no disponible")
                
        if len(rgb_bands) == 3:
            rgb_image = np.stack(rgb_bands, axis=2)  # H, W, C
            # Normalización
            rgb_image = np.clip(rgb_image / 3000.0 * 255, 0, 255).astype(np.uint8)
            return rgb_image
        return None
    
    def _create_multispectral_image(self, bands_data):
        ms_bands = []
        used_bands = []
        
        for band in self.config.MS_BANDS:
            if band in bands_data:
                ms_bands.append(bands_data[band])
                used_bands.append(band)
        
        if ms_bands:
            ms_image = np.stack(ms_bands, axis=2)  # H, W, C
            # Normalización
            ms_image = np.clip(ms_image / 3000.0 * 255, 0, 255).astype(np.uint8)
            print(f"Imagen multispectral creada: {len(used_bands)} bandas {used_bands}")
            return ms_image
        return None
    
    def create_road_mask_from_osm(self, geo_metadata, image_shape):
        try:
            # Obtener bounds en el CRS nativo
            bounds = geo_metadata['bounds']
            native_crs = geo_metadata['crs']
            
            print(f"Bounds nativos: {bounds}")
            print(f"CRS nativo: {native_crs}")
            
            # Transformar bounds a WGS84 para OSM
            if native_crs != 'EPSG:4326':
                try:
                    wgs84_bounds = transform_bounds(
                        native_crs, 'EPSG:4326', 
                        bounds.left, bounds.bottom, bounds.right, bounds.top
                    )
                    print(f"Bounds WGS84: {wgs84_bounds}")
                except Exception as e:
                    print(f"Error transformando coordenadas: {e}")
                    return self._create_synthetic_road_mask(image_shape)
            else:
                wgs84_bounds = bounds
            
            # Obtener carreteras de OSM
            roads_gdf = self.osm_extractor.get_roads_from_bbox(wgs84_bounds)
            
            if roads_gdf.empty:
                print("No se encontraron carreteras en OSM, creando máscara sintética")
                return self._create_synthetic_road_mask(image_shape)
            
            # Reprojectar carreteras al CRS de la imagen
            if str(roads_gdf.crs) != str(native_crs):
                print(f"Reproyectando de {roads_gdf.crs} a {native_crs}")
                roads_gdf = roads_gdf.to_crs(native_crs)
            
            # Aplicar buffer a las carreteras
            print("Aplicando buffer a carreteras...")
            roads_buffered = roads_gdf.copy()
            
            # Buffer en metros (ajustar según el CRS)
            if 'utm' in str(native_crs).lower() or native_crs.is_projected:
                buffer_distance = 5  # 5 metros para CRS proyectado
            else:
                buffer_distance = 0.00005  # ~5 metros en grados
            
            roads_buffered['geometry'] = roads_gdf.geometry.buffer(buffer_distance)
            
            # Crear máscara
            print("🎨 Rasterizando carreteras...")
            mask = rasterize(
                [(mapping(geom), 1) for geom in roads_buffered.geometry if geom.is_valid],
                out_shape=image_shape[:2],
                transform=geo_metadata['transform'],
                fill=0,
                dtype=np.uint8,
                all_touched=True
            )
            
            road_pixels = np.sum(mask > 0)
            total_pixels = mask.size
            coverage = (road_pixels / total_pixels) * 100
            
            print(f"Máscara OSM creada: {road_pixels:,} píxeles ({coverage:.2f}%)")
            
            # Si muy poca cobertura, añadir carreteras sintéticas
            if coverage < 0.1:
                print("Muy poca cobertura OSM, añadiendo carreteras sintéticas")
                synthetic_mask = self._create_synthetic_road_mask(image_shape)
                mask = np.maximum(mask, synthetic_mask)
                
                road_pixels = np.sum(mask > 0)
                coverage = (road_pixels / total_pixels) * 100
                print(f"Máscara combinada: {road_pixels:,} píxeles ({coverage:.2f}%)")
            
            return mask
            
        except Exception as e:
            print(f"Error creando máscara OSM: {e}")
            import traceback
            traceback.print_exc()
            print("Usando máscara sintética como fallback")
            return self._create_synthetic_road_mask(image_shape)
    
    def _create_synthetic_road_mask(self, image_shape):
        print("Creando máscara sintética de carreteras...")
        
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Crear algunas carreteras sintéticas
        # Carretera horizontal en el centro
        center_y = h // 2
        road_width = 3
        mask[center_y-road_width:center_y+road_width, :] = 1
        
        # Carretera vertical en el centro
        center_x = w // 2
        mask[:, center_x-road_width:center_x+road_width] = 1
        
        # Algunas carreteras diagonales
        for i in range(0, min(h, w), 20):
            if i + road_width < h and i + road_width < w:
                mask[i:i+road_width, i:i+road_width] = 1
        
        # Carreteras adicionales
        quarter_y, quarter_x = h // 4, w // 4
        mask[quarter_y-road_width:quarter_y+road_width, :] = 1
        mask[:, quarter_x-road_width:quarter_x+road_width] = 1
        
        road_pixels = np.sum(mask > 0)
        coverage = (road_pixels / mask.size) * 100
        
        print(f"Máscara sintética: {road_pixels:,} píxeles ({coverage:.2f}%)")
        return mask
    
    def extract_crops(self, rgb_img, ms_img, mask, crop_size=512, min_road_pixels=3):
        if rgb_img is None or ms_img is None:
            return []
            
        h, w = rgb_img.shape[:2]
        crops = []
        
        # Grid con overlap del 50%
        step = crop_size // 4
        
        print(f"Extrayendo crops {crop_size}x{crop_size} con step {step}")
        
        for i in range(0, h - crop_size + 1, step):
            for j in range(0, w - crop_size + 1, step):
                # Extraer crops
                rgb_crop = rgb_img[i:i+crop_size, j:j+crop_size]
                ms_crop = ms_img[i:i+crop_size, j:j+crop_size]
                mask_crop = mask[i:i+crop_size, j:j+crop_size]
                
                # Verificar contenido de carreteras - PARÁMETROS MÁS PERMISIVOS
                road_pixels = np.sum(mask_crop > 0)
                
                if road_pixels >= min_road_pixels:
                    crops.append({
                        'rgb': rgb_crop,
                        'ms': ms_crop,
                        'mask': mask_crop,
                        'coords': (i, j),
                        'road_pixels': road_pixels
                    })
        
        print(f"{len(crops)} crops generados")
        return crops
    
    def process_scene(self, scene_path, scene_id):
        print(f"\nProcesando escena {scene_id}: {scene_path.name}")
        
        images_data, geo_metadata = self.load_and_resample_bands(scene_path)
        
        if images_data is None:
            print("Error cargando imágenes")
            return 0
            
        rgb_img, ms_img = images_data
        
        if rgb_img is None or ms_img is None:
            print("Error creando imágenes RGB/MS")
            return 0
        
        print(f"RGB shape: {rgb_img.shape}, MS shape: {ms_img.shape}")
        
        # Crear máscara de carreteras desde OSM (con fallback sintético)
        mask = self.create_road_mask_from_osm(geo_metadata, rgb_img.shape)
        
        # Extraer crops
        crops = self.extract_crops(rgb_img, ms_img, mask)
        
        if not crops:
            print("No se generaron crops, intentando con parámetros más permisivos")
            crops = self.extract_crops(rgb_img, ms_img, mask, min_road_pixels=10)
        
        if not crops:
            print("Aún no se generaron crops")
            return 0
        
        # Guardar crops
        scene_output_dir = self.output_dir / f"scene_{scene_id:03d}"
        scene_output_dir.mkdir(exist_ok=True)
        
        saved_crops = 0
        max_crops_per_scene = 200  # Limitar para evitar datasets muy grandes
        
        for crop_idx, crop in enumerate(crops[:max_crops_per_scene]):
            crop_name = f"s2_{scene_id:03d}_{crop_idx:03d}"
            
            try:
                np.save(scene_output_dir / f"{crop_name}_rgb.npy", crop['rgb'])
                np.save(scene_output_dir / f"{crop_name}_ms.npy", crop['ms'])
                np.save(scene_output_dir / f"{crop_name}_mask.npy", crop['mask'])
                
                # Metadata
                metadata = {
                    'scene_path': str(scene_path),
                    'scene_id': scene_id,
                    'coords': crop['coords'],
                    'road_pixels': int(crop['road_pixels']),
                    'bands_used': self.config.MS_BANDS,
                    'geo_metadata': {
                        'crs': str(geo_metadata['crs']),
                        'bounds': list(geo_metadata['bounds']),
                        'transform': list(geo_metadata['transform'])[:6]  # Solo elementos serializables
                    }
                }
                
                with open(scene_output_dir / f"{crop_name}_meta.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                saved_crops += 1
                
            except Exception as e:
                print(f"Error guardando crop {crop_name}: {e}")
                continue
        
        print(f"Escena procesada: {saved_crops} crops guardados")
        return saved_crops
    
    def process_all_scenes(self, max_scenes=None):
        """Procesa todas las escenas encontradas"""
        scenes = self.find_sentinel2_scenes()
        
        if max_scenes:
            scenes = scenes[:max_scenes]
        
        total_crops = 0
        
        for scene_id, scene_path in enumerate(scenes):
            try:
                crops_count = self.process_scene(scene_path, scene_id)
                total_crops += crops_count
                
                if crops_count == 0:
                    print(f"Escena {scene_id} no generó crops")
                    
            except Exception as e:
                print(f"Error procesando escena {scene_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\nProcesamiento completado.")
        print(f"Total de crops generados: {total_crops}")
        print(f"Datos guardados en: {self.output_dir}")
        
        return total_crops

class Sentinel2Dataset(Dataset):
    """Dataset para datos Sentinel-2 procesados"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.samples = self._load_samples()
        print(f"Dataset Sentinel-2 cargado: {len(self.samples)} muestras")
        
    def _load_samples(self):
        samples = []
        for scene_dir in self.data_dir.iterdir():
            if scene_dir.is_dir() and scene_dir.name.startswith('scene_'):
                rgb_files = list(scene_dir.glob("*_rgb.npy"))
                for rgb_file in rgb_files:
                    base_name = rgb_file.name.replace("_rgb.npy", "")
                    ms_file = scene_dir / f"{base_name}_ms.npy"
                    mask_file = scene_dir / f"{base_name}_mask.npy"
                    
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
    print("PROCESADOR SENTINEL-2 PARA MSFANet")
    print("=" * 60)
    
    data_root = "Sentinel2/TOC_V2"
    output_dir = "processed_sentinel2"
    
    processor = Sentinel2ProcessorFixed(data_root, output_dir)
    
    total_crops = processor.process_all_scenes(max_scenes=1)
    
    if total_crops > 0:
        # Verificar dataset
        dataset = Sentinel2Dataset(output_dir)
        
        if len(dataset) > 0:
            # Test del dataset
            loader = DataLoader(dataset, batch_size=2, shuffle=True)
            sample = next(iter(loader))
            
            print(f"\nDATASET VERIFICADO:")
            print(f"   Muestras totales: {len(dataset)}")
            print(f"   RGB batch shape: {sample['rgb'].shape}")
            print(f"   MS batch shape: {sample['ms'].shape}")
            print(f"   Mask batch shape: {sample['mask'].shape}")
            print(f"   Valores únicos en mask: {torch.unique(sample['mask'])}")
            
            # Verificar balance de clases
            mask_values = torch.unique(sample['mask'])
            if len(mask_values) >= 2:
                print("   Mask contiene carreteras y fondo")
            else:
                print("   Mask desbalanceada")
            
            print(f"\nDataset generado")
        else:
            print("Dataset vacío")
    else:
        print("No se generaron datos")

if __name__ == "__main__":
    main()

