#!/usr/bin/env python3

"""
Sentinel-2 data processing file
Provides Sentinel-2 data processing functionality
"""

import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
import geopandas as gpd
from shapely.geometry import LineString, mapping
from rasterio.features import rasterize
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import requests
import time
import warnings
warnings.filterwarnings('ignore')

from sentinel2_config import Sentinel2Config, get_global_config
from sentinel2_dataset import Sentinel2Dataset
from osm_processing import OSMMaskGenerator

class Sentinel2Processor:
    """Processor for Sentinel-2 data"""
    
    def __init__(self, data_root, output_dir="processed_sentinel2"):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = get_global_config()
        self.osm_mask_generator = OSMMaskGenerator()
        
    def find_sentinel2_scenes(self):
        """Find all Sentinel-2 scenes available"""
        scenes = []
        
        # Search typical Sentinel-2 structure
        for root, dirs, files in os.walk(self.data_root):
            if any(f.endswith('_TOC-B02_10M_V210.tif') for f in files):
                scenes.append(Path(root))
                
        print(f"Found {len(scenes)} Sentinel-2 scenes")
        return scenes
    
    def load_and_resample_bands(self, scene_path):
        """Load and resample all bands to the target resolution"""
        print(f"Processing scene: {scene_path.name}")
        
        # Find band files
        band_files = {}
        for band in self.config.MS_BANDS:
            pattern = f"*TOC-{band}_*M_*.tif"
            matches = list(scene_path.glob(pattern))
            if matches:
                band_files[band] = matches[0]
            else:
                print(f"Band {band} not found")
        
        if len(band_files) < 4:  # Minimum for RGB
            print(f"Insufficient bands found: {len(band_files)}")
            return None, None
            
        # Load reference band (B02 - 10M)
        ref_band = 'B02'
        if ref_band not in band_files:
            print(f"Reference band {ref_band} not found")
            return None, None
            
        with rasterio.open(band_files[ref_band]) as ref_src:
            ref_profile = ref_src.profile
            ref_bounds = ref_src.bounds
            ref_crs = ref_src.crs
            ref_transform = ref_src.transform
            
            print(f"Reference - CRS: {ref_crs}, Bounds: {ref_bounds}")
            print(f"Shape: {ref_profile['height']}x{ref_profile['width']}")
            
        # Load and resample all bands
        bands_data = {}
        
        for band, file_path in tqdm(band_files.items(), desc="Cargando bandas"):
            with rasterio.open(file_path) as src:
                if src.profile['width'] == ref_profile['width'] and src.profile['height'] == ref_profile['height']:
                    # Same resolution, load directly
                    bands_data[band] = src.read(1).astype(np.float32)
                else:
                    # Different resolution, resample
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
        
        # Create RGB and multispectral images
        rgb_image = self._create_rgb_image(bands_data)
        ms_image = self._create_multispectral_image(bands_data)
        
        # Metadata for georeferencing
        geo_metadata = {
            'transform': ref_transform,
            'crs': ref_crs,
            'bounds': ref_bounds,
            'profile': ref_profile
        }
        
        return (rgb_image, ms_image), geo_metadata
        
    def _create_rgb_image(self, bands_data):
        """Create RGB image from bands"""
        rgb_bands = []
        for color in ['red', 'green', 'blue']:
            band_name = self.config.RGB_BANDS[color]
            if band_name in bands_data:
                rgb_bands.append(bands_data[band_name])
            else:
                print(f"Band {band_name} for {color} not available")
                
        if len(rgb_bands) == 3:
            rgb_image = np.stack(rgb_bands, axis=2)  # H, W, C
            # Normalization of RGB image
            normalization = Sentinel2Config.NORMALIZATION
            divisor = normalization['rgb_divisor']
            clip_min = normalization['clip_min']
            clip_max = normalization['clip_max']
            rgb_image = np.clip(rgb_image / divisor * 255, clip_min, clip_max).astype(np.uint8)
            return rgb_image
        return None
    
    def _create_multispectral_image(self, bands_data):
        """Create multispectral image from all bands"""
        ms_bands = []
        used_bands = []
        
        for band in self.config.MS_BANDS:
            if band in bands_data:
                ms_bands.append(bands_data[band])
                used_bands.append(band)
        
        if ms_bands:
            ms_image = np.stack(ms_bands, axis=2)  # H, W, C
            # Normalization of MS image
            normalization = Sentinel2Config.NORMALIZATION
            divisor = normalization['ms_divisor']
            clip_min = normalization['clip_min']
            clip_max = normalization['clip_max']
            ms_image = np.clip(ms_image / divisor * 255, clip_min, clip_max).astype(np.uint8)
            print(f"Multispectral image created: {len(used_bands)} bands {used_bands}")
            return ms_image
        return None
    
    def create_road_mask(self, geo_metadata, image_shape):
        """Road mask generation using OSM processing module"""
        return self.osm_mask_generator.create_road_mask_from_osm(geo_metadata, image_shape)
    
    def extract_crops(self, rgb_img, ms_img, mask, crop_size=512, min_road_pixels=50):
        """Dataset crops extraction"""
        h, w = rgb_img.shape[:2]
        crops = []
        
        # Grid with 50% overlap for better coverage
        step = crop_size // 2
        
        for i in range(0, h - crop_size + 1, step):
            for j in range(0, w - crop_size + 1, step):
                # Extract tiles
                rgb_crop = rgb_img[i:i+crop_size, j:j+crop_size]
                ms_crop = ms_img[i:i+crop_size, j:j+crop_size]
                mask_crop = mask[i:i+crop_size, j:j+crop_size]
                
                # Filter by minimum road content
                road_pixels = np.sum(mask_crop > 0)
                if road_pixels >= min_road_pixels:
                    crops.append({
                        'rgb': rgb_crop,
                        'ms': ms_crop,
                        'mask': mask_crop,
                        'coords': (i, j),
                        'road_pixels': road_pixels
                    })
                    
        return crops
                    
    def process_scene(self, scene_path, scene_id):
        """Process a complete scene"""
        print(f"\nProcessing scene {scene_id}: {scene_path.name}")
        
        # Load and process bands
        images_data, geo_metadata = self.load_and_resample_bands(scene_path)
        
        if images_data is None:
            print("Error loading images")
            return 0
            
        rgb_img, ms_img = images_data
        
        if rgb_img is None or ms_img is None:
            print("Error creating RGB/MS images")
            return 0
        
        print(f"RGB shape: {rgb_img.shape}, MS shape: {ms_img.shape}")
        
        road_mask = self.create_road_mask(geo_metadata, rgb_img.shape)
        combined_mask = road_mask.copy()
        
        # Extract crops
        crops = self.extract_crops(rgb_img, ms_img, combined_mask)
        
        if not crops:
            print("No crops generated, trying with more permissive parameters")
            permissive_min = Sentinel2Config.CROP_CONFIG['permissive_min_road_pixels']
            crops = self.extract_crops(rgb_img, ms_img, combined_mask, min_road_pixels=permissive_min)
        
        if not crops:
            print("No crops generated")
            return 0
        
        # Save crops
        scene_output_dir = self.output_dir / f"scene_{scene_id:03d}"
        scene_output_dir.mkdir(exist_ok=True)
        
        saved_crops = 0

        for crop in crops:
            crop_name = f"s2_{scene_id:03d}_{crop['coords'][0]:03d}_{crop['coords'][1]:03d}"
            
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
                        'transform': list(geo_metadata['transform'])[:6]  # Only serializable elements
                    }
                }
                
                with open(scene_output_dir / f"{crop_name}_meta.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                saved_crops += 1
                
            except Exception as e:
                print(f"Error saving crop {crop_name}: {e}")
                continue
        
        print(f"Scene processed: {saved_crops} crops saved")
        return saved_crops
    
    def process_all_scenes(self, max_scenes=None):
        """Process all scenes found"""
        scenes = self.find_sentinel2_scenes()
        
        if max_scenes:
            scenes = scenes[:max_scenes]
        
        total_crops = 0
        
        for scene_id, scene_path in enumerate(scenes):
            try:
                crops_count = self.process_scene(scene_path, scene_id)
                total_crops += crops_count
                
                if crops_count == 0:
                    print(f"Scene {scene_id} did not generate crops")
                    
            except Exception as e:
                print(f"Error processing scene {scene_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\nProcessing completed")
        print(f"Total crops generated: {total_crops}")
        print(f"Data saved in: {self.output_dir}")
        
        return total_crops

def main():
    """Main function"""
    print("Sentinel-2 data processor")
    print("=" * 60)
    
    # Configuration
    data_root = "Sentinel2/TOC_V2"  # Search recursively
    output_dir = "processed_sentinel2"
    
    # Create processor
    processor = Sentinel2Processor(data_root, output_dir)
    
    # Process scenes
    total_crops = processor.process_all_scenes(max_scenes=1)
    
    if total_crops > 0:
        # Verify dataset
        dataset = Sentinel2Dataset(output_dir)
        
        if len(dataset) > 0:
            # Test dataset
            loader = DataLoader(dataset, batch_size=2, shuffle=True)
            sample = next(iter(loader))
            
            print(f"\nDataset verified:")
            print(f"   Total samples: {len(dataset)}")
            print(f"   RGB batch shape: {sample['rgb'].shape}")
            print(f"   MS batch shape: {sample['ms'].shape}")
            print(f"   Mask batch shape: {sample['mask'].shape}")
            print(f"   Unique values in mask: {torch.unique(sample['mask'])}")
            
            # Verify class balance
            class_counts = dataset.class_distribution()
            print(f"\nClass distribution:")
            for value, count in class_counts.items():
                print(f"  Class {value}: {count} samples")
            
            print(f"\nDataset generated")
        else:
            print("Dataset empty")
    else:
        print("No data generated")

if __name__ == "__main__":
    main()