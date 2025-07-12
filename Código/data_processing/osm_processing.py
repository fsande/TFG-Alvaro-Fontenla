#!/usr/bin/env python3

"""
OSM processing script
Handles all OpenStreetMap data extraction and processing functionality
"""

import os
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import LineString, Polygon, mapping
from pathlib import Path
import requests
import time
import json
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

from sentinel2_config import get_global_config

class OSMCoordinateValidator:
    """Utility class for validating and processing OSM coordinates"""
    
    @staticmethod
    def validate_wgs84_bbox(bbox: Tuple[float, float, float, float]) -> bool:
        """
        Validate WGS84 bounding box coordinates
        
        Args:
            bbox: (minx, miny, maxx, maxy) in WGS84 coordinates
            
        Returns:
            True if valid, False otherwise
        """
        minx, miny, maxx, maxy = bbox
        
        # Check coordinate ranges
        if not (-180 <= minx <= 180 and -180 <= maxx <= 180):
            return False
        if not (-90 <= miny <= 90 and -90 <= maxy <= 90):
            return False
        
        # Check logical consistency
        if minx >= maxx or miny >= maxy:
            return False
            
        return True
    
    @staticmethod
    def limit_bbox_area(bbox: Tuple[float, float, float, float], 
                       max_area: float = 1.0) -> Tuple[float, float, float, float]:
        """
        Limit bounding box area to prevent OSM API timeouts
        
        Args:
            bbox: (minx, miny, maxx, maxy) in WGS84 coordinates
            max_area: Maximum area in square degrees
            
        Returns:
            Limited bounding box
        """
        minx, miny, maxx, maxy = bbox
        area_size = (maxx - minx) * (maxy - miny)
        
        if area_size <= max_area:
            return bbox
        
        # Take center of the area and apply margin
        center_x, center_y = (minx + maxx) / 2, (miny + maxy) / 2
        margin = get_global_config().OSM_CONFIG.get('fallback_margin', 0.1)
        
        new_minx, new_miny = center_x - margin, center_y - margin
        new_maxx, new_maxy = center_x + margin, center_y + margin
        
        print(f"Area too large ({area_size:.3f} degrees²), limited to center region")
        return (new_minx, new_miny, new_maxx, new_maxy)

class OSMCacheManager:
    """Manages caching for OSM data"""
    
    def __init__(self, cache_dir: str = "osm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_path(self, feature_type: str, bbox: Tuple[float, float, float, float]) -> Path:
        """Generate cache file path for given feature type and bbox"""
        minx, miny, maxx, maxy = bbox
        return self.cache_dir / f"{feature_type}_{minx:.4f}_{miny:.4f}_{maxx:.4f}_{maxy:.4f}.geojson"
    
    def load_from_cache(self, cache_path: Path) -> Optional[gpd.GeoDataFrame]:
        """Load data from cache file"""
        if not cache_path.exists():
            return None
        
        try:
            print(f"Loading from cache: {cache_path.name}")
            return gpd.read_file(cache_path)
        except Exception as e:
            print(f"Error reading cache: {e}")
            cache_path.unlink()  # Delete corrupted cache
            return None
    
    def save_to_cache(self, gdf: gpd.GeoDataFrame, cache_path: Path) -> bool:
        """Save GeoDataFrame to cache"""
        try:
            gdf.to_file(cache_path, driver='GeoJSON')
            print(f"Cache saved: {cache_path.name}")
            return True
        except Exception as e:
            print(f"Error saving cache: {e}")
            return False

class OSMOverpassAPI:
    """Handles communication with Overpass API"""
    
    def __init__(self):
        self.base_url = "http://overpass-api.de/api/interpreter"
    
    def query_roads(self, bbox: Tuple[float, float, float, float], 
                   road_types: Optional[List[str]] = None) -> Dict:
        """
        Query roads from Overpass API
        
        Args:
            bbox: (minx, miny, maxx, maxy) in WGS84 coordinates
            road_types: List of road types to filter (if None, uses default)
            
        Returns:
            JSON response from Overpass API
        """
        minx, miny, maxx, maxy = bbox
        
        if road_types is None:
            # Default road types
            road_filter = '["highway"]'
        else:
            # Specific road types
            road_filter = f'["highway"~"^({"|".join(road_types)})$"]'
        
        overpass_query = f"""
        [out:json][timeout:120];
        (
          way{road_filter}({miny:.6f},{minx:.6f},{maxy:.6f},{maxx:.6f});
        );
        out geom;
        """
        
        return self._execute_query(overpass_query)
    
    def query_buildings(self, bbox: Tuple[float, float, float, float]) -> Dict:
        """
        Query buildings from Overpass API
        
        Args:
            bbox: (minx, miny, maxx, maxy) in WGS84 coordinates
            
        Returns:
            JSON response from Overpass API
        """
        minx, miny, maxx, maxy = bbox
        
        overpass_query = f"""
        [out:json][timeout:120];
        (
          way["building"]({miny:.6f},{minx:.6f},{maxy:.6f},{maxx:.6f});
          relation["building"]({miny:.6f},{minx:.6f},{maxy:.6f},{maxx:.6f});
        );
        out geom;
        """
        
        return self._execute_query(overpass_query)
    
    def _execute_query(self, query: str, max_retries: int = 3) -> Dict:
        """Execute Overpass API query with retries"""
        for attempt in range(max_retries):
            try:
                print(f"Querying Overpass API... (attempt {attempt+1}/{max_retries})")
                
                response = requests.post(
                    self.base_url,
                    data=query,
                    timeout=180,
                    headers={'User-Agent': self.user_agent}
                )
                
                print(f"API response: {response.status_code}")
                
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"HTTP error {response.status_code}: {response.text[:200]}")
                    if attempt < max_retries - 1:
                        time.sleep(10)
                        continue
                    else:
                        return {'elements': []}
            
            except requests.exceptions.Timeout:
                print(f"Timeout in attempt {attempt+1}")
                if attempt < max_retries - 1:
                    time.sleep(15)
            except requests.exceptions.RequestException as e:
                print(f"Connection error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(10)
            except Exception as e:
                print(f"Unexpected error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
        
        print("All API attempts failed")
        return {'elements': []}

class OSMRoadExtractor:
    """Extract roads from OpenStreetMap"""
    
    def __init__(self, cache_dir: str = "osm_cache"):
        self.cache_manager = OSMCacheManager(cache_dir)
        self.api = OSMOverpassAPI()
        self.valid_road_types = [
            'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
            'residential', 'service', 'motorway_link',
            'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link'
        ]
    
    def get_roads_from_bbox(self, bbox: Tuple[float, float, float, float], 
                           max_retries: int = 3) -> gpd.GeoDataFrame:
        """
        Get roads from OSM for a bounding box
        
        Args:
            bbox: (minx, miny, maxx, maxy) in WGS84 coordinates
            max_retries: Maximum number of retry attempts
            
        Returns:
            GeoDataFrame with road geometries
        """
        # Validate coordinates
        if not OSMCoordinateValidator.validate_wgs84_bbox(bbox):
            print(f"Invalid bounding box: {bbox}")
            return gpd.GeoDataFrame()
        
        minx, miny, maxx, maxy = bbox
        print(f"Querying OSM for roads in bbox: [{minx:.6f}, {miny:.6f}, {maxx:.6f}, {maxy:.6f}]")
        
        # Limit area size to prevent timeouts
        config = get_global_config()
        max_area = config.OSM_CONFIG.get('max_area_degrees', 1.0)
        bbox = OSMCoordinateValidator.limit_bbox_area(bbox, max_area)
        
        # Check cache first
        cache_path = self.cache_manager.get_cache_path("roads", bbox)
        cached_data = self.cache_manager.load_from_cache(cache_path)
        if cached_data is not None:
            return cached_data
        
        # Query API
        minx, miny, maxx, maxy = bbox
        print(f"Query OSM roads: area {(maxx-minx)*111:.1f}km x {(maxy-miny)*111:.1f}km")
        
        data = self.api.query_roads(bbox)
        
        # Process response
        roads = self._process_road_elements(data.get('elements', []))
        
        if roads:
            gdf = gpd.GeoDataFrame(roads, crs='EPSG:4326')
            self.cache_manager.save_to_cache(gdf, cache_path)
            print(f"Downloaded {len(gdf)} roads")
            return gdf
        else:
            print("No valid roads found")
            return gpd.GeoDataFrame()
    
    def _process_road_elements(self, elements: List[Dict]) -> List[Dict]:
        """Process OSM elements into road features"""
        roads = []
        print(f"Processing {len(elements)} OSM elements")
        
        for element in elements:
            if element['type'] == 'way' and 'geometry' in element:
                coords = [(node['lon'], node['lat']) for node in element['geometry']]
                
                if len(coords) >= 2:
                    highway_type = element.get('tags', {}).get('highway', 'unknown')
                    
                    # Filter relevant road types
                    if highway_type in self.valid_road_types:
                        road = {
                            'geometry': LineString(coords),
                            'highway': highway_type,
                            'osm_id': element.get('id'),
                            'name': element.get('tags', {}).get('name', ''),
                            'surface': element.get('tags', {}).get('surface', ''),
                            'lanes': element.get('tags', {}).get('lanes', '')
                        }
                        roads.append(road)
        
        print(f"Valid roads found: {len(roads)}")
        return roads

class OSMBuildingExtractor:
    """Extract buildings from OpenStreetMap"""
    
    def __init__(self, cache_dir: str = "osm_cache_buildings"):
        self.cache_manager = OSMCacheManager(cache_dir)
        self.api = OSMOverpassAPI()
    
    def get_buildings_from_bbox(self, bbox: Tuple[float, float, float, float]) -> gpd.GeoDataFrame:
        """
        Get buildings from OSM for a bounding box
        
        Args:
            bbox: (minx, miny, maxx, maxy) in WGS84 coordinates
            
        Returns:
            GeoDataFrame with building geometries
        """
        # Validate coordinates
        if not OSMCoordinateValidator.validate_wgs84_bbox(bbox):
            print(f"Invalid bounding box: {bbox}")
            return gpd.GeoDataFrame()
        
        minx, miny, maxx, maxy = bbox
        print(f"Querying OSM for buildings in bbox: [{minx:.6f}, {miny:.6f}, {maxx:.6f}, {maxy:.6f}]")
        
        # Limit area size
        config = get_global_config()
        max_area = config.OSM_CONFIG.get('max_area_degrees', 1.0)
        bbox = OSMCoordinateValidator.limit_bbox_area(bbox, max_area)
        
        # Check cache
        cache_path = self.cache_manager.get_cache_path("buildings", bbox)
        cached_data = self.cache_manager.load_from_cache(cache_path)
        if cached_data is not None:
            return cached_data
        
        # Query API
        minx, miny, maxx, maxy = bbox
        print(f"Query OSM buildings: area {(maxx-minx)*111:.1f}km x {(maxy-miny)*111:.1f}km")
        
        data = self.api.query_buildings(bbox)
        
        # Process response
        buildings = self._process_building_elements(data.get('elements', []))
        
        if buildings:
            gdf = gpd.GeoDataFrame(buildings, crs='EPSG:4326')
            self.cache_manager.save_to_cache(gdf, cache_path)
            print(f"Downloaded {len(gdf)} buildings")
            return gdf
        else:
            print("No valid buildings found")
            return gpd.GeoDataFrame()
    
    def _process_building_elements(self, elements: List[Dict]) -> List[Dict]:
        """Process OSM elements into building features"""
        buildings = []
        print(f"Processing {len(elements)} OSM elements")
        
        for element in elements:
            if element['type'] == 'way' and 'geometry' in element:
                coords = [(node['lon'], node['lat']) for node in element['geometry']]
                
                if len(coords) >= 3:
                    # Close polygon if not already closed
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                    
                    building_type = element.get('tags', {}).get('building', 'yes')
                    
                    building = {
                        'geometry': Polygon(coords),
                        'building': building_type,
                        'osm_id': element.get('id'),
                        'name': element.get('tags', {}).get('name', ''),
                        'height': element.get('tags', {}).get('height', ''),
                        'levels': element.get('tags', {}).get('building:levels', '')
                    }
                    buildings.append(building)
            
            elif element['type'] == 'relation' and 'members' in element:
                # Handle building relations (multipolygons)
                for member in element['members']:
                    if member['type'] == 'way' and 'geometry' in member:
                        coords = [(node['lon'], node['lat']) for node in member['geometry']]
                        if len(coords) >= 3:
                            if coords[0] != coords[-1]:
                                coords.append(coords[0])
                            
                            building_type = element.get('tags', {}).get('building', 'yes')
                            
                            building = {
                                'geometry': Polygon(coords),
                                'building': building_type,
                                'osm_id': element.get('id'),
                                'name': element.get('tags', {}).get('name', ''),
                                'height': element.get('tags', {}).get('height', ''),
                                'levels': element.get('tags', {}).get('building:levels', '')
                            }
                            buildings.append(building)
        
        print(f"Valid buildings found: {len(buildings)}")
        return buildings

class OSMMaskGenerator:
    """Generate raster masks from OSM data"""
    
    def __init__(self):
        self.road_extractor = OSMRoadExtractor()
        self.building_extractor = OSMBuildingExtractor()
        self.config = get_global_config()
    
    def create_road_mask_from_osm(self, geo_metadata: Dict, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create road mask from OSM data
        
        Args:
            geo_metadata: Dictionary with 'bounds', 'crs', 'transform' keys
            image_shape: (height, width) of the target image
            
        Returns:
            Binary mask array with roads as 1, background as 0
        """
        try:
            # Transform bounds to WGS84 for OSM
            bounds = geo_metadata['bounds']
            native_crs = geo_metadata['crs']
            
            if native_crs != 'EPSG:4326':
                wgs84_bounds = transform_bounds(
                    native_crs, 'EPSG:4326',
                    bounds.left, bounds.bottom, bounds.right, bounds.top
                )
            else:
                wgs84_bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)
            
            # Get roads from OSM
            roads_gdf = self.road_extractor.get_roads_from_bbox(wgs84_bounds)
            
            if roads_gdf.empty:
                print("No roads found in OSM")
                return np.zeros(image_shape[:2], dtype=np.uint8)
            
            # Reproject to native CRS and apply buffer
            roads_gdf = roads_gdf.to_crs(native_crs)
            buffer_distance = self._get_road_buffer_distance(native_crs)
            roads_gdf['geometry'] = roads_gdf.geometry.buffer(buffer_distance)
            
            # Rasterize roads
            mask = rasterize(
                [(mapping(geom), 1) for geom in roads_gdf.geometry if geom.is_valid],
                out_shape=image_shape[:2],
                transform=geo_metadata['transform'],
                fill=0,
                dtype=np.uint8,
                all_touched=True
            )
            
            road_pixels = np.sum(mask > 0)
            total_pixels = mask.size
            coverage = (road_pixels / total_pixels) * 100
            print(f"Road mask created: {road_pixels:,} pixels ({coverage:.2f}%)")
            
            return mask
            
        except Exception as e:
            print(f"Error creating OSM road mask: {e}")
            return np.zeros(image_shape[:2], dtype=np.uint8)
    
    def create_building_mask_from_osm(self, geo_metadata: Dict, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create building mask from OSM data
        
        Args:
            geo_metadata: Dictionary with 'bounds', 'crs', 'transform' keys
            image_shape: (height, width) of the target image
            
        Returns:
            Binary mask array with buildings as 1, background as 0
        """
        try:
            # Transform bounds to WGS84 for OSM
            bounds = geo_metadata['bounds']
            native_crs = geo_metadata['crs']
            
            wgs84_bounds = transform_bounds(
                native_crs, 'EPSG:4326',
                bounds.left, bounds.bottom, bounds.right, bounds.top
            )
            
            # Get buildings from OSM
            buildings_gdf = self.building_extractor.get_buildings_from_bbox(wgs84_bounds)
            
            if buildings_gdf.empty:
                return np.zeros(image_shape[:2], dtype=np.uint8)
            
            # Reproject to native CRS if needed
            if str(buildings_gdf.crs) != str(native_crs):
                buildings_gdf = buildings_gdf.to_crs(native_crs)
            
            # Apply buffer to buildings
            buffer_distance = self._get_building_buffer_distance(native_crs)
            buildings_gdf['geometry'] = buildings_gdf.geometry.buffer(buffer_distance)
            
            # Rasterize buildings
            mask = rasterize(
                [(mapping(geom), 1) for geom in buildings_gdf.geometry if geom.is_valid],
                out_shape=image_shape[:2],
                transform=geo_metadata['transform'],
                fill=0,
                dtype=np.uint8,
                all_touched=True
            )
            
            building_pixels = np.sum(mask > 0)
            total_pixels = mask.size
            coverage = (building_pixels / total_pixels) * 100
            print(f"Building mask created: {building_pixels:,} pixels ({coverage:.2f}%)")
            
            return mask
            
        except Exception as e:
            print(f"Error creating OSM building mask: {e}")
            return np.zeros(image_shape[:2], dtype=np.uint8)
    
    def create_multiclass_mask_from_osm(self, geo_metadata: Dict, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create multiclass mask with roads=1, buildings=2, background=0
        
        Args:
            geo_metadata: Dictionary with 'bounds', 'crs', 'transform' keys
            image_shape: (height, width) of the target image
            
        Returns:
            Multiclass mask array
        """
        road_mask = self.create_road_mask_from_osm(geo_metadata, image_shape)
        building_mask = self.create_building_mask_from_osm(geo_metadata, image_shape)
        
        # Combine masks: roads=1, buildings=2, overlaps=1 (roads have priority)
        combined_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        combined_mask[building_mask > 0] = 2
        combined_mask[road_mask > 0] = 1
        
        return combined_mask
    
    def _get_road_buffer_distance(self, crs) -> float:
        """Get appropriate buffer distance for roads based on CRS"""
        osm_config = self.config.OSM_CONFIG
        
        if hasattr(crs, 'is_projected') and crs.is_projected:
            return osm_config.get('road_buffer_projected', 5)
        else:
            return osm_config.get('road_buffer_geographic', 0.00005)
    
    def _get_building_buffer_distance(self, crs) -> float:
        """Get appropriate buffer distance for buildings based on CRS"""
        osm_config = self.config.OSM_CONFIG
        
        if hasattr(crs, 'is_projected') and crs.is_projected:
            return osm_config.get('building_buffer_projected', 5)
        else:
            return osm_config.get('building_buffer_geographic', 0.00005)

# Utility functions for easy access
def get_osm_roads(bbox: Tuple[float, float, float, float], cache_dir: str = "osm_cache") -> gpd.GeoDataFrame:
    """
    Convenience function to get roads from OSM
    
    Args:
        bbox: (minx, miny, maxx, maxy) in WGS84 coordinates
        cache_dir: Directory for caching
        
    Returns:
        GeoDataFrame with road geometries
    """
    extractor = OSMRoadExtractor(cache_dir)
    return extractor.get_roads_from_bbox(bbox)

def get_osm_buildings(bbox: Tuple[float, float, float, float], cache_dir: str = "osm_cache_buildings") -> gpd.GeoDataFrame:
    """
    Convenience function to get buildings from OSM
    
    Args:
        bbox: (minx, miny, maxx, maxy) in WGS84 coordinates
        cache_dir: Directory for caching
        
    Returns:
        GeoDataFrame with building geometries
    """
    extractor = OSMBuildingExtractor(cache_dir)
    return extractor.get_buildings_from_bbox(bbox)

def create_osm_road_mask(geo_metadata: Dict, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convenience function to create road mask from OSM
    
    Args:
        geo_metadata: Dictionary with 'bounds', 'crs', 'transform' keys
        image_shape: (height, width) of the target image
        
    Returns:
        Binary mask array with roads
    """
    generator = OSMMaskGenerator()
    return generator.create_road_mask_from_osm(geo_metadata, image_shape)

def create_osm_building_mask(geo_metadata: Dict, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convenience function to create building mask from OSM
    
    Args:
        geo_metadata: Dictionary with 'bounds', 'crs', 'transform' keys
        image_shape: (height, width) of the target image
        
    Returns:
        Binary mask array with buildings
    """
    generator = OSMMaskGenerator()
    return generator.create_building_mask_from_osm(geo_metadata, image_shape)