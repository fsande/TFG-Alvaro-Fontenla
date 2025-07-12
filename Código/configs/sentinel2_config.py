#!/usr/bin/env python3

"""
Configuration for Sentinel-2 data processing
Includes band configurations, resolutions and specific parameters
"""

import os
import yaml

class Sentinel2Config:
    """
    Configuration for Sentinel-2 bands and parameters
    """
    
    def __init__(self, config_path=None):
        """
        Initialize configuration loading from YAML config file
        
        Args:
            config_path: Path to YAML config file
        """
        # Load config from YAML file
        self.yaml_config = self._load_yaml_config(config_path)
        
        # Initialize default configurations
        self._init_default_configs()
        
        # Apply config from YAML file
        self._apply_yaml_overrides()
    
    def _load_yaml_config(self, config_path=None):
        """
        Load config from YAML file
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            Dictionary with loaded config
        """
        # Search for configuration file
        if config_path is None:
            # Search in common locations
            possible_paths = [
                'config.yaml',
                'configs/config.yaml'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
        
        if config_path is None or not os.path.exists(config_path):
            print(f"Warning: Config file not found at {config_path}")
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                print(f"Config loaded from: {config_path}")
                return config if config else {}
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")
            return {}
    
    def _init_default_configs(self):
        """Initialize default configs specific to Sentinel-2"""
        
        # Bands by spatial resolution
        self.BANDS_10M = ['B02', 'B03', 'B04', 'B08']  # Blue, Green, Red, NIR
        self.BANDS_20M = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']  # Red edges, SWIR
        
        # RGB configuration (using 10M bands)
        self.RGB_BANDS = {
            'red': 'B04',    # 665 nm
            'green': 'B03',  # 560 nm  
            'blue': 'B02'    # 490 nm
        }
        
        # Multispectral configuration (all useful bands)
        self.MS_BANDS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
        
        # Target resolution for processing
        self.TARGET_RESOLUTION = 10
        
        # Normalization parameters
        self.NORMALIZATION = {
            'rgb_divisor': 3000.0,  # Normalization factor for RGB
            'ms_divisor': 3000.0,   # Normalization factor for multispectral
            'clip_max': 255,        # Maximum value after clipping
            'clip_min': 0          # Minimum value after clipping
        }
        
        # Config for OSM buffers
        self.OSM_CONFIG = {
            'road_buffer_projected': 5,      # Buffer for roads in projected CRS (meters)
            'road_buffer_geographic': 0.00005,  # Buffer for roads in geographic CRS (degrees)
            'building_buffer_projected': 5,   # Buffer for buildings in projected CRS (meters)
            'building_buffer_geographic': 0.00005,  # Buffer for buildings in geographic CRS (degrees)
            'max_area_degrees': 1.0,        # Maximum area for OSM queries (degrees squared)
            'fallback_margin': 0.1          # Margin to reduce area in large queries (degrees)
        }
        
        # Training config (default values)
        self.TRAINING_CONFIG = {
            'epochs': 100,
            'batch_size': 4,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'poly_power': 0.9,
            'use_class_weights': True,
            'num_workers': 4,
            'train_split': 0.8,
            'val_split': 0.2,
            'val_frequency': 5,
            'save_frequency': 10,
            'device': 'cuda'
        }
        
        # Model config
        self.MODEL_CONFIG = {
            'num_classes': 2,
            'model_width': 32,
            'rgb_channels': 3,
            'msi_channels': 9
        }
        
        # Directory config
        self.DIRS_CONFIG = {
            'data_dir': 'processed_data',
            'log_dir': 'logs',
            'checkpoint_dir': 'checkpoints'
        }
    
    def _apply_yaml_overrides(self):
        """Apply config from YAML overriding default values"""
        
        # Override training configuration
        for key in self.TRAINING_CONFIG:
            if key in self.yaml_config:
                self.TRAINING_CONFIG[key] = self.yaml_config[key]
        
        # Override model configuration
        for key in self.MODEL_CONFIG:
            if key in self.yaml_config:
                self.MODEL_CONFIG[key] = self.yaml_config[key]
        
        # Override directory configuration
        for key in self.DIRS_CONFIG:
            if key in self.yaml_config:
                self.DIRS_CONFIG[key] = self.yaml_config[key]
        
        # Apply specific Sentinel-2 config from YAML if exists
        if 'sentinel2' in self.yaml_config:
            s2_config = self.yaml_config['sentinel2']
            
            # Update OSM configuration
            if 'osm_config' in s2_config:
                self.OSM_CONFIG.update(s2_config['osm_config'])
            
            # Update normalization configuration
            if 'normalization' in s2_config:
                self.NORMALIZATION.update(s2_config['normalization'])
            
            # Update target resolution
            if 'target_resolution' in s2_config:
                self.TARGET_RESOLUTION = s2_config['target_resolution']
    
    def get_band_by_resolution(self, resolution):
        """
        Get bands by specific resolution
        
        Args:
            resolution: Desired resolution (10, 20)
            
        Returns:
            List of bands with that resolution
        """
        if resolution == 10:
            return self.BANDS_10M
        elif resolution == 20:
            return self.BANDS_20M
        else:
            raise ValueError(f"Invalid resolution: {resolution}. Use 10, 20.")
    
    def get_rgb_band_order(self):
        """
        Get RGB band order
        
        Returns:
            List with bands in order [R, G, B]
        """
        return [self.RGB_BANDS['red'], self.RGB_BANDS['green'], self.RGB_BANDS['blue']]
    
    def get_processing_config(self):
        """
        Get complete configuration for processing
        
        Returns:
            Dictionary with all configuration
        """
        return {
            'bands_10m': self.BANDS_10M,
            'bands_20m': self.BANDS_20M,
            'rgb_bands': self.RGB_BANDS,
            'ms_bands': self.MS_BANDS,
            'target_resolution': self.TARGET_RESOLUTION,
            'normalization': self.NORMALIZATION,
            'osm_config': self.OSM_CONFIG,
            'training_config': self.TRAINING_CONFIG,
            'model_config': self.MODEL_CONFIG,
            'dirs_config': self.DIRS_CONFIG,
            'yaml_config': self.yaml_config
        }
    
    def get_training_config(self):
        """Get training configuration"""
        return self.TRAINING_CONFIG.copy()
    
    def get_model_config(self):
        """Get model configuration"""
        return self.MODEL_CONFIG.copy()
    
    def get_dirs_config(self):
        """Get directory configuration"""
        return self.DIRS_CONFIG.copy()
    
    def get_config_value(self, key, default=None):
        """
        Get specific configuration value
        
        Args:
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        return self.yaml_config.get(key, default)
    
    def print_config(self):
        """Print configuration summary"""
        print("=" * 70)
        print("SENTINEL-2 CONFIGURATION")
        print("=" * 70)
        
        # Band configuration
        print(f"Target resolution: {self.TARGET_RESOLUTION}m")
        print(f"10m bands: {self.BANDS_10M}")
        print(f"20m bands: {self.BANDS_20M}")
        print(f"RGB configuration: {self.RGB_BANDS}")
        print(f"Multispectral bands: {self.MS_BANDS}")
        
        # Model configuration
        print(f"\nMODEL:")
        print(f"Classes: {self.MODEL_CONFIG['num_classes']}")
        print(f"Model width: {self.MODEL_CONFIG['model_width']}")
        print(f"RGB channels: {self.MODEL_CONFIG['rgb_channels']}")
        print(f"MSI channels: {self.MODEL_CONFIG['msi_channels']}")
        
        # Training configuration
        print(f"\nTRAINING:")
        print(f"Epochs: {self.TRAINING_CONFIG['epochs']}")
        print(f"Batch size: {self.TRAINING_CONFIG['batch_size']}")
        print(f"Learning rate: {self.TRAINING_CONFIG['learning_rate']}")
        print(f"Weight decay: {self.TRAINING_CONFIG['weight_decay']}")
        print(f"Device: {self.TRAINING_CONFIG['device']}")
        
        # Directory configuration
        print(f"\nDIRECTORIES:")
        print(f"Data: {self.DIRS_CONFIG['data_dir']}")
        print(f"Logs: {self.DIRS_CONFIG['log_dir']}")
        print(f"Checkpoints: {self.DIRS_CONFIG['checkpoint_dir']}")
        
        print("=" * 70)

_global_config = None

def get_global_config(config_path=None):
    """
    Get global configuration instance
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Sentinel2Config instance
    """
    global _global_config
    if _global_config is None:
        _global_config = Sentinel2Config(config_path)
    return _global_config

def reset_global_config():
    """Reset global configuration"""
    global _global_config
    _global_config = None