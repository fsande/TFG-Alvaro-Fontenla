#!/usr/bin/env python3

"""
Sentinel-2 Dataset generator
Provides PyTorch dataset interface for processed Sentinel-2 data
"""

import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

class Sentinel2Dataset(Dataset):
    """Dataset for processed Sentinel-2 data"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.samples = self._load_samples()
        print(f"Sentinel-2 dataset loaded: {len(self.samples)} samples")
        
    def _load_samples(self):
        """Load all available samples from the processed data directory"""
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
        """Get a sample from the dataset"""
        sample = self.samples[idx]
        
        # Load and normalize
        rgb = np.load(sample['rgb']).astype(np.float32) / 255.0
        ms = np.load(sample['ms']).astype(np.float32) / 255.0
        mask = np.load(sample['mask']).astype(np.int64)
        
        # Convert to tensors
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)  # CHW
        ms = torch.from_numpy(ms).permute(2, 0, 1)    # CHW
        mask = torch.from_numpy(mask)                 # HW
        
        return {'rgb': rgb, 'ms': ms, 'mask': mask}
    
    def class_distribution(self):
        """Get class distribution"""
        class_counts = {}
        for sample in self.samples:
            mask = np.load(sample['mask']).astype(np.int64)
            unique_values = np.unique(mask)
            for value in unique_values:
                if value not in class_counts:
                    class_counts[value] = 0
                class_counts[value] += 1
        return class_counts