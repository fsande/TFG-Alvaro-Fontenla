#!/usr/bin/env python3

"""
Multi-modal input processor
Handles RGB and MSI data fusion for satellite imagery processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalProcessor(nn.Module):
    """Multi-modal input processor for RGB and MSI data"""
    
    def __init__(self, rgb_channels=3, msi_channels=10):
        super().__init__()
        self.rgb_conv = nn.Conv2d(rgb_channels, 64, 3, padding=1)
        self.msi_conv = nn.Conv2d(msi_channels, 64, 3, padding=1)
        self.fusion_conv = nn.Conv2d(128, 64, 1)
    
    def forward(self, rgb, msi):
        # Process RGB and MSI independently
        rgb_feat = F.relu(self.rgb_conv(rgb))
        msi_feat = F.relu(self.msi_conv(msi))
        
        # Spatial alignment if necessary
        if msi.shape[-2:] != rgb.shape[-2:]:
            msi_feat = F.interpolate(msi_feat, size=rgb.shape[-2:],
                                   mode='bilinear', align_corners=False)
        
        # Feature fusion
        fused = torch.cat([rgb_feat, msi_feat], dim=1)
        return F.relu(self.fusion_conv(fused))


def create_multimodal_processor(rgb_channels=3, msi_channels=9):
    """Factory function to create MultiModalProcessor instance"""
    return MultiModalProcessor(rgb_channels=rgb_channels, msi_channels=msi_channels)