#!/usr/bin/env python3

"""
MSFANet model implementation
"""

import torch
import torch.nn as nn
from .hrnet import HRNet
from .multimodal_processor import MultiModalProcessor

class MSFANet(nn.Module):
    """Implementation of MSFANet"""
    
    def __init__(self, num_classes=2, rgb_channels=3, msi_channels=8, width=32, **kwargs):
        super().__init__()
        
        # Multi-modal processor for RGB and MSI fusion
        self.multimodal_processor = MultiModalProcessor(
            rgb_channels=rgb_channels, 
            msi_channels=msi_channels
        )
        
        # HRNet encoder (input channels = 64 from multimodal processor)
        self.hrnet = HRNet(num_classes=num_classes, rgb_channels=64, width=width)
        
        # Multiscale decoder with attention
        self.attention = nn.MultiheadAttention(width*4, 8, batch_first=True)
        
    def forward(self, rgb, msi):
        # Multi-modal feature fusion using MultiModalProcessor
        fused_features = self.multimodal_processor(rgb, msi)
        
        # HRNet processing
        output = self.hrnet(fused_features)
        
        return output

def create_msfanet(**kwargs):
    """Factory function to create MSFANet instance"""
    return MSFANet(**kwargs)