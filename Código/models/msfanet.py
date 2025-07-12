#!/usr/bin/env python3

"""
MSFANet model implementation
"""

import torch
import torch.nn as nn
from .hrnet import HRNet

class MSFANet(nn.Module):
    """Implementation of MSFANet"""
    
    def __init__(self, num_classes=2, rgb_channels=3, msi_channels=8, width=32, **kwargs):
        super().__init__()
        
        # Cross-source feature fusion
        self.rgb_conv = nn.Conv2d(rgb_channels, 64, 3, padding=1)
        self.msi_conv = nn.Conv2d(msi_channels, 64, 3, padding=1)
        self.fusion_conv = nn.Conv2d(128, 64, 1)
        
        # HRNet encoder
        self.hrnet = HRNet(num_classes=num_classes, rgb_channels=64, width=width)
        
        # Multiscale decoder with attention
        self.attention = nn.MultiheadAttention(width*4, 8, batch_first=True)
        
    def forward(self, rgb, msi):
        # Cross-source fusion
        if msi.shape[-2:] != rgb.shape[-2:]:
            msi = torch.nn.functional.interpolate(msi, size=rgb.shape[-2:], mode='bilinear', align_corners=False)
        
        rgb_feat = self.rgb_conv(rgb)
        msi_feat = self.msi_conv(msi)
        
        fused = torch.cat([rgb_feat, msi_feat], dim=1)
        fused = self.fusion_conv(fused)
        
        # HRNet processing
        output = self.hrnet(fused)
        
        return output

def create_msfanet(**kwargs):
    """Factory function to create MSFANet instance"""
    return MSFANet(**kwargs)