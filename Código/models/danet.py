#!/usr/bin/env python3

"""
DANet model implementation
"""

import torch
import torch.nn as nn
from .hrnet import HRNet

class DANet(HRNet):
    """Implementation of DANet"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        width = kwargs.get('width', 32)
        total_channels = width + width*2 + width*4  # Total after concatenation
        
        # Dual attention
        # Ensure embed_dim is divisible by num_heads
        embed_dim = ((total_channels // 8) // 8) * 8  # Make divisible by 8
        if embed_dim == 0:
            embed_dim = 8
            
        self.position_attention = nn.Sequential(
            nn.Conv2d(total_channels, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, total_channels, 1),
        )
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_channels, total_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels // 4, total_channels, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, rgb, msi=None):
        x = self.stem(rgb)
        
        # Multi-scale processing
        x1 = self.stage1(x)
        x2 = self.transition1(x1)
        x2 = self.stage2(x2)
        x3 = self.transition2(x2)
        x3 = self.stage3(x3)
        
        # Upsample and concatenate
        size = x1.shape[-2:]
        x2_up = torch.nn.functional.interpolate(x2, size=size, mode='bilinear', align_corners=False)
        x3_up = torch.nn.functional.interpolate(x3, size=size, mode='bilinear', align_corners=False)
        
        x_concat = torch.cat([x1, x2_up, x3_up], dim=1)
        
        # Apply dual attention
        pos_att = self.position_attention(x_concat)
        chan_att = self.channel_attention(x_concat)
        
        x_att = x_concat + pos_att + chan_att * x_concat
        
        x = self.final_layer(x_att)
        
        # Upsample to original size
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        return x

def create_danet(**kwargs):
    """Factory function to create DANet instance"""
    return DANet(**kwargs)