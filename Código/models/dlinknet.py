#!/usr/bin/env python3

"""
D-LinkNet model implementation
"""

import torch
import torch.nn as nn
from .linknet import LinkNet


class DLinkNet(LinkNet):
    """D-LinkNet with dilated convolutions and corrected dimensions"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Add dilated convolutions in the center
        self.dilated_center = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=4, dilation=4),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, rgb, msi=None):
        # Encoder
        x1 = self.encoder1(rgb)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        
        # Dilated center
        x4 = self.dilated_center(x4)
        
        # Decoder
        d4 = self.decoder4(x4)
        d3 = self.decoder3(d4)
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2)
        
        output = self.final_conv(d1)
        
        # Ensure correct dimensions
        if output.shape[-2:] != rgb.shape[-2:]:
            output = torch.nn.functional.interpolate(
                output, size=rgb.shape[-2:], mode='bilinear', align_corners=False
            )
        
        return output

def create_dlinknet(**kwargs):
    """Factory function to create DLinkNet instance"""
    return DLinkNet(**kwargs)