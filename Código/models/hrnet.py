#!/usr/bin/env python3

"""
HRNet model implementation
"""

import torch
import torch.nn as nn


class HRNet(nn.Module):
    """Implementation of HRNet"""
    
    def __init__(self, num_classes=2, rgb_channels=3, width=32, **kwargs):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(rgb_channels, width, 3, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        
        # Multi-scale branches
        self.stage1 = self._make_stage(width, width, 4)
        self.stage2 = self._make_stage(width * 2, width * 2, 4)
        self.stage3 = self._make_stage(width * 4, width * 4, 4)
        
        # Transitions
        self.transition1 = nn.Conv2d(width, width * 2, 3, stride=2, padding=1)
        self.transition2 = nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1)
        
        # Final layers
        self.final_layer = nn.Conv2d(width + width*2 + width*4, num_classes, 1)
        
    def _make_stage(self, inplanes, planes, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(nn.Conv2d(inplanes, planes, 3, padding=1))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            inplanes = planes
        return nn.Sequential(*layers)
    
    def forward(self, rgb, msi=None):
        x = self.stem(rgb)
        
        # Stage 1
        x1 = self.stage1(x)
        
        # Stage 2
        x2 = self.transition1(x1)
        x2 = self.stage2(x2)
        
        # Stage 3
        x3 = self.transition2(x2)
        x3 = self.stage3(x3)
        
        # Upsample and concatenate
        size = x1.shape[-2:]
        x2_up = torch.nn.functional.interpolate(x2, size=size, mode='bilinear', align_corners=False)
        x3_up = torch.nn.functional.interpolate(x3, size=size, mode='bilinear', align_corners=False)
        
        x_concat = torch.cat([x1, x2_up, x3_up], dim=1)
        x = self.final_layer(x_concat)
        
        # Upsample to original size
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        return x

def create_hrnet(**kwargs):
    """Factory function to create HRNet instance"""
    return HRNet(**kwargs)