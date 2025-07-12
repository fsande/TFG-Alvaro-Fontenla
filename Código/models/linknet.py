#!/usr/bin/env python3

"""
LinkNet model implementation
"""

import torch
import torch.nn as nn

class LinkNet(nn.Module):
    """Implementation of LinkNet"""
    
    def __init__(self, num_classes=2, rgb_channels=3, msi_channels=8, **kwargs):
        super().__init__()
        
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(rgb_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            self._make_layer(64, 128, 2, stride=2),
        )
        
        self.encoder3 = self._make_layer(128, 256, 2, stride=2)
        self.encoder4 = self._make_layer(256, 512, 2, stride=2)
        
        # Decoder with skip connections
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.final_conv = nn.Conv2d(32, num_classes, 1)
        
    def _make_layer(self, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(planes, planes, 3, padding=1))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, rgb, msi=None):
        # Encoder
        x1 = self.encoder1(rgb)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        
        # Decoder
        d4 = self.decoder4(x4)
        d3 = self.decoder3(d4)
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2)
        
        output = self.final_conv(d1)
        
        # Ensure dimensions match the input
        if output.shape[-2:] != rgb.shape[-2:]:
            output = torch.nn.functional.interpolate(
                output, size=rgb.shape[-2:], mode='bilinear', align_corners=False
            )
        
        return output

def create_linknet(**kwargs):
    """Factory function to create LinkNet instance"""
    return LinkNet(**kwargs)