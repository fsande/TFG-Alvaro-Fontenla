#!/usr/bin/env python3

"""
CCNet model implementation
"""

import torch
import torch.nn as nn
from .hrnet import HRNet

class CCNet(HRNet):
    """Implementation of CCNet"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Criss-cross attention
        self.cc_attention = nn.Sequential(
            nn.Conv2d(kwargs.get('width', 32) * 4, kwargs.get('width', 32), 1),
            nn.BatchNorm2d(kwargs.get('width', 32)),
            nn.ReLU(inplace=True),
        )


def create_ccnet(**kwargs):
    """Factory function to create CCNet instance"""
    return CCNet(**kwargs)