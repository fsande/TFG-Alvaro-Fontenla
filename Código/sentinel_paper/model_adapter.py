#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np

try:
    from msfanet_model import (
        MSFANet, CrossSourceFeatureFusionModule, 
        SimpleHRNet, MultiscaleSemanticAggregationDecoder
    )
    ORIGINAL_MODEL_AVAILABLE = True
    print("Modelo original MSFANet disponible")
except ImportError as e:
    print(f"Modelo original no disponible: {e}")
    ORIGINAL_MODEL_AVAILABLE = False

try:
    from simple_msfanet import get_simple_msfanet as get_simple_msfanet_original
    SIMPLE_MODEL_AVAILABLE = True
    print("Modelo simple MSFANet disponible")
except ImportError as e:
    print(f"Modelo simple no disponible: {e}")
    SIMPLE_MODEL_AVAILABLE = False

class CrossSourceFeatureFusionModuleAdapted(nn.Module):
    """CFFM adaptado para cualquier número de canales MSI"""
    
    def __init__(self, rgb_channels=3, msi_channels=8, output_channels=64):
        super().__init__()
        
        print(f"Adaptando CFFM para RGB:{rgb_channels}, MSI:{msi_channels}")
        
        # Feature extraction para RGB (mismo que original)
        self.rgb_conv1 = nn.Sequential(
            nn.Conv2d(rgb_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
        # Feature extraction para MSI - ADAPTADO para número variable de canales
        # Escalar canales intermedios según número de bandas MSI
        intermediate_channels = max(32, min(64, msi_channels * 4))
        
        self.msi_conv1 = nn.Sequential(
            nn.Conv2d(msi_channels, intermediate_channels, 3, padding=1),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale feature extraction
        self.rgb_conv2 = nn.Sequential(
            nn.Conv2d(output_channels, output_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(output_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.msi_conv2 = nn.Sequential(
            nn.Conv2d(output_channels, output_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(output_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.rgb_conv3 = nn.Sequential(
            nn.Conv2d(output_channels * 2, output_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(output_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        self.msi_conv3 = nn.Sequential(
            nn.Conv2d(output_channels * 2, output_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(output_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        # Importar CFRM del modelo original
        try:
            from msfanet_model import CrossSourceFeatureRecalibrationModule
            self.cfrm1 = CrossSourceFeatureRecalibrationModule(output_channels)
            self.cfrm2 = CrossSourceFeatureRecalibrationModule(output_channels * 2)
            self.cfrm3 = CrossSourceFeatureRecalibrationModule(output_channels * 4)
        except ImportError:
            # Implementación simple si no está disponible
            self.cfrm1 = SimpleCFRM(output_channels)
            self.cfrm2 = SimpleCFRM(output_channels * 2)
            self.cfrm3 = SimpleCFRM(output_channels * 4)
        
    def forward(self, rgb, msi):
        # Upsample MSI a resolución RGB si es necesario
        if msi.shape[-2:] != rgb.shape[-2:]:
            msi = torch.nn.functional.interpolate(
                msi, size=rgb.shape[-2:], mode='bilinear', align_corners=False
            )
        
        # Scale 1
        rgb_feat1 = self.rgb_conv1(rgb)
        msi_feat1 = self.msi_conv1(msi)
        fused_feat1 = self.cfrm1(rgb_feat1, msi_feat1)
        
        # Scale 2
        rgb_feat2 = self.rgb_conv2(rgb_feat1)
        msi_feat2 = self.msi_conv2(msi_feat1)
        fused_feat2 = self.cfrm2(rgb_feat2, msi_feat2)
        
        # Scale 3
        rgb_feat3 = self.rgb_conv3(rgb_feat2)
        msi_feat3 = self.msi_conv3(msi_feat2)
        fused_feat3 = self.cfrm3(rgb_feat3, msi_feat3)
        
        return [fused_feat1, fused_feat2, fused_feat3]

class SimpleCFRM(nn.Module):
    """CFRM simplificado si el original no está disponible"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 8, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, rgb_feat, msi_feat):
        batch_size = rgb_feat.size(0)
        
        # Squeeze operation para ambas características
        rgb_squeezed = self.squeeze(rgb_feat).view(batch_size, self.channels)
        msi_squeezed = self.squeeze(msi_feat).view(batch_size, self.channels)
        
        # Excitation operation
        rgb_excited = self.excitation(rgb_squeezed).view(batch_size, self.channels, 1, 1)
        msi_excited = self.excitation(msi_squeezed).view(batch_size, self.channels, 1, 1)
        
        # Recalibración
        rgb_calibrated = rgb_excited * rgb_feat
        msi_calibrated = msi_excited * msi_feat
        
        # Fusión
        fused = rgb_calibrated + msi_calibrated
        return fused

class MSFANetAdapted(nn.Module):
    """MSFANet adaptado para parámetros variables"""
    
    def __init__(self, rgb_channels=3, msi_channels=8, num_classes=2, width=32):
        super().__init__()
        
        print(f"Inicializando MSFANet Adaptado:")
        print(f"   RGB channels: {rgb_channels}")
        print(f"   MSI channels: {msi_channels}")
        print(f"   Classes: {num_classes}")
        print(f"   Width: {width}")
        
        # Cross-source Feature Fusion Module adaptado
        self.cffm = CrossSourceFeatureFusionModuleAdapted(
            rgb_channels=rgb_channels,
            msi_channels=msi_channels,
            output_channels=64
        )
        
        # HRNet Encoder (usar el original)
        if ORIGINAL_MODEL_AVAILABLE:
            try:
                self.hrnet = SimpleHRNet(input_channels=64, width=width)
            except:
                self.hrnet = SimpleHRNetAdapted(input_channels=64, width=width)
        else:
            self.hrnet = SimpleHRNetAdapted(input_channels=64, width=width)
        
        # Multiscale Semantic Aggregation Decoder
        feature_channels = [width, width * 2, width * 4, width * 8]
        if ORIGINAL_MODEL_AVAILABLE:
            try:
                self.msad = MultiscaleSemanticAggregationDecoder(feature_channels, num_classes)
            except:
                self.msad = SimpleMSAD(feature_channels, num_classes)
        else:
            self.msad = SimpleMSAD(feature_channels, num_classes)
        
    def forward(self, rgb, msi):
        # Cross-source feature fusion
        fused_features = self.cffm(rgb, msi)
        
        # HRNet encoding
        hrnet_features = self.hrnet(fused_features[0])
        
        # Multiscale semantic aggregation
        output = self.msad(hrnet_features)
        
        return output

class SimpleHRNetAdapted(nn.Module):
    """HRNet simplificado si el original no está disponible"""
    
    def __init__(self, input_channels=64, width=32):
        super().__init__()
        
        # Stem - reduce spatial size
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale branches
        self.branch1 = self._make_branch(width, width, 2)
        self.branch2 = self._make_branch(width * 2, width * 2, 2)
        self.branch3 = self._make_branch(width * 4, width * 4, 2)
        self.branch4 = self._make_branch(width * 8, width * 8, 2)
        
        # Transitions
        self.transition1 = nn.Sequential(
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True)
        )
        self.transition2 = nn.Sequential(
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True)
        )
        self.transition3 = nn.Sequential(
            nn.Conv2d(width * 4, width * 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 8),
            nn.ReLU(inplace=True)
        )
        
    def _make_branch(self, inplanes, planes, blocks):
        layers = []
        layers.append(nn.Conv2d(inplanes, planes, 3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(blocks - 1):
            layers.append(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Branch 1
        x1 = self.branch1(x)
        
        # Branch 2
        x2 = self.transition1(x1)
        x2 = self.branch2(x2)
        
        # Branch 3
        x3 = self.transition2(x2)
        x3 = self.branch3(x3)
        
        # Branch 4
        x4 = self.transition3(x3)
        x4 = self.branch4(x4)
        
        return [x1, x2, x3, x4]

class SimpleMSAD(nn.Module):
    """MSAD simplificado si el original no está disponible"""
    
    def __init__(self, feature_channels, num_classes):
        super().__init__()
        
        # Upsampling y fusión simple
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        
        # Classifier
        total_channels = sum(feature_channels)
        self.classifier = nn.Sequential(
            nn.Conv2d(total_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
        
        # Upsampling final
        self.final_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
    def forward(self, features):
        feat1, feat2, feat3, feat4 = features
        
        # Upsample todas las características al tamaño de feat1
        size = feat1.shape[-2:]
        feat1_up = feat1
        feat2_up = torch.nn.functional.interpolate(feat2, size=size, mode='bilinear', align_corners=False)
        feat3_up = torch.nn.functional.interpolate(feat3, size=size, mode='bilinear', align_corners=False)
        feat4_up = torch.nn.functional.interpolate(feat4, size=size, mode='bilinear', align_corners=False)
        
        # Concatenar características multi-escala
        concat_feat = torch.cat([feat1_up, feat2_up, feat3_up, feat4_up], dim=1)
        
        # Clasificación final
        out = self.classifier(concat_feat)
        
        # Upsample a resolución original
        out = self.final_upsample(out)
        
        return out

def get_msfanet_adapted(num_classes=2, width=32, rgb_channels=3, msi_channels=8):
    """Factory function adaptada para MSFANet con parámetros variables"""
    return MSFANetAdapted(
        rgb_channels=rgb_channels,
        msi_channels=msi_channels,
        num_classes=num_classes,
        width=width
    )

def get_msfanet(num_classes=2, width=32, **kwargs):
    """Wrapper para compatibilidad con código original"""
    rgb_channels = kwargs.get('rgb_channels', 3)
    msi_channels = kwargs.get('msi_channels', 8)
    
    if ORIGINAL_MODEL_AVAILABLE and rgb_channels == 3 and msi_channels == 8:
        # Usar modelo original si los parámetros son estándar
        try:
            from msfanet_model import get_msfanet as get_msfanet_original
            return get_msfanet_original(num_classes=num_classes, width=width)
        except:
            pass
    
    # Usar modelo adaptado
    return get_msfanet_adapted(num_classes, width, rgb_channels, msi_channels)

def get_simple_msfanet(num_classes=2, width=32, **kwargs):
    """Wrapper para compatibilidad con modelo simple"""
    rgb_channels = kwargs.get('rgb_channels', 3)
    msi_channels = kwargs.get('msi_channels', 8)
    
    if SIMPLE_MODEL_AVAILABLE and rgb_channels == 3 and msi_channels == 8:
        # Usar modelo simple original si los parámetros son estándar
        try:
            return get_simple_msfanet_original(num_classes=num_classes, width=width)
        except:
            pass
    
    # Usar modelo adaptado
    return get_msfanet_adapted(num_classes, width, rgb_channels, msi_channels)

# Test del adaptador
if __name__ == "__main__":
    print("Testeando Model Adapter...")
    
    # Test con parámetros Sentinel-2
    model = get_msfanet(num_classes=2, width=32, rgb_channels=3, msi_channels=10)
    
    # Test con datos dummy
    batch_size = 2
    height, width = 512, 512
    
    rgb = torch.randn(batch_size, 3, height, width)
    msi = torch.randn(batch_size, 10, height, width)
    
    print(f"\nInput shapes:")
    print(f"   RGB: {rgb.shape}")
    print(f"   MSI: {msi.shape}")
    
    # Forward pass
    with torch.no_grad():
        try:
            output = model(rgb, msi)
            print(f"\nForward pass exitoso")
            print(f"   Output shape: {output.shape}")
            
            # Contar parámetros
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   Parámetros: {total_params:,}")
            
        except Exception as e:
            print(f"Error en forward pass: {e}")
            import traceback
            traceback.print_exc()
