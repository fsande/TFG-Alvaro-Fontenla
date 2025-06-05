import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Softmax

class PositionAttentionModule(nn.Module):
    """Position Attention Module (PAM) según el paper"""
    
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate Q, K, V
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)
        
        # Attention computation
        attention = torch.bmm(query, key)
        attention = self.softmax(attention)
        
        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Residual connection with learnable parameter
        out = self.gamma * out + x
        return out

class ChannelAttentionModule(nn.Module):
    """Channel Attention Module (CAM) según el paper"""
    
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Reshape for matrix multiplication
        query = x.view(batch_size, channels, -1)
        key = x.view(batch_size, channels, -1).permute(0, 2, 1)
        value = x.view(batch_size, channels, -1)
        
        # Channel attention computation
        attention = torch.bmm(query, key)
        attention_max = torch.max(attention, -1, keepdim=True)[0].expand_as(attention)
        attention = attention - attention_max
        attention = self.softmax(attention)
        
        # Apply attention
        out = torch.bmm(attention, value)
        out = out.view(batch_size, channels, height, width)
        
        # Residual connection
        out = self.gamma * out + x
        return out

class DualAttentionModule(nn.Module):
    """Dual Attention Module combinando PAM y CAM"""
    
    def __init__(self, in_channels):
        super().__init__()
        self.pam = PositionAttentionModule(in_channels)
        self.cam = ChannelAttentionModule(in_channels)
        
    def forward(self, x):
        pam_out = self.pam(x)
        cam_out = self.cam(x)
        return pam_out + cam_out

class CrossSourceFeatureRecalibrationModule(nn.Module):
    """Cross-source Feature Recalibration Module (CFRM)"""
    
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
        
        # Squeeze operation for both features
        rgb_squeezed = self.squeeze(rgb_feat).view(batch_size, self.channels)
        msi_squeezed = self.squeeze(msi_feat).view(batch_size, self.channels)
        
        # Excitation operation
        rgb_excited = self.excitation(rgb_squeezed).view(batch_size, self.channels, 1, 1)
        msi_excited = self.excitation(msi_squeezed).view(batch_size, self.channels, 1, 1)
        
        # Recalibration
        rgb_calibrated = rgb_excited * rgb_feat
        msi_calibrated = msi_excited * msi_feat
        
        # Fusion
        fused = rgb_calibrated + msi_calibrated
        return fused

class CrossSourceFeatureFusionModule(nn.Module):
    """Cross-source Feature Fusion Module (CFFM)"""
    
    def __init__(self, rgb_channels=3, msi_channels=8, output_channels=64):
        super().__init__()
        
        # Separate feature extraction for RGB and MSI
        self.rgb_conv1 = nn.Sequential(
            nn.Conv2d(rgb_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
        self.msi_conv1 = nn.Sequential(
            nn.Conv2d(msi_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_channels, 3, padding=1),
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
        
        # CFRM modules for each scale
        self.cfrm1 = CrossSourceFeatureRecalibrationModule(output_channels)
        self.cfrm2 = CrossSourceFeatureRecalibrationModule(output_channels * 2)
        self.cfrm3 = CrossSourceFeatureRecalibrationModule(output_channels * 4)
        
    def forward(self, rgb, msi):
        # Upsample MSI to match RGB spatial resolution
        msi_upsampled = F.interpolate(msi, size=rgb.shape[-2:], mode='bilinear', align_corners=False)
        
        # Scale 1
        rgb_feat1 = self.rgb_conv1(rgb)
        msi_feat1 = self.msi_conv1(msi_upsampled)
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

class BasicBlock(nn.Module):
    """Basic Block para HRNet corregido"""
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        # Solo sumar si las dimensiones coinciden
        if residual.shape == out.shape:
            out += residual
        else:
            # Si no coinciden, usar solo la salida (sin residual)
            pass
            
        out = self.relu(out)
        
        return out

class SmoothFusionModule(nn.Module):
    """Smooth Fusion Module (SFM)"""
    
    def __init__(self, high_channels, low_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(low_channels, high_channels, 3, padding=1),
            nn.BatchNorm2d(high_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, high_feat, low_feat):
        # Upsample low resolution feature
        low_upsampled = F.interpolate(low_feat, size=high_feat.shape[-2:], 
                                    mode='bilinear', align_corners=False)
        low_upsampled = self.conv(low_upsampled)
        
        # Element-wise addition
        return high_feat + low_upsampled

class MultiscaleSemanticAggregationDecoder(nn.Module):
    """Multiscale Semantic Aggregation Decoder (MSAD)"""
    
    def __init__(self, feature_channels=[64, 128, 256, 512], num_classes=2):
        super().__init__()
        
        # Dual attention for the lowest resolution feature
        self.dual_attention = DualAttentionModule(feature_channels[-1])
        
        # Smooth fusion modules
        self.sfm3 = SmoothFusionModule(feature_channels[2], feature_channels[3])
        self.sfm2 = SmoothFusionModule(feature_channels[1], feature_channels[2])
        self.sfm1 = SmoothFusionModule(feature_channels[0], feature_channels[1])
        
        # Final classifier with upsampling
        self.classifier = nn.Sequential(
            nn.Conv2d(sum(feature_channels), 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
        
        # Upsampling final para recuperar resolución original
        self.final_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
    def forward(self, features):
        feat1, feat2, feat3, feat4 = features
        
        # Apply dual attention to the lowest resolution feature
        feat4_att = self.dual_attention(feat4)
        
        # Progressive fusion
        feat3_fused = self.sfm3(feat3, feat4_att)
        feat2_fused = self.sfm2(feat2, feat3_fused)
        feat1_fused = self.sfm1(feat1, feat2_fused)
        
        # Upsample all features to feat1 size (should be 1/4 of original)
        size = feat1.shape[-2:]
        feat1_up = feat1_fused
        feat2_up = F.interpolate(feat2_fused, size=size, mode='bilinear', align_corners=False)
        feat3_up = F.interpolate(feat3_fused, size=size, mode='bilinear', align_corners=False)
        feat4_up = F.interpolate(feat4_att, size=size, mode='bilinear', align_corners=False)
        
        # Concatenate multi-scale features
        concat_feat = torch.cat([feat1_up, feat2_up, feat3_up, feat4_up], dim=1)
        
        # Final classification
        out = self.classifier(concat_feat)
        
        # Upsample to original input size (4x upsampling desde stem que reduce 4x)
        out = self.final_upsample(out)
        
        return out

class SimpleHRNet(nn.Module):
    """HRNet simplificado"""
    
    def __init__(self, input_channels=64, width=32):
        super().__init__()
        
        # Stem - reduce spatial size but keep channels manageable
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale branches con dimensiones correctas
        self.branch1 = self._make_branch(width, width, 4)
        self.branch2 = self._make_branch(width * 2, width * 2, 4)
        self.branch3 = self._make_branch(width * 4, width * 4, 4)
        self.branch4 = self._make_branch(width * 8, width * 8, 4)
        
        # Transitions con dimensiones correctas
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
        # Primera capa puede cambiar canales
        if inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            downsample = None
            
        layers.append(BasicBlock(inplanes, planes, downsample=downsample))
        
        # Resto de capas mantienen dimensiones
        for i in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        # Stem: reduce spatial resolution
        x = self.stem(x)  # Input: 64 channels -> width channels
        
        # Branch 1: mantiene resolución del stem
        x1 = self.branch1(x)  # width channels
        
        # Branch 2: reduce resolución, aumenta canales
        x2 = self.transition1(x1)  # width -> width*2 channels, /2 spatial
        x2 = self.branch2(x2)
        
        # Branch 3: reduce resolución, aumenta canales  
        x3 = self.transition2(x2)  # width*2 -> width*4 channels, /2 spatial
        x3 = self.branch3(x3)
        
        # Branch 4: reduce resolución, aumenta canales
        x4 = self.transition3(x3)  # width*4 -> width*8 channels, /2 spatial
        x4 = self.branch4(x4)
        
        return [x1, x2, x3, x4]

class MSFANet(nn.Module):
    """MSFANet: Multiscale Fusion Attention Network"""
    
    def __init__(self, rgb_channels=3, msi_channels=8, num_classes=2, width=32):
        super().__init__()
        
        # Cross-source Feature Fusion Module
        self.cffm = CrossSourceFeatureFusionModule(rgb_channels, msi_channels, 64)
        
        # HRNet Encoder
        self.hrnet = SimpleHRNet(input_channels=64, width=width)
        
        # Multiscale Semantic Aggregation Decoder - dimensiones corregidas
        feature_channels = [width, width * 2, width * 4, width * 8]
        self.msad = MultiscaleSemanticAggregationDecoder(feature_channels, num_classes)
        
    def forward(self, rgb, msi):
        # Cross-source feature fusion
        fused_features = self.cffm(rgb, msi)
        
        # Use the first scale for HRNet input (should be 64 channels)
        hrnet_input = fused_features[0]
        
        # HRNet encoding
        hrnet_features = self.hrnet(hrnet_input)
        
        # Multiscale semantic aggregation
        output = self.msad(hrnet_features)
        
        return output

def get_msfanet(num_classes=2, width=32):
    """Factory function para crear MSFANet"""
    return MSFANet(
        rgb_channels=3,
        msi_channels=8, 
        num_classes=num_classes,
        width=width
    )

if __name__ == "__main__":
    model = get_msfanet(num_classes=2, width=32)
    
    rgb = torch.randn(2, 3, 512, 512)
    msi = torch.randn(2, 8, 128, 128)
    
    print("Modelo MSFANet creado")
    print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
    
    with torch.no_grad():
        output = model(rgb, msi)
        print(f"Output shape: {output.shape}")
        
    print("Test del modelo completado exitosamente")
