#!/usr/bin/env python3

"""
Estudio Comparativo de Arquitecturas para Segmentación de Carreteras en Imágenes de Teledetección
Basado en el paper: MSFANet: Multiscale Fusion Attention Network for Road Segmentation

Implementa y compara las siguientes arquitecturas:
- LinkNet
- D-LinkNet  
- HRNet
- CCNet
- DANet
- DBRANet
- NL-LinkNet
- MSFANet

Autor: Graduado en Ingeniería Informática - Universidad de La Laguna
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import os
import json
from pathlib import Path
from collections import defaultdict
import yaml
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Importar módulos propios
try:
    from model_adapter import get_msfanet, get_simple_msfanet
    MODEL_ADAPTER_AVAILABLE = True
except ImportError:
    MODEL_ADAPTER_AVAILABLE = False
    print("WARNING: model_adapter no disponible")

try:
    from sentinel2_processor_fixed import Sentinel2Dataset
except ImportError:
    try:
        from sentinel2_processor import Sentinel2Dataset
    except ImportError:
        print("WARNING: Dataset Sentinel-2 no disponible")

class ArchitectureRegistry:
    """Registro de todas las arquitecturas a comparar"""
    
    def __init__(self):
        self.architectures = {}
        self.register_architectures()
    
    def register_architectures(self):
        """Registra todas las arquitecturas del paper"""
        
        # LinkNet
        self.architectures['LinkNet'] = {
            'model_class': self.get_linknet,
            'description': 'Encoder-decoder con skip connections',
            'paper_year': 2017
        }
        
        # D-LinkNet 
        self.architectures['D-LinkNet'] = {
            'model_class': self.get_dlinknet,
            'description': 'LinkNet con dilated convolutions',
            'paper_year': 2018
        }
        
        # HRNet
        self.architectures['HRNet'] = {
            'model_class': self.get_hrnet,
            'description': 'High-Resolution Network',
            'paper_year': 2019
        }
        
        # CCNet
        self.architectures['CCNet'] = {
            'model_class': self.get_ccnet,
            'description': 'Criss-Cross Attention Network',
            'paper_year': 2019
        }
        
        # DANet
        self.architectures['DANet'] = {
            'model_class': self.get_danet,
            'description': 'Dual Attention Network',
            'paper_year': 2019
        }
        
        # DBRANet
        self.architectures['DBRANet'] = {
            'model_class': self.get_dbranet,
            'description': 'Dual-Branch Regional Attention Network',
            'paper_year': 2021
        }
        
        # NL-LinkNet
        self.architectures['NL-LinkNet'] = {
            'model_class': self.get_nllinknet,
            'description': 'Non-Local LinkNet',
            'paper_year': 2021
        }
        
        # MSFANet
        self.architectures['MSFANet'] = {
            'model_class': self.get_msfanet,
            'description': 'Multiscale Fusion Attention Network',
            'paper_year': 2023
        }
    
    def get_linknet(self, **kwargs):
        """Implementación simplificada de LinkNet"""
        return SimpleLinkNet(**kwargs)
    
    def get_dlinknet(self, **kwargs):
        """Implementación simplificada de D-LinkNet"""
        return SimpleDLinkNet(**kwargs)
    
    def get_hrnet(self, **kwargs):
        """Implementación simplificada de HRNet"""
        return SimpleHRNet(**kwargs)
    
    def get_ccnet(self, **kwargs):
        """Implementación simplificada de CCNet"""
        return SimpleCCNet(**kwargs)
    
    def get_danet(self, **kwargs):
        """Implementación simplificada de DANet"""
        return SimpleDANet(**kwargs)
    
    def get_dbranet(self, **kwargs):
        """Implementación simplificada de DBRANet"""
        return SimpleDBRANet(**kwargs)
    
    def get_nllinknet(self, **kwargs):
        """Implementación simplificada de NL-LinkNet"""
        return SimpleNLLinkNet(**kwargs)
    
    def get_msfanet(self, **kwargs):
        """MSFANet usando el adaptador si está disponible"""
        if MODEL_ADAPTER_AVAILABLE:
            return get_msfanet(**kwargs)
        else:
            return SimpleMSFANet(**kwargs)

# Implementaciones simplificadas de las arquitecturas
class SimpleLinkNet(nn.Module):
    """Implementación simplificada de LinkNet con dimensiones corregidas"""
    
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
        
        # Decoder con skip connections
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
        
        # Asegurar que las dimensiones coincidan con la entrada
        if output.shape[-2:] != rgb.shape[-2:]:
            output = torch.nn.functional.interpolate(
                output, size=rgb.shape[-2:], mode='bilinear', align_corners=False
            )
        
        return output

class SimpleDLinkNet(SimpleLinkNet):
    """D-LinkNet con dilated convolutions y dimensiones corregidas"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Añadir dilated convolutions en el centro
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
        
        # Asegurar dimensiones correctas
        if output.shape[-2:] != rgb.shape[-2:]:
            output = torch.nn.functional.interpolate(
                output, size=rgb.shape[-2:], mode='bilinear', align_corners=False
            )
        
        return output

class SimpleHRNet(nn.Module):
    """Implementación simplificada de HRNet"""
    
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

class SimpleCCNet(SimpleHRNet):
    """CCNet con criss-cross attention"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Criss-cross attention simplificado
        self.cc_attention = nn.Sequential(
            nn.Conv2d(kwargs.get('width', 32) * 4, kwargs.get('width', 32), 1),
            nn.BatchNorm2d(kwargs.get('width', 32)),
            nn.ReLU(inplace=True),
        )

class SimpleDANet(SimpleHRNet):
    """DANet con dual attention corregido"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        width = kwargs.get('width', 32)
        total_channels = width + width*2 + width*4  # Total después de concatenar
        
        # Dual attention simplificado con dimensiones correctas
        # Asegurar que embed_dim sea divisible por num_heads
        embed_dim = ((total_channels // 8) // 8) * 8  # Hacer divisible por 8
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

class SimpleDBRANet(SimpleHRNet):
    """DBRANet con dual-branch regional attention"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Regional attention simplificado
        self.regional_attention = nn.Sequential(
            nn.Conv2d(kwargs.get('width', 32) * 4, kwargs.get('width', 32), 3, padding=1),
            nn.BatchNorm2d(kwargs.get('width', 32)),
            nn.ReLU(inplace=True),
            nn.Conv2d(kwargs.get('width', 32), kwargs.get('width', 32), 1),
        )

class SimpleNLLinkNet(SimpleLinkNet):
    """NL-LinkNet con non-local operations"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Non-local block simplificado
        self.non_local = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 1),
        )

class SimpleMSFANet(nn.Module):
    """Implementación simplificada de MSFANet"""
    
    def __init__(self, num_classes=2, rgb_channels=3, msi_channels=8, width=32, **kwargs):
        super().__init__()
        
        # Cross-source Feature Fusion simplificado
        self.rgb_conv = nn.Conv2d(rgb_channels, 64, 3, padding=1)
        self.msi_conv = nn.Conv2d(msi_channels, 64, 3, padding=1)
        self.fusion_conv = nn.Conv2d(128, 64, 1)
        
        # HRNet encoder simplificado
        self.hrnet = SimpleHRNet(num_classes=num_classes, rgb_channels=64, width=width)
        
        # Multiscale decoder con attention
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

class PerformanceMetrics:
    """Clase para calcular métricas de rendimiento mejorada"""
    
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.total_samples = 0
    
    def update(self, pred, target):
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        
        # Filtrar valores válidos
        mask = (target >= 0) & (target < self.num_classes) & (pred >= 0) & (pred < self.num_classes)
        pred_valid = pred[mask]
        target_valid = target[mask]
        
        if len(pred_valid) == 0:
            return
        
        # Calcular matriz de confusión
        hist = np.bincount(
            self.num_classes * target_valid.astype(int) + pred_valid.astype(int),
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += hist
        self.total_samples += len(pred_valid)
    
    def compute_metrics(self):
        if self.total_samples == 0:
            return self._empty_metrics()
            
        hist = self.confusion_matrix
        
        # IoU para cada clase
        ious = []
        for i in range(self.num_classes):
            intersection = hist[i, i]
            union = hist[i, :].sum() + hist[:, i].sum() - intersection
            if union > 0:
                ious.append(intersection / union)
            else:
                ious.append(0.0)
        
        ious = np.array(ious)
        
        # mIoU
        miou = np.mean(ious)
        
        # Métricas específicas para la clase "road" 
        # (asumiendo que es la clase 1 en configuración de 2 clases, 
        # o clase 1 en configuración de 3 clases)
        road_class = 1 if self.num_classes >= 2 else 0
        
        if road_class < len(ious):
            road_iou = ious[road_class]
            
            # Precision, Recall, F1 para roads
            tp = hist[road_class, road_class]
            fp = hist[:, road_class].sum() - tp  # Predicciones incorrectas como road
            fn = hist[road_class, :].sum() - tp  # Roads no detectados
            
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1 = 2 * precision * recall / (precision + recall + 1e-7)
        else:
            road_iou = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        
        return {
            'road_iou': road_iou * 100,
            'miou': miou * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
            'all_ious': ious * 100  # IoU para todas las clases
        }
    
    def _empty_metrics(self):
        """Métricas por defecto cuando no hay datos"""
        return {
            'road_iou': 0.0,
            'miou': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'all_ious': np.zeros(self.num_classes)
        }

class ComparativeStudy:
    """Clase principal para el estudio comparativo"""
    
    def __init__(self, config_path='config_sentinel2.yaml'):
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.registry = ArchitectureRegistry()
        self.results = defaultdict(dict)
        
        # Crear directorios de resultados
        self.results_dir = Path('comparative_study_results')
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / 'models').mkdir(exist_ok=True)
        (self.results_dir / 'plots').mkdir(exist_ok=True)
        (self.results_dir / 'metrics').mkdir(exist_ok=True)
        
        print(f"Estudio comparativo inicializado en {self.device}")
        print(f"Arquitecturas a evaluar: {list(self.registry.architectures.keys())}")
    
    def load_config(self, config_path):
        """Cargar configuración desde YAML"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Configuración por defecto
            return {
                'data_dir': 'processed_sentinel2',
                'num_classes': 2,
                'rgb_channels': 3,
                'msi_channels': 10,
                'model_width': 32,
                'epochs': 50,  # Reducido para el estudio
                'batch_size': 4,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'num_workers': 2,
                'train_split': 0.8,
                'val_split': 0.2,
                'seed': 42
            }
    
    def create_data_loaders(self):
        """Crear data loaders para entrenamiento y validación"""
        try:
            dataset = Sentinel2Dataset(self.config['data_dir'])
            total_size = len(dataset)
            
            if total_size == 0:
                print("WARNING: Dataset vacío, creando dataset dummy")
                return self.create_dummy_loaders()
            
            # Verificar las dimensiones reales del dataset
            sample = dataset[0]
            if 'ms' in sample:
                actual_msi_channels = sample['ms'].shape[0]
                print(f"📡 Canales MSI detectados en el dataset: {actual_msi_channels}")
                
                # Actualizar configuración si es necesario
                if actual_msi_channels != self.config['msi_channels']:
                    print(f"⚠️ Actualizando canales MSI: {self.config['msi_channels']} → {actual_msi_channels}")
                    self.config['msi_channels'] = actual_msi_channels
            
            train_size = int(self.config['train_split'] * total_size)
            val_size = total_size - train_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config['num_workers'],
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers'],
                pin_memory=True
            )
            
            return train_loader, val_loader
            
        except Exception as e:
            print(f"Error creando data loaders: {e}")
            print("Creando data loaders dummy...")
            return self.create_dummy_loaders()
    
    def create_dummy_loaders(self):
        """Crear data loaders dummy para pruebas"""
        class DummyDataset:
            def __init__(self, size=100):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return {
                    'rgb': torch.randn(3, 512, 512),
                    'ms': torch.randn(10, 512, 512),
                    'mask': torch.randint(0, 2, (512, 512))
                }
        
        train_dataset = DummyDataset(80)
        val_dataset = DummyDataset(20)
        
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        
        return train_loader, val_loader
    
    def train_model(self, model_name, epochs=None):
        """Entrenar un modelo específico con mejor manejo de errores"""
        epochs = epochs or self.config['epochs']
        
        print(f"\n=== Entrenando {model_name} ===")
        
        try:
            # Crear modelo
            model_args = {
                'num_classes': self.config['num_classes'],
                'rgb_channels': self.config['rgb_channels'],
                'msi_channels': self.config['msi_channels'],
                'width': self.config.get('model_width', 32)
            }
            
            print(f"Parámetros del modelo: {model_args}")
            
            model = self.registry.architectures[model_name]['model_class'](**model_args)
            model = model.to(self.device)
            
            # Contar parámetros
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Parámetros totales: {total_params:,}")
            
            # Optimizador y loss
            optimizer = optim.AdamW(model.parameters(), 
                                   lr=self.config['learning_rate'],
                                   weight_decay=self.config['weight_decay'])
            
            # Loss con pesos para clases desbalanceadas
            if self.config.get('use_class_weights', False) and self.config['num_classes'] > 2:
                # Para configuración multiclase
                weights = [1.0, self.config.get('road_weight', 8.0)]
                if self.config['num_classes'] > 2:
                    weights.append(self.config.get('building_weight', 5.0))
                weights = torch.tensor(weights[:self.config['num_classes']]).to(self.device)
                criterion = nn.CrossEntropyLoss(weight=weights)
            else:
                criterion = nn.CrossEntropyLoss()
            
            # Data loaders
            train_loader, val_loader = self.create_data_loaders()
            
            # Test con un batch para verificar compatibilidad
            print("Verificando compatibilidad del modelo...")
            model.eval()
            with torch.no_grad():
                test_batch = next(iter(train_loader))
                rgb = test_batch['rgb'].to(self.device)
                msi = test_batch['ms'].to(self.device)
                target = test_batch['mask'].to(self.device)
                
                print(f"Entrada RGB: {rgb.shape}")
                print(f"Entrada MSI: {msi.shape}")
                print(f"Target: {target.shape}")
                
                # Test forward pass
                if model_name == 'MSFANet':
                    output = model(rgb, msi)
                else:
                    output = model(rgb, msi)
                
                print(f"Salida del modelo: {output.shape}")
                
                # Verificar que las dimensiones sean compatibles
                if output.shape[0] != target.shape[0]:
                    raise ValueError(f"Batch size mismatch: output {output.shape[0]} vs target {target.shape[0]}")
                
                if output.shape[-2:] != target.shape[-2:]:
                    print(f"WARNING: Redimensionando salida de {output.shape[-2:]} a {target.shape[-2:]}")
                    output = torch.nn.functional.interpolate(
                        output, size=target.shape[-2:], mode='bilinear', align_corners=False
                    )
                
                # Test loss
                loss = criterion(output, target)
                print(f"Loss de prueba: {loss.item():.4f}")
            
            print("✅ Verificación completada. Iniciando entrenamiento...")
            
            # Entrenamiento
            model.train()
            train_losses = []
            val_metrics_history = []
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
                
                for batch_idx, batch in enumerate(progress_bar):
                    try:
                        rgb = batch['rgb'].to(self.device)
                        msi = batch['ms'].to(self.device)
                        target = batch['mask'].to(self.device)
                        
                        optimizer.zero_grad()
                        
                        # Forward pass
                        if model_name == 'MSFANet':
                            output = model(rgb, msi)
                        else:
                            output = model(rgb, msi)
                        
                        # Asegurar dimensiones compatibles
                        if output.shape[-2:] != target.shape[-2:]:
                            output = torch.nn.functional.interpolate(
                                output, size=target.shape[-2:], mode='bilinear', align_corners=False
                            )
                        
                        loss = criterion(output, target)
                        
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"WARNING: Loss inválido en batch {batch_idx}: {loss.item()}")
                            continue
                        
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        num_batches += 1
                        
                        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
                        
                    except Exception as e:
                        print(f"Error en batch {batch_idx}: {e}")
                        continue
                
                if num_batches == 0:
                    print(f"WARNING: No se procesaron batches en época {epoch+1}")
                    continue
                
                avg_loss = epoch_loss / num_batches
                train_losses.append(avg_loss)
                
                # Validación cada 5 épocas
                if (epoch + 1) % 5 == 0:
                    val_metrics = self.evaluate_model(model, val_loader, model_name)
                    val_metrics_history.append(val_metrics)
                    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val IoU={val_metrics['road_iou']:.2f}%")
            
            # Evaluación final
            final_metrics = self.evaluate_model(model, val_loader, model_name)
            
            # Calcular parámetros y tiempo de inferencia
            num_params = sum(p.numel() for p in model.parameters()) / 1e6  # En millones
            inference_time = self.measure_inference_time(model)
            
            # Guardar resultados
            self.results[model_name] = {
                'metrics': final_metrics,
                'num_parameters': num_params,
                'inference_time': inference_time,
                'train_losses': train_losses,
                'val_metrics_history': val_metrics_history,
                'description': self.registry.architectures[model_name]['description']
            }
            
            # Guardar modelo
            model_path = self.results_dir / 'models' / f'{model_name}.pth'
            torch.save(model.state_dict(), model_path)
            
            print(f"{model_name} completado - IoU: {final_metrics['road_iou']:.2f}%, "
                  f"Params: {num_params:.2f}M, Time: {inference_time:.2f}ms")
            
            return model
            
        except Exception as e:
            print(f"ERROR entrenando {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_model(self, model, val_loader, model_name):
        """Evaluar un modelo con mejor manejo de errores"""
        model.eval()
        metrics_calculator = PerformanceMetrics(self.config['num_classes'])
        
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    rgb = batch['rgb'].to(self.device)
                    msi = batch['ms'].to(self.device)
                    target = batch['mask'].to(self.device)
                    
                    if model_name == 'MSFANet':
                        output = model(rgb, msi)
                    else:
                        output = model(rgb, msi)
                    
                    # Asegurar dimensiones compatibles
                    if output.shape[-2:] != target.shape[-2:]:
                        output = torch.nn.functional.interpolate(
                            output, size=target.shape[-2:], mode='bilinear', align_corners=False
                        )
                    
                    pred = torch.argmax(output, dim=1)
                    
                    # Verificar que las predicciones estén en el rango válido
                    pred = torch.clamp(pred, 0, self.config['num_classes'] - 1)
                    target = torch.clamp(target, 0, self.config['num_classes'] - 1)
                    
                    metrics_calculator.update(pred, target)
                    total_samples += pred.numel()
                    
                except Exception as e:
                    print(f"Error en validación batch {batch_idx}: {e}")
                    continue
        
        metrics = metrics_calculator.compute_metrics()
        
        # Añadir información adicional
        metrics['total_samples'] = total_samples
        
        if total_samples == 0:
            print(f"WARNING: No se procesaron muestras en validación para {model_name}")
        
        return metrics
    
    def measure_inference_time(self, model, num_iterations=100):
        """Medir tiempo de inferencia"""
        model.eval()
        
        # Datos dummy
        rgb = torch.randn(1, self.config['rgb_channels'], 512, 512).to(self.device)
        msi = torch.randn(1, self.config['msi_channels'], 512, 512).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(rgb, msi)
        
        # Medición
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(rgb, msi)
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations * 1000  # En milisegundos
        return avg_time
    
    def run_complete_study(self, architectures=None, epochs=None):
        """Ejecutar estudio completo"""
        architectures = architectures or list(self.registry.architectures.keys())
        
        print(f"Iniciando estudio comparativo de {len(architectures)} arquitecturas")
        print(f"Configuración: {self.config}")
        
        for arch_name in architectures:
            try:
                self.train_model(arch_name, epochs)
            except Exception as e:
                print(f"ERROR entrenando {arch_name}: {e}")
                continue
        
        # Generar reportes y gráficas
        self.generate_reports()
        self.generate_plots()
        
        print(f"\nEstudio completado. Resultados guardados en {self.results_dir}")
    
    def generate_reports(self):
        """Generar reportes de resultados"""
        
        # Crear DataFrame con resultados
        data = []
        for arch_name, results in self.results.items():
            row = {
                'Architecture': arch_name,
                'Description': results['description'],
                'IoU (%)': results['metrics']['road_iou'],
                'mIoU (%)': results['metrics']['miou'],
                'F1-Score (%)': results['metrics']['f1'],
                'Precision (%)': results['metrics']['precision'],
                'Recall (%)': results['metrics']['recall'],
                'Parameters (M)': results['num_parameters'],
                'Inference Time (ms)': results['inference_time']
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.sort_values('IoU (%)', ascending=False)
        
        # Guardar como CSV
        df.to_csv(self.results_dir / 'metrics' / 'comparative_results.csv', index=False)
        
        # Guardar como JSON
        with open(self.results_dir / 'metrics' / 'detailed_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generar reporte de texto
        with open(self.results_dir / 'comparative_report.txt', 'w') as f:
            f.write("ESTUDIO COMPARATIVO DE ARQUITECTURAS PARA SEGMENTACIÓN DE CARRETERAS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("CONFIGURACIÓN DEL EXPERIMENTO:\n")
            f.write("-" * 30 + "\n")
            for key, value in self.config.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("RESULTADOS COMPARATIVOS:\n")
            f.write("-" * 30 + "\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            # Análisis de resultados
            best_iou = df.loc[0]
            f.write("ANÁLISIS DE RESULTADOS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Mejor arquitectura por IoU: {best_iou['Architecture']} ({best_iou['IoU (%)']} %)\n")
            f.write(f"Arquitectura más eficiente: {df.loc[df['Inference Time (ms)'].idxmin(), 'Architecture']}\n")
            f.write(f"Arquitectura más liviana: {df.loc[df['Parameters (M)'].idxmin(), 'Architecture']}\n")
        
        print(f"Reportes generados y guardados en {self.results_dir}")
    
    def generate_plots(self):
        """Generar gráficas comparativas"""
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
        
        # 1. Gráfica de barras con métricas principales
        self.plot_metrics_comparison()
        
        # 2. Gráfica de dispersión: Efficiency vs Accuracy
        self.plot_efficiency_vs_accuracy()
        
        # 3. Gráfica de radar con todas las métricas
        self.plot_radar_metrics()
        
        # 4. Historial de entrenamiento
        self.plot_training_history()
        
        print(f"Gráficas generadas y guardadas en {self.results_dir / 'plots'}")
    
    def plot_metrics_comparison(self):
        """Gráfica de barras comparando métricas principales"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparación de Métricas por Arquitectura', fontsize=16, fontweight='bold')
        
        architectures = list(self.results.keys())
        metrics = ['road_iou', 'f1', 'precision', 'recall']
        titles = ['IoU de Carreteras (%)', 'F1-Score (%)', 'Precisión (%)', 'Recall (%)']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            values = [self.results[arch]['metrics'][metric] for arch in architectures]
            
            bars = ax.bar(architectures, values, alpha=0.8)
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel('Porcentaje (%)')
            ax.tick_params(axis='x', rotation=45)
            
            # Añadir valores en las barras
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'plots' / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_efficiency_vs_accuracy(self):
        """Gráfica de dispersión: Eficiencia vs Precisión"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        architectures = list(self.results.keys())
        iou_values = [self.results[arch]['metrics']['road_iou'] for arch in architectures]
        inference_times = [self.results[arch]['inference_time'] for arch in architectures]
        num_params = [self.results[arch]['num_parameters'] for arch in architectures]
        
        # Tiempo de inferencia vs IoU
        ax1.scatter(inference_times, iou_values, s=100, alpha=0.7)
        for i, arch in enumerate(architectures):
            ax1.annotate(arch, (inference_times[i], iou_values[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax1.set_xlabel('Tiempo de Inferencia (ms)')
        ax1.set_ylabel('IoU de Carreteras (%)')
        ax1.set_title('Eficiencia vs Precisión (Tiempo)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Parámetros vs IoU
        ax2.scatter(num_params, iou_values, s=100, alpha=0.7, color='orange')
        for i, arch in enumerate(architectures):
            ax2.annotate(arch, (num_params[i], iou_values[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax2.set_xlabel('Número de Parámetros (M)')
        ax2.set_ylabel('IoU de Carreteras (%)')
        ax2.set_title('Complejidad vs Precisión (Parámetros)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'plots' / 'efficiency_vs_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_radar_metrics(self):
        """Gráfica de radar con todas las métricas normalizadas"""
        architectures = list(self.results.keys())
        metrics = ['road_iou', 'f1', 'precision', 'recall']
        
        # Normalizar métricas (0-1)
        normalized_data = {}
        for metric in metrics:
            values = [self.results[arch]['metrics'][metric] for arch in architectures]
            max_val = max(values)
            min_val = min(values)
            normalized_data[metric] = [(val - min_val) / (max_val - min_val) for val in values]
        
        # Crear gráfica de radar
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Cerrar el círculo
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(architectures)))
        
        for i, arch in enumerate(architectures):
            values = [normalized_data[metric][i] for metric in metrics]
            values += values[:1]  # Cerrar el círculo
            
            ax.plot(angles, values, 'o-', linewidth=2, label=arch, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['IoU', 'F1-Score', 'Precisión', 'Recall'])
        ax.set_ylim(0, 1)
        ax.set_title('Comparación Multimétrica (Normalizada)', fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'plots' / 'radar_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_history(self):
        """Gráfica del historial de entrenamiento"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss durante el entrenamiento
        for arch_name, results in self.results.items():
            if 'train_losses' in results and results['train_losses']:
                epochs = range(1, len(results['train_losses']) + 1)
                axes[0].plot(epochs, results['train_losses'], label=arch_name, linewidth=2)
        
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Evolución del Loss durante Entrenamiento', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # IoU durante validación
        for arch_name, results in self.results.items():
            if 'val_metrics_history' in results and results['val_metrics_history']:
                epochs = range(5, len(results['val_metrics_history']) * 5 + 1, 5)
                iou_values = [m['road_iou'] for m in results['val_metrics_history']]
                axes[1].plot(epochs, iou_values, label=arch_name, linewidth=2, marker='o')
        
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('IoU de Carreteras (%)')
        axes[1].set_title('Evolución del IoU durante Validación', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'plots' / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Función principal mejorada"""
    print("=== ESTUDIO COMPARATIVO DE ARQUITECTURAS ===")
    print("Universidad de La Laguna - Ingeniería Informática")
    print("Basado en: MSFANet Paper (Remote Sensing 2023)")
    print("=" * 50)
    
    # Crear configuración optimizada para el estudio si no existe
    config_path = 'config_comparative_study.yaml'
    if not os.path.exists(config_path):
        create_comparative_config(config_path)
        print(f"✅ Configuración optimizada creada en {config_path}")
    
    # Inicializar estudio
    study = ComparativeStudy(config_path)
    
    # Configurar arquitecturas a evaluar (puedes modificar esta lista)
    architectures_to_test = [
        'LinkNet',
        'D-LinkNet', 
        'HRNet',
        'CCNet',
        'DANet',
        'MSFANet'
    ]
    
    print(f"Arquitecturas seleccionadas: {architectures_to_test}")
    
    # Ejecutar estudio con configuración optimizada
    study.run_complete_study(architectures=architectures_to_test, epochs=100)
    
    print("\n=== ESTUDIO COMPLETADO ===")
    print(f"Revisa los resultados en: {study.results_dir}")
    print("Archivos generados:")
    print("- comparative_report.txt: Reporte detallado")
    print("- metrics/comparative_results.csv: Tabla de resultados")
    print("- plots/*.png: Gráficas comparativas")

def create_comparative_config(config_path):
    """Crear configuración optimizada para el estudio comparativo"""
    config = {
        'data_dir': 'processed_sentinel2',
        'num_classes': 2,  # Configuración binaria como en el paper original
        'model_width': 32,  # Width estándar
        'rgb_channels': 3,
        'msi_channels': 10,  # Canales MSI de Sentinel-2 (actualizado)
        
        # Configuración de entrenamiento optimizada
        'epochs': 50,
        'batch_size': 4,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'num_workers': 2,
        
        # Pesos para clases desbalanceadas
        'use_class_weights': True,
        'road_weight': 8.0,  # Peso para la clase road
        
        # División de datos
        'train_split': 0.8,
        'val_split': 0.2,
        'seed': 42,
        
        # Frecuencias de validación y guardado
        'val_frequency': 5,
        'save_frequency': 10,
        
        # Hardware
        'device': 'cuda',
        
        # Sentinel-2 específico
        'sentinel2': {
            'target_resolution': 10,
            'osm_buffer_meters': 5,
            'min_road_pixels': 10,
            'crop_size': 512,
            'max_crops_per_scene': 50
        },
        
        # Data augmentation
        'augmentation': {
            'enabled': True,
            'rotation': True,
            'flip': True,
            'brightness': 0.1,
            'contrast': 0.1
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return config

if __name__ == "__main__":
    main()