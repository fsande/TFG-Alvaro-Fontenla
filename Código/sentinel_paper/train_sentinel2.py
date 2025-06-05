#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
from pathlib import Path
from tqdm import tqdm
import yaml
import argparse
from collections import OrderedDict

MODEL_ADAPTER_AVAILABLE = False
ORIGINAL_MODEL_AVAILABLE = False
SIMPLE_MODEL_AVAILABLE = False

try:
    from model_adapter import get_msfanet, get_simple_msfanet
    MODEL_ADAPTER_AVAILABLE = True
    ORIGINAL_MODEL_AVAILABLE = True
    SIMPLE_MODEL_AVAILABLE = True
    print("Adaptador de modelos cargado")
except ImportError as e:
    print(f"Adaptador no disponible: {e}")
    
    try:
        from msfanet_model import get_msfanet
        ORIGINAL_MODEL_AVAILABLE = True
        print("Usando modelo original")
    except ImportError:
        ORIGINAL_MODEL_AVAILABLE = False
        print("Modelo original no disponible")

    try:
        from simple_msfanet import get_simple_msfanet
        SIMPLE_MODEL_AVAILABLE = True
        print("Usando modelo simple")
    except ImportError:
        SIMPLE_MODEL_AVAILABLE = False
        print("Modelo simple no disponible")

# Importar dataset
try:
    from sentinel2_processor_fixed import Sentinel2Dataset
    print("Dataset Sentinel-2 disponible")
except ImportError:
    try:
        from sentinel2_processor import Sentinel2Dataset
        print("Dataset Sentinel-2 disponible")
    except ImportError:
        print("Dataset Sentinel-2 no disponible")
        raise ImportError("No se puede importar Sentinel2Dataset")

class PolynomialLR:
    """Polynomial Learning Rate Scheduler"""
    
    def __init__(self, optimizer, max_iterations, power=0.9):
        self.optimizer = optimizer
        self.max_iterations = max_iterations
        self.power = power
        self.current_iteration = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self):
        self.current_iteration += 1
        for i, group in enumerate(self.optimizer.param_groups):
            lr = self.base_lrs[i] * (1 - self.current_iteration / self.max_iterations) ** self.power
            group['lr'] = lr
            
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

class SegmentationMetrics:
    """Métricas de segmentación"""
    
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        
    def update(self, pred, target):
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        
        mask = (target >= 0) & (target < self.num_classes)
        
        hist = np.bincount(
            self.num_classes * target[mask].astype(int) + pred[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += hist
        
    def compute_metrics(self):
        hist = self.confusion_matrix
        
        # IoU para cada clase
        ious = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-7)
        
        # mIoU
        miou = np.nanmean(ious)
        
        # Para clase road (clase 1)
        tp = hist[1, 1]
        fp = hist[0, 1] 
        fn = hist[1, 0]
        
        # Precision, Recall, F1 para roads
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        return {
            'road_iou': ious[1] * 100,
            'miou': miou * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100
        }

class MSFANetSentinel2Trainer:
    """Trainer especializado para Sentinel-2 con Transfer Learning"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
        
        # Crear directorios
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear modelo
        self.model = self.create_model(config)
        if self.model is None:
            raise ValueError("No se pudo crear el modelo")
            
        # Aplicar transfer learning si está configurado
        if config.get('transfer_learning', {}).get('enabled', False):
            self.load_pretrained_weights(config['transfer_learning'])
        
        self.model = self.model.to(self.device)
        
        # Loss function con pesos específicos para OSM
        road_weight = config.get('road_weight', 3.0)
        weights = torch.tensor([1.0, road_weight]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        
        # Optimizer con diferentes learning rates
        self.optimizer = self.create_optimizer(config)
        
        # Data loaders
        self.train_loader, self.val_loader = self.create_data_loaders(config)
        
        # Learning rate scheduler
        max_iterations = len(self.train_loader) * config['epochs']
        self.scheduler = PolynomialLR(
            self.optimizer,
            max_iterations=max_iterations,
            power=config.get('poly_power', 0.9)
        )
        
        # Métricas
        self.train_metrics = SegmentationMetrics(config['num_classes'])
        self.val_metrics = SegmentationMetrics(config['num_classes'])
        
        # Tensorboard
        self.writer = SummaryWriter(config['log_dir'])
        
        # Estado de entrenamiento
        self.epoch = 0
        self.best_miou = 0.0
        self.global_step = 0
        
        print(f"Entrenador Sentinel-2 inicializado:")
        print(f"   Modelo: {self.model.__class__.__name__}")
        print(f"   Device: {self.device}")
        print(f"   Parámetros: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Train samples: {len(self.train_loader.dataset)}")
        print(f"   Val samples: {len(self.val_loader.dataset)}")
        
        # Información de transfer learning
        if config.get('transfer_learning', {}).get('enabled', False):
            print(f"Transfer Learning activado")
            print(f"Modelo base: {config['transfer_learning'].get('pretrained_path')}")
    
    def create_model(self, config):
        print(f"Configuración del modelo:")
        print(f"   RGB channels: {config.get('rgb_channels', 3)}")
        print(f"   MSI channels: {config.get('msi_channels', 10)}")
        print(f"   Classes: {config['num_classes']}")
        print(f"   Width: {config.get('model_width', 32)}")
        
        # Verificar disponibilidad del adaptador
        if not MODEL_ADAPTER_AVAILABLE:
            print("Model adapter no disponible")
            return None
        
        # Parámetros del modelo
        model_args = {
            'num_classes': config['num_classes'],
            'width': config.get('model_width', 32),
            'rgb_channels': config.get('rgb_channels', 3),
            'msi_channels': config.get('msi_channels', 10)
        }
        
        model_type = config.get('model_type', 'auto')
        
        if model_type == 'auto':
            # Probar modelo original primero (a través del adaptador)
            try:
                model = get_msfanet(**model_args)
                print("Usando modelo MSFANet (via adaptador)")
                return model
            except Exception as e:
                print(f"Error con modelo original via adaptador: {e}")
            
            # Fallback a modelo simple (a través del adaptador)
            try:
                model = get_simple_msfanet(**model_args)
                print("Usando modelo simple MSFANet (via adaptador)")
                return model
            except Exception as e:
                print(f"Error con modelo simple via adaptador: {e}")
                    
        elif model_type == 'original':
            try:
                return get_msfanet(**model_args)
            except Exception as e:
                print(f"Error creando modelo original: {e}")
                
        elif model_type == 'simple':
            try:
                return get_simple_msfanet(**model_args)
            except Exception as e:
                print(f"Error creando modelo simple: {e}")
        
        print("No se pudo crear ningún modelo")
        return None
    
    def load_pretrained_weights(self, tl_config):
        pretrained_path = tl_config.get('pretrained_path')
        
        if not pretrained_path or not os.path.exists(pretrained_path):
            print(f"Archivo de pesos no encontrado: {pretrained_path}")
            return
        
        print(f"Cargando pesos pre-entrenados...")
        
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                pretrained_state = checkpoint['model_state_dict']
            else:
                pretrained_state = checkpoint
            
            # Estado actual del modelo
            model_state = self.model.state_dict()
            
            # Transfer learning inteligente
            transferred_keys = []
            skipped_keys = []
            
            for key, param in pretrained_state.items():
                if key in model_state:
                    # Verificar compatibilidad de dimensiones
                    if param.shape == model_state[key].shape:
                        model_state[key] = param
                        transferred_keys.append(key)
                    else:
                        # Dimensiones diferentes - típico en CFFM con diferentes canales MS
                        if 'msi_conv' in key or 'cffm' in key.lower():
                            print(f"Adaptando capa {key}: {param.shape} -> {model_state[key].shape}")
                            # Estrategias de adaptación
                            if 'weight' in key and len(param.shape) == 4:  # Conv weights
                                # Interpolar o recortar canales de entrada
                                self._adapt_conv_weights(key, param, model_state)
                                transferred_keys.append(key)
                            else:
                                skipped_keys.append(key)
                        else:
                            skipped_keys.append(key)
                else:
                    skipped_keys.append(key)
            
            # Cargar estado adaptado
            self.model.load_state_dict(model_state)
            
            print(f"Transfer learning completado:")
            print(f"   Capas transferidas: {len(transferred_keys)}")
            print(f"   Capas omitidas: {len(skipped_keys)}")
            
            if len(skipped_keys) > 0:
                print(f"   Capas omitidas: {skipped_keys[:5]}{'...' if len(skipped_keys) > 5 else ''}")
                
        except Exception as e:
            print(f"Error en transfer learning: {e}")
    
    def _adapt_conv_weights(self, key, pretrained_param, model_state):
        target_shape = model_state[key].shape
        source_shape = pretrained_param.shape
        
        if len(source_shape) == 4:  # Conv2d weights [out_ch, in_ch, h, w]
            target_param = model_state[key].clone()
            
            # Adaptar canales de entrada
            min_in_ch = min(source_shape[1], target_shape[1])
            target_param[:, :min_in_ch] = pretrained_param[:, :min_in_ch]
            
            # Si necesitamos más canales, duplicar o interpolar
            if target_shape[1] > source_shape[1]:
                remaining_ch = target_shape[1] - source_shape[1]
                # Duplicar los últimos canales
                for i in range(remaining_ch):
                    src_ch = source_shape[1] - 1 - (i % source_shape[1])
                    target_ch = source_shape[1] + i
                    if target_ch < target_shape[1]:
                        target_param[:, target_ch] = pretrained_param[:, src_ch]
            
            model_state[key] = target_param
    
    def create_optimizer(self, config):
        base_lr = config['learning_rate']
        weight_decay = config.get('weight_decay', 1e-4)
        
        # Parámetros con diferentes learning rates
        param_groups = []
        
        if config.get('transfer_learning', {}).get('enabled', False):
            # LR más bajo para capas pre-entrenadas
            pretrained_params = []
            new_params = []
            
            for name, param in self.model.named_parameters():
                if 'cffm' in name.lower() and 'msi' in name.lower():
                    # Nuevas capas MSI - LR normal
                    new_params.append(param)
                else:
                    # Capas pre-entrenadas - LR reducido
                    pretrained_params.append(param)
            
            if pretrained_params:
                param_groups.append({
                    'params': pretrained_params,
                    'lr': base_lr * 0.1,  # 10% del LR base
                    'weight_decay': weight_decay
                })
            
            if new_params:
                param_groups.append({
                    'params': new_params,
                    'lr': base_lr,  # LR completo
                    'weight_decay': weight_decay
                })
        else:
            # Sin transfer learning - todos los parámetros igual
            param_groups.append({
                'params': self.model.parameters(),
                'lr': base_lr,
                'weight_decay': weight_decay
            })
        
        return optim.AdamW(param_groups)
    
    def create_data_loaders(self, config):
        # Dataset completo
        full_dataset = Sentinel2Dataset(config['data_dir'])
        total_size = len(full_dataset)
        
        if total_size == 0:
            raise ValueError("Dataset Sentinel-2 vacío")
        
        # Dividir dataset
        train_split = config.get('train_split', 0.8)
        train_size = int(train_split * total_size)
        val_size = total_size - train_size
        
        torch.manual_seed(config.get('seed', 42))
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Crear loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True  # Para estabilidad con OSM data
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self):
        self.model.train()
        self.train_metrics.reset()
        
        epoch_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch+1}')
        
        for batch_idx, data in enumerate(progress_bar):
            rgb = data['rgb'].to(self.device)
            msi = data['ms'].to(self.device)
            target = data['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(rgb, msi)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping para estabilidad con OSM data
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Métricas
            pred = torch.argmax(output, dim=1)
            self.train_metrics.update(pred, target)
            
            # Logging
            epoch_loss += loss.item()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Estadísticas específicas de OSM
            road_pixels = torch.sum(target == 1).item()
            total_pixels = target.numel()
            road_ratio = road_pixels / total_pixels * 100
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{current_lr:.6f}',
                'Roads%': f'{road_ratio:.1f}'
            })
            
            # Tensorboard logging
            if self.global_step % 100 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/LR', current_lr, self.global_step)
                self.writer.add_scalar('Train/RoadRatio', road_ratio, self.global_step)
                
            self.global_step += 1
        
        # Métricas de época
        metrics = self.train_metrics.compute_metrics()
        avg_loss = epoch_loss / len(self.train_loader)
        
        return avg_loss, metrics
    
    def validate_epoch(self):
        self.model.eval()
        self.val_metrics.reset()
        
        val_loss = 0.0
        total_road_pixels = 0
        total_pixels = 0
        
        with torch.no_grad():
            for data in tqdm(self.val_loader, desc='Validation'):
                rgb = data['rgb'].to(self.device)
                msi = data['ms'].to(self.device)
                target = data['mask'].to(self.device)
                
                # Forward pass
                output = self.model(rgb, msi)
                loss = self.criterion(output, target)
                
                # Métricas
                pred = torch.argmax(output, dim=1)
                self.val_metrics.update(pred, target)
                val_loss += loss.item()
                
                # Estadísticas OSM
                total_road_pixels += torch.sum(target == 1).item()
                total_pixels += target.numel()
        
        # Métricas finales
        metrics = self.val_metrics.compute_metrics()
        avg_loss = val_loss / len(self.val_loader)
        road_coverage = (total_road_pixels / total_pixels) * 100
        
        # Añadir cobertura de carreteras a métricas
        metrics['road_coverage'] = road_coverage
        
        return avg_loss, metrics
    
    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_miou': self.best_miou,
            'config': self.config,
            'model_type': 'sentinel2_msfanet'
        }
        
        # Guardar último checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'last_s2.pth')
        
        # Guardar mejor modelo
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best_s2.pth')
            print(f"Mejor modelo Sentinel-2 guardado")
    
    def train(self):
        print("Iniciando entrenamiento Sentinel-2...")
        
        for epoch in range(self.config['epochs']):
            self.epoch = epoch
            start_time = time.time()
            
            # Training
            train_loss, train_metrics = self.train_epoch()
            
            # Validation
            if epoch % self.config.get('val_frequency', 1) == 0:
                val_loss, val_metrics = self.validate_epoch()
            else:
                val_loss = 0
                val_metrics = {'road_iou': 0, 'f1': 0, 'miou': 0, 'road_coverage': 0}
            
            epoch_time = time.time() - start_time
            
            # Logging detallado
            print(f"\nEpoch {epoch+1}/{self.config['epochs']} - {epoch_time:.2f}s")
            print(f"Train - Loss: {train_loss:.4f}, Road IoU: {train_metrics['road_iou']:.2f}%, "
                  f"F1: {train_metrics['f1']:.2f}%")
            
            if epoch % self.config.get('val_frequency', 1) == 0:
                print(f"Val   - Loss: {val_loss:.4f}, Road IoU: {val_metrics['road_iou']:.2f}%, "
                      f"F1: {val_metrics['f1']:.2f}%, Coverage: {val_metrics['road_coverage']:.2f}%")
            
            # Tensorboard logging
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Train_Road_IoU', train_metrics['road_iou'], epoch)
            self.writer.add_scalar('Epoch/Train_F1', train_metrics['f1'], epoch)
            
            if epoch % self.config.get('val_frequency', 1) == 0:
                self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
                self.writer.add_scalar('Epoch/Val_Road_IoU', val_metrics['road_iou'], epoch)
                self.writer.add_scalar('Epoch/Val_F1', val_metrics['f1'], epoch)
                self.writer.add_scalar('Epoch/Val_Road_Coverage', val_metrics['road_coverage'], epoch)
            
            # Guardar checkpoints
            is_best = val_metrics['miou'] > self.best_miou
            if is_best:
                self.best_miou = val_metrics['miou']
                print(f"Nuevo mejor modelo Sentinel-2. mIoU: {self.best_miou:.2f}%")
            
            if epoch % self.config.get('save_frequency', 10) == 0 or is_best:
                self.save_checkpoint(is_best)
        
        print(f"\nEntrenamiento Sentinel-2 completado")
        print(f"Mejor mIoU: {self.best_miou:.2f}%")
        self.writer.close()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validaciones específicas
    required_keys = ['data_dir', 'rgb_channels', 'msi_channels', 'num_classes']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Configuración faltante: {key}")
    
    return config

def main():
    parser = argparse.ArgumentParser(description='MSFANet Sentinel-2 Training')
    parser.add_argument('--config', type=str, default='config_sentinel2.yaml',
                       help='Path to Sentinel-2 config file')
    
    args = parser.parse_args()
    
    # Cargar configuración
    if not os.path.exists(args.config):
        print(f"Config file {args.config} not found")
        return
    
    config = load_config(args.config)
    
    # Verificar dependencias
    if not MODEL_ADAPTER_AVAILABLE:
        print("ERROR: model_adapter.py no está disponible")
        return
    
    try:
        # Crear trainer
        trainer = MSFANetSentinel2Trainer(config)
        
        # Iniciar entrenamiento
        trainer.train()
        
    except Exception as e:
        print(f"Error durante entrenamiento Sentinel-2: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
