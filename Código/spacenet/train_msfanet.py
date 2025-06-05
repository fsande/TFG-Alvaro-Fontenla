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

try:
    from msfanet_model import get_msfanet
    ORIGINAL_MODEL_AVAILABLE = True
except ImportError:
    ORIGINAL_MODEL_AVAILABLE = False
    print("⚠️ Modelo original no disponible")

try:
    from simple_msfanet import get_simple_msfanet
    SIMPLE_MODEL_AVAILABLE = True
except ImportError:
    SIMPLE_MODEL_AVAILABLE = False
    print("⚠️ Modelo simple no disponible")

from final_spacenet_processor import WorkingSpaceNetDataset

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

class MSFANetTrainer:
    """Trainer mejorado para MSFANet"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
        
        # Crear directorios
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear modelo
        self.model = self.create_model(config)
        
        if self.model is None:
            raise ValueError("No se pudo crear ningún modelo")
        
        self.model = self.model.to(self.device)
        
        # Loss function
        if config.get('use_class_weights', False):
            # Peso mayor para clase road
            weights = torch.tensor([1.0, 3.0]).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Data loaders
        self.train_loader, self.val_loader = self.create_data_loaders(config)
        
        # Learning rate scheduler
        max_iterations = len(self.train_loader) * config['epochs']
        self.scheduler = PolynomialLR(
            self.optimizer, 
            max_iterations=max_iterations,
            power=config.get('poly_power', 0.9)
        )
        
        # Metrics
        self.train_metrics = SegmentationMetrics(config['num_classes'])
        self.val_metrics = SegmentationMetrics(config['num_classes'])
        
        # Tensorboard
        self.writer = SummaryWriter(config['log_dir'])
        
        # Training state
        self.epoch = 0
        self.best_miou = 0.0
        self.global_step = 0
        
        print(f"Entrenador inicializado:")
        print(f"  Modelo: {self.model.__class__.__name__}")
        print(f"  Device: {self.device}")
        print(f"  Parámetros: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Train samples: {len(self.train_loader.dataset)}")
        print(f"  Val samples: {len(self.val_loader.dataset)}")
        
    def create_model(self, config):
        model_type = config.get('model_type', 'auto')
        
        if model_type == 'auto':
            if ORIGINAL_MODEL_AVAILABLE:
                try:
                    model = get_msfanet(
                        num_classes=config['num_classes'],
                        width=config.get('model_width', 32)
                    )
                    print("Usando modelo original MSFANet")
                    return model
                except Exception as e:
                    print(f"Error con modelo original: {e}")
            
            if SIMPLE_MODEL_AVAILABLE:
                try:
                    model = get_simple_msfanet(
                        num_classes=config['num_classes'],
                        width=config.get('model_width', 32)
                    )
                    print("Usando modelo simplificado MSFANet")
                    return model
                except Exception as e:
                    print(f"Error con modelo simple: {e}")
                    
        elif model_type == 'original' and ORIGINAL_MODEL_AVAILABLE:
            return get_msfanet(config['num_classes'], config.get('model_width', 32))
        elif model_type == 'simple' and SIMPLE_MODEL_AVAILABLE:
            return get_simple_msfanet(config['num_classes'], config.get('model_width', 32))
        
        return None
        
    def create_data_loaders(self, config):
        full_dataset = WorkingSpaceNetDataset(config['data_dir'])
        total_size = len(full_dataset)
        
        if total_size == 0:
            raise ValueError("Dataset vacío")
        
        # Calcular tamaños de split
        train_split = config.get('train_split', 0.8)
        train_size = int(train_split * total_size)
        val_size = total_size - train_size
        
        # Dividir dataset
        torch.manual_seed(config.get('seed', 42))
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Crear loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
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
            self.optimizer.step()
            self.scheduler.step()
            
            # Metrics
            pred = torch.argmax(output, dim=1)
            self.train_metrics.update(pred, target)
            
            # Logging
            epoch_loss += loss.item()
            current_lr = self.scheduler.get_last_lr()[0]
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{current_lr:.6f}'
            })
            
            # Tensorboard logging
            if self.global_step % 100 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/LR', current_lr, self.global_step)
                
            self.global_step += 1
            
        # Compute epoch metrics
        metrics = self.train_metrics.compute_metrics()
        avg_loss = epoch_loss / len(self.train_loader)
        
        return avg_loss, metrics
        
    def validate_epoch(self):
        self.model.eval()
        self.val_metrics.reset()
        
        val_loss = 0.0
        
        with torch.no_grad():
            for data in tqdm(self.val_loader, desc='Validation'):
                rgb = data['rgb'].to(self.device)
                msi = data['ms'].to(self.device)
                target = data['mask'].to(self.device)
                
                # Forward pass
                output = self.model(rgb, msi)
                loss = self.criterion(output, target)
                
                # Metrics
                pred = torch.argmax(output, dim=1)
                self.val_metrics.update(pred, target)
                val_loss += loss.item()
                
        # Compute metrics
        metrics = self.val_metrics.compute_metrics()
        avg_loss = val_loss / len(self.val_loader)
        
        return avg_loss, metrics
        
    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_miou': self.best_miou,
            'config': self.config
        }
        
        torch.save(checkpoint, self.checkpoint_dir / 'last.pth')
        
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
            
    def train(self):
        print("Iniciando entrenamiento...")
        
        for epoch in range(self.config['epochs']):
            self.epoch = epoch
            start_time = time.time()
            
            train_loss, train_metrics = self.train_epoch()
            
            if epoch % self.config.get('val_frequency', 1) == 0:
                val_loss, val_metrics = self.validate_epoch()
            else:
                val_loss, val_metrics = 0, {'road_iou': 0, 'f1': 0, 'miou': 0}
            
            epoch_time = time.time() - start_time
            
            print(f"\nEpoch {epoch+1}/{self.config['epochs']} - {epoch_time:.2f}s")
            print(f"Train - Loss: {train_loss:.4f}, IoU: {train_metrics['road_iou']:.2f}%, "
                  f"F1: {train_metrics['f1']:.2f}%")
            
            if epoch % self.config.get('val_frequency', 1) == 0:
                print(f"Val   - Loss: {val_loss:.4f}, IoU: {val_metrics['road_iou']:.2f}%, "
                      f"F1: {val_metrics['f1']:.2f}%")
            
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Train_IoU', train_metrics['road_iou'], epoch)
            self.writer.add_scalar('Epoch/Train_F1', train_metrics['f1'], epoch)
            
            if epoch % self.config.get('val_frequency', 1) == 0:
                self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
                self.writer.add_scalar('Epoch/Val_IoU', val_metrics['road_iou'], epoch)
                self.writer.add_scalar('Epoch/Val_F1', val_metrics['f1'], epoch)
            
            is_best = val_metrics['miou'] > self.best_miou
            if is_best:
                self.best_miou = val_metrics['miou']
                print(f"Nuevo mejor modelo. mIoU: {self.best_miou:.2f}%")
                
            if epoch % self.config.get('save_frequency', 10) == 0:
                self.save_checkpoint(is_best)
            
        print(f"\nEntrenamiento completado. Mejor mIoU: {self.best_miou:.2f}%")
        self.writer.close()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['learning_rate'] = float(config.get('learning_rate', 0.001))
    config['weight_decay'] = float(config.get('weight_decay', 0.0001))
    config['poly_power'] = float(config.get('poly_power', 0.9))
    config['train_split'] = float(config.get('train_split', 0.8))
    config['batch_size'] = int(config.get('batch_size', 4))
    config['epochs'] = int(config.get('epochs', 100))
    config['num_classes'] = int(config.get('num_classes', 2))
    config['model_width'] = int(config.get('model_width', 32))
    config['num_workers'] = int(config.get('num_workers', 4))
    config['seed'] = int(config.get('seed', 42))
    config['val_frequency'] = int(config.get('val_frequency', 5))
    config['save_frequency'] = int(config.get('save_frequency', 10))
    
    return config

def main():
    parser = argparse.ArgumentParser(description='MSFANet Training')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--model', type=str, choices=['auto', 'original', 'simple'], 
                       default='auto', help='Model type to use')
    
    args = parser.parse_args()
    
    # Load config
    if not os.path.exists(args.config):
        print(f"Config file {args.config} not found")
        return
        
    config = load_config(args.config)
    config['model_type'] = args.model
    
    try:
        trainer = MSFANetTrainer(config)
        
        trainer.train()
        
    except Exception as e:
        print(f"Error durante entrenamiento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
