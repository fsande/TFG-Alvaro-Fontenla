#!/usr/bin/env python3

"""
Comparative study of architectures for satellite imagery segmentation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
from pathlib import Path
from collections import defaultdict
import yaml
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from sentinel2_dataset import Sentinel2Dataset
except ImportError:
    print("Error: Dataset Sentinel-2 not available")

from metrics import PerformanceMetrics
from timing_profiler import create_timing_profiler

from models import (
    LinkNet,
    DLinkNet,
    HRNet,
    CCNet,
    DANet,
    MSFANet
)

class ArchitectureRegistry:
    """Registry of all architectures to compare"""
    
    def __init__(self):
        self.architectures = {}
        self.register_architectures()
    
    def register_architectures(self):
        """Register selected architectures"""
        
        # LinkNet
        self.architectures['LinkNet'] = {
            'model_class': self.get_linknet,
            'description': 'Encoder-decoder with skip connections',
            'paper_year': 2017
        }
        
        # D-LinkNet 
        self.architectures['D-LinkNet'] = {
            'model_class': self.get_dlinknet,
            'description': 'LinkNet with dilated convolutions',
            'paper_year': 2018
        }
        
        # HRNet
        self.architectures['HRNet'] = {
            'model_class': self.get_hrnet,
            'description': 'High-Resolution Network (HRNet)',
            'paper_year': 2019
        }
        
        # CCNet
        self.architectures['CCNet'] = {
            'model_class': self.get_ccnet,
            'description': 'Criss-Cross Attention Network (CCNet)',
            'paper_year': 2019
        }
        
        # DANet
        self.architectures['DANet'] = {
            'model_class': self.get_danet,
            'description': 'Dual Attention Network (DANet)',
            'paper_year': 2019
        }
        
        # MSFANet
        self.architectures['MSFANet'] = {
            'model_class': self.get_msfanet,
            'description': 'Multiscale Fusion Attention Network',
            'paper_year': 2023
        }
    
    def get_linknet(self, **kwargs):
        """Implementation of LinkNet"""
        return LinkNet(**kwargs)
    
    def get_dlinknet(self, **kwargs):
        """Implementation of D-LinkNet"""
        return DLinkNet(**kwargs)
    
    def get_hrnet(self, **kwargs):
        """Implementation of HRNet"""
        return HRNet(**kwargs)
    
    def get_ccnet(self, **kwargs):
        """Implementation of CCNet"""
        return CCNet(**kwargs)
    
    def get_danet(self, **kwargs):
        """Implementation of DANet"""
        return DANet(**kwargs)
    
    def get_msfanet(self, **kwargs):
        """Implementation of MSFANet"""
        return MSFANet(**kwargs)

class ComparativeStudy:
    """
    Main class for the comparative study.
    """
    
    def __init__(self, config_path='config_sentinel2.yaml'):
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.registry = ArchitectureRegistry()
        self.results = defaultdict(dict)
        
        # Create result directories
        self.results_dir = Path('comparative_study_results')
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / 'models').mkdir(exist_ok=True)
        (self.results_dir / 'plots').mkdir(exist_ok=True)
        (self.results_dir / 'metrics').mkdir(exist_ok=True)
        (self.results_dir / 'timing').mkdir(exist_ok=True)
        
        # Initialize timing profiler
        self.timing_profiler = create_timing_profiler(
            device=str(self.device),
            output_dir=str(self.results_dir / 'timing')
        )
        
        print(f"Comparative study initialized on {self.device}")
        print(f"Architectures to evaluate: {list(self.registry.architectures.keys())}")
    
    def load_config(self, config_path):
        """Load configuration from YAML"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'data_dir': 'processed_sentinel2',
                'num_classes': 2,
                'rgb_channels': 3,
                'msi_channels': 10,
                'model_width': 32,
                'epochs': 50,
                'batch_size': 4,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'num_workers': 2,
                'train_split': 0.8,
                'val_split': 0.2,
            }
    
    def create_data_loaders(self):
        """Create data loaders for training and validation"""
        try:
            dataset = Sentinel2Dataset(self.config['data_dir'])
            total_size = len(dataset)
            
            if total_size == 0:
                print("Warning: Empty dataset, creating dummy dataset")
                return self.create_dummy_loaders()
            
            # Verify the actual dimensions of the dataset
            sample = dataset[0]
            if 'ms' in sample:
                actual_msi_channels = sample['ms'].shape[0]
                print(f"MSI channels detected in the dataset: {actual_msi_channels}")
                
                # Update configuration if necessary
                if actual_msi_channels != self.config['msi_channels']:
                    print(f"Actualizando canales MSI: {self.config['msi_channels']} → {actual_msi_channels}")
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
            print(f"Error creating data loaders: {e}")
            raise e
    
    def train_model(self, model_name, epochs=None):
        """Train a specific model with error handling and timing profiling"""
        epochs = epochs or self.config['epochs']
        
        print(f"\n=== Training {model_name} ===")
        
        # Start timing session
        self.timing_profiler.start_session(model_name, self.config)
        
        try:
            # Create model
            model_args = {
                'num_classes': self.config['num_classes'],
                'rgb_channels': self.config['rgb_channels'],
                'msi_channels': self.config['msi_channels'],
                'width': self.config.get('model_width', 32)
            }
            
            print(f"Model parameters: {model_args}")
            
            model = self.registry.architectures[model_name]['model_class'](**model_args)
            model = model.to(self.device)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total parameters: {total_params:,}")
            
            # Optimizer and loss
            optimizer = optim.AdamW(model.parameters(), 
                                   lr=self.config['learning_rate'],
                                   weight_decay=self.config['weight_decay'])
            
            # Loss with weights for unbalanced classes
            if self.config.get('use_class_weights', False) and self.config['num_classes'] > 2:
                # For multiclass configuration
                weights = [1.0, self.config.get('road_weight', 8.0)]
                if self.config['num_classes'] > 2:
                    weights.append(self.config.get('building_weight', 5.0))
                weights = torch.tensor(weights[:self.config['num_classes']]).to(self.device)
                criterion = nn.CrossEntropyLoss(weight=weights)
            else:
                criterion = nn.CrossEntropyLoss()
            
            # Data loaders
            train_loader, val_loader = self.create_data_loaders()
            
            # Test with a batch to verify compatibility
            print("Verifying model compatibility...")
            model.eval()
            with torch.no_grad():
                test_batch = next(iter(train_loader))
                rgb = test_batch['rgb'].to(self.device)
                msi = test_batch['ms'].to(self.device)
                target = test_batch['mask'].to(self.device)
                
                print(f"RGB input: {rgb.shape}")
                print(f"MSI input: {msi.shape}")
                print(f"Target: {target.shape}")
                
                output = model(rgb, msi)
                
                if output.shape[0] != target.shape[0]:
                    raise ValueError(f"Batch size mismatch: output {output.shape[0]} vs target {target.shape[0]}")
                
                if output.shape[-2:] != target.shape[-2:]:
                    print(f"Warning: Resizing output from {output.shape[-2:]} to {target.shape[-2:]}")
                    output = torch.nn.functional.interpolate(
                        output, size=target.shape[-2:], mode='bilinear', align_corners=False
                    )
                
                # Test loss
                loss = criterion(output, target)
                print(f"Test loss: {loss.item():.4f}")
            
            print("Verification completed. Starting training...")
            
            # Training
            model.train()
            train_losses = []
            val_metrics_history = []
            
            for epoch in range(epochs):
                # Start epoch timing
                self.timing_profiler.start_epoch_timing()
                self.timing_profiler.record_memory_usage(f"epoch_{epoch}_start")
                
                epoch_loss = 0.0
                num_batches = 0
                
                progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
                
                for batch_idx, batch in enumerate(progress_bar):
                    try:
                        # Start batch timing
                        self.timing_profiler.start_batch_timing()
                        
                        rgb = batch['rgb'].to(self.device)
                        msi = batch['ms'].to(self.device)
                        target = batch['mask'].to(self.device)
                        
                        optimizer.zero_grad()
                        
                        # Forward pass with timing
                        self.timing_profiler.start_forward_timing()
                        output = model(rgb, msi)
                        self.timing_profiler.end_forward_timing()
                        
                        # Ensure compatible dimensions
                        if output.shape[-2:] != target.shape[-2:]:
                            output = torch.nn.functional.interpolate(
                                output, size=target.shape[-2:], mode='bilinear', align_corners=False
                            )
                        
                        loss = criterion(output, target)
                        
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"Warning: Invalid loss in batch {batch_idx}: {loss.item()}")
                            continue
                        
                        # Backward pass with timing
                        self.timing_profiler.start_backward_timing()
                        loss.backward()
                        self.timing_profiler.end_backward_timing()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        num_batches += 1
                        
                        # End batch timing
                        self.timing_profiler.end_batch_timing()
                        
                        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
                        
                    except Exception as e:
                        print(f"Error in batch {batch_idx}: {e}")
                        continue
                
                if num_batches == 0:
                    print(f"Warning: No batches processed in epoch {epoch+1}")
                    continue
                
                avg_loss = epoch_loss / num_batches
                train_losses.append(avg_loss)
                
                # Record memory usage at end of epoch
                self.timing_profiler.record_memory_usage(f"epoch_{epoch}_end")
                
                # End epoch timing
                self.timing_profiler.end_epoch_timing()
                
                # Validation every 5 epochs
                if (epoch + 1) % 5 == 0:
                    self.timing_profiler.start_validation_timing()
                    val_metrics = self.evaluate_model(model, val_loader, model_name)
                    self.timing_profiler.end_validation_timing()
                    val_metrics_history.append(val_metrics)
                    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val IoU={val_metrics['road_iou']:.2f}%")
            
            # Final evaluation
            self.timing_profiler.start_validation_timing()
            final_metrics = self.evaluate_model(model, val_loader, model_name)
            self.timing_profiler.end_validation_timing()
            
            # Calculate parameters and inference time
            num_params = sum(p.numel() for p in model.parameters()) / 1e6  # In millions
            inference_time = self.measure_inference_time(model)
            
            # End timing session and save timing analysis
            self.timing_profiler.end_session()
            self.timing_profiler.save_timing_analysis(model_name)
            self.timing_profiler.save_batch_times(model_name)
            self.timing_profiler.save_epoch_times(model_name)
            
            # Get timing statistics for results
            timing_stats = self.timing_profiler.get_timing_statistics(model_name)
            
            # Save results
            self.results[model_name] = {
                'metrics': final_metrics,
                'num_parameters': num_params,
                'inference_time': inference_time,
                'train_losses': train_losses,
                'val_metrics_history': val_metrics_history,
                'description': self.registry.architectures[model_name]['description'],
                'timing_stats': timing_stats
            }
            
            # Save model
            model_path = self.results_dir / 'models' / f'{model_name}.pth'
            torch.save(model.state_dict(), model_path)
            
            print(f"{model_name} completed - IoU: {final_metrics['road_iou']:.2f}%, "
                  f"Params: {num_params:.2f}M, Time: {inference_time:.2f}ms")
            print(f"Training time: {timing_stats.get('total_training_time_minutes', 0):.1f} min")
            
            return model
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # End timing session even if there's an error
            if self.timing_profiler.current_session:
                self.timing_profiler.end_session()
            
            return None
    
    def evaluate_model(self, model, val_loader, model_name):
        """Evaluate a model with error handling"""
        model.eval()
        metrics_calculator = PerformanceMetrics(
            num_classes=self.config['num_classes'],
            target_class=1  # Road class
        )
        
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    rgb = batch['rgb'].to(self.device)
                    msi = batch['ms'].to(self.device)
                    target = batch['mask'].to(self.device)
                    
                    output = model(rgb, msi)
                    
                    # Ensure compatible dimensions
                    if output.shape[-2:] != target.shape[-2:]:
                        output = torch.nn.functional.interpolate(
                            output, size=target.shape[-2:], mode='bilinear', align_corners=False
                        )
                    
                    pred = torch.argmax(output, dim=1)
                    
                    # Verify that the predictions are in the valid range
                    pred = torch.clamp(pred, 0, self.config['num_classes'] - 1)
                    target = torch.clamp(target, 0, self.config['num_classes'] - 1)
                    
                    metrics_calculator.update(pred, target)
                    total_samples += pred.numel()
                    
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        metrics = metrics_calculator.compute_metrics()
        
        # Add additional information
        metrics['total_samples'] = total_samples
        
        if total_samples == 0:
            print(f"Warning: No samples processed in validation for {model_name}")
        
        return metrics
    
    def measure_inference_time(self, model, num_iterations=100):
        """Measure inference time using timing profiler"""
        input_shape = (1, self.config['rgb_channels'], 512, 512)
        msi_shape = (1, self.config['msi_channels'], 512, 512)
        
        return self.timing_profiler.measure_inference_time(
            model=model,
            input_shape=input_shape,
            msi_shape=msi_shape,
            num_iterations=num_iterations,
            warmup_iterations=10
        )
    
    def run_complete_study(self, epochs=None):
        """Run complete study"""
        architectures = list(self.registry.architectures.keys())
        
        print(f"Starting comparative study of {len(architectures)} architectures")
        print(f"Configuration: {self.config}")
        
        for arch_name in architectures:
            try:
                self.train_model(arch_name, epochs)
            except Exception as e:
                print(f"Error training {arch_name}: {e}")
                continue
        
        print(f"\nStudy completed. Results saved in {self.results_dir}")
    
def main():
    """Main function"""
    print("=== COMPARATIVE STUDY ===")
    print("=" * 50)
    
    # Create configuration if it doesn't exist
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        create_comparative_config(config_path)
        print(f"Configuration created in {config_path}")
    
    # Initialize study
    study = ComparativeStudy(config_path)
    
    # Run study
    study.run_complete_study(epochs=300)
    
    print("\n=== STUDY COMPLETED ===")
    print(f"Check results in: {study.results_dir}")
    print("Generated files:")
    print("- comparative_report.txt: Report")
    print("- metrics/comparative_results.csv: Results table")

def create_comparative_config(config_path):
    """Create configuration for the comparative study"""
    config = {
        'data_dir': 'processed_sentinel2',
        'num_classes': 2,  # Binary configuration
        'model_width': 32,  # Standard width
        'rgb_channels': 3,
        'msi_channels': 9,  # MSI channels of Sentinel-2
        
        # Training configuration
        'epochs': 300,
        'batch_size': 4,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'num_workers': 2,
        
        # Class weights for unbalanced classes
        'use_class_weights': True,
        'road_weight': 8.0,
        
        # Data split
        'train_split': 0.8,
        'val_split': 0.2,
        
        # Validation and saving frequencies
        'val_frequency': 5,
        'save_frequency': 10,
        
        # Hardware
        'device': 'cuda',
        
        # Sentinel-2 specific
        'sentinel2': {
            'target_resolution': 10,
            'osm_buffer_meters': 5,
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