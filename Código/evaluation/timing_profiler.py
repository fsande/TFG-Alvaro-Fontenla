#!/usr/bin/env python3

"""
Timing profiler for comparative study
Handles all timing measurements and performance analysis
"""

import time
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class TimingProfiler:
    """
    Comprehensive timing profiler for deep learning models
    Tracks training time, inference time, memory usage, and other performance metrics
    """
    
    def __init__(self, device: str = 'auto', output_dir: str = 'timing_results'):
        """
        Initialize the timing profiler
        
        Args:
            device: 'cuda', 'cpu', or 'auto'
            output_dir: Directory to save timing results
        """
        self.device = self._setup_device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Storage for timing data
        self.timing_data = defaultdict(dict)
        self.current_session = None
        
        # Timing accumulators
        self.epoch_times = []
        self.batch_times = []
        self.forward_times = []
        self.backward_times = []
        self.validation_times = []
        
        # Memory tracking
        self.memory_snapshots = []
        
        print(f"TimingProfiler initialized on device: {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup the compute device"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)
    
    def start_session(self, model_name: str, config: Dict[str, Any] = None):
        """Start a new timing session for a model"""
        self.current_session = model_name
        self.timing_data[model_name] = {
            'model_name': model_name,
            'config': config or {},
            'start_time': time.time(),
            'epoch_times': [],
            'batch_times': [],
            'forward_times': [],
            'backward_times': [],
            'validation_times': [],
            'memory_snapshots': []
        }
        
        # Clear accumulators
        self.epoch_times = []
        self.batch_times = []
        self.forward_times = []
        self.backward_times = []
        self.validation_times = []
        self.memory_snapshots = []
        
        print(f"Started timing session for: {model_name}")
    
    def end_session(self):
        """End the current timing session"""
        if self.current_session:
            self.timing_data[self.current_session]['end_time'] = time.time()
            self.timing_data[self.current_session]['total_time'] = (
                self.timing_data[self.current_session]['end_time'] - 
                self.timing_data[self.current_session]['start_time']
            )
            
            # Store accumulated data
            self.timing_data[self.current_session]['epoch_times'] = self.epoch_times.copy()
            self.timing_data[self.current_session]['batch_times'] = self.batch_times.copy()
            self.timing_data[self.current_session]['forward_times'] = self.forward_times.copy()
            self.timing_data[self.current_session]['backward_times'] = self.backward_times.copy()
            self.timing_data[self.current_session]['validation_times'] = self.validation_times.copy()
            self.timing_data[self.current_session]['memory_snapshots'] = self.memory_snapshots.copy()
            
            print(f"Ended timing session for: {self.current_session}")
            self.current_session = None
    
    def measure_inference_time(self, model, input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
                              msi_shape: Tuple[int, int, int, int] = (1, 10, 512, 512),
                              num_iterations: int = 100, warmup_iterations: int = 10) -> float:
        """
        Measure model inference time
        
        Args:
            model: PyTorch model to measure
            input_shape: Shape of RGB input (B, C, H, W)
            msi_shape: Shape of MSI input (B, C, H, W)
            num_iterations: Number of inference iterations
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Average inference time in milliseconds
        """
        model.eval()
        
        # Create dummy inputs
        rgb = torch.randn(input_shape).to(self.device)
        msi = torch.randn(msi_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(rgb, msi)
        
        # Synchronize before measurement
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Measure inference time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(rgb, msi)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        # Calculate average time in milliseconds
        avg_time = (end_time - start_time) / num_iterations * 1000
        
        return avg_time
    
    def start_epoch_timing(self):
        """Start timing an epoch"""
        self.epoch_start_time = time.time()
    
    def end_epoch_timing(self):
        """End timing an epoch"""
        if hasattr(self, 'epoch_start_time'):
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            return epoch_time
        return 0
    
    def start_batch_timing(self):
        """Start timing a batch"""
        self.batch_start_time = time.time()
    
    def end_batch_timing(self):
        """End timing a batch"""
        if hasattr(self, 'batch_start_time'):
            batch_time = time.time() - self.batch_start_time
            self.batch_times.append(batch_time)
            return batch_time
        return 0
    
    def start_forward_timing(self):
        """Start timing forward pass"""
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        self.forward_start_time = time.time()
    
    def end_forward_timing(self):
        """End timing forward pass"""
        if hasattr(self, 'forward_start_time'):
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            forward_time = time.time() - self.forward_start_time
            self.forward_times.append(forward_time)
            return forward_time
        return 0
    
    def start_backward_timing(self):
        """Start timing backward pass"""
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        self.backward_start_time = time.time()
    
    def end_backward_timing(self):
        """End timing backward pass"""
        if hasattr(self, 'backward_start_time'):
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            backward_time = time.time() - self.backward_start_time
            self.backward_times.append(backward_time)
            return backward_time
        return 0
    
    def start_validation_timing(self):
        """Start timing validation"""
        self.validation_start_time = time.time()
    
    def end_validation_timing(self):
        """End timing validation"""
        if hasattr(self, 'validation_start_time'):
            validation_time = time.time() - self.validation_start_time
            self.validation_times.append(validation_time)
            return validation_time
        return 0
    
    def record_memory_usage(self, label: str = ""):
        """Record current memory usage"""
        if self.device.type == 'cuda':
            memory_info = {
                'label': label,
                'timestamp': time.time(),
                'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'cached_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024
            }
            self.memory_snapshots.append(memory_info)
        else:
            # For CPU, we can use psutil if available
            try:
                import psutil
                memory_info = {
                    'label': label,
                    'timestamp': time.time(),
                    'allocated_mb': psutil.virtual_memory().used / 1024 / 1024,
                    'cached_mb': 0,
                    'max_allocated_mb': psutil.virtual_memory().available / 1024 / 1024
                }
                self.memory_snapshots.append(memory_info)
            except ImportError:
                pass
    
    def get_timing_statistics(self, model_name: str) -> Dict[str, Any]:
        """
        Get comprehensive timing statistics for a model
        
        Args:
            model_name: Name of the model to analyze
            
        Returns:
            Dictionary with timing statistics
        """
        if model_name not in self.timing_data:
            return {}
        
        data = self.timing_data[model_name]
        
        stats = {
            'model_name': model_name,
            'total_training_time_minutes': data.get('total_time', 0) / 60,
            'avg_epoch_time_seconds': np.mean(data['epoch_times']) if data['epoch_times'] else 0,
            'avg_batch_time_ms': np.mean(data['batch_times']) * 1000 if data['batch_times'] else 0,
            'avg_forward_time_ms': np.mean(data['forward_times']) * 1000 if data['forward_times'] else 0,
            'avg_backward_time_ms': np.mean(data['backward_times']) * 1000 if data['backward_times'] else 0,
            'avg_validation_time_seconds': np.mean(data['validation_times']) if data['validation_times'] else 0,
        }
        
        # Memory statistics
        if data['memory_snapshots']:
            peak_memory = max(snap['allocated_mb'] for snap in data['memory_snapshots'])
            current_memory = data['memory_snapshots'][-1]['allocated_mb']
            cached_memory = data['memory_snapshots'][-1]['cached_mb']
            
            stats['memory_usage'] = {
                'peak_memory_mb': peak_memory,
                'current_memory_mb': current_memory,
                'cached_memory_mb': cached_memory
            }
        
        # Statistical measures
        if data['epoch_times']:
            stats['statistics'] = {
                'total_epochs': len(data['epoch_times']),
                'total_batches': len(data['batch_times']),
                'forward_time_std': np.std(data['forward_times']) * 1000 if data['forward_times'] else 0,
                'backward_time_std': np.std(data['backward_times']) * 1000 if data['backward_times'] else 0,
                'batch_time_std': np.std(data['batch_times']) * 1000 if data['batch_times'] else 0,
            }
        
        return stats
    
    def save_timing_analysis(self, model_name: str, output_file: Optional[str] = None):
        """
        Save timing analysis to JSON file
        
        Args:
            model_name: Name of the model
            output_file: Optional output file path
        """
        if output_file is None:
            output_file = self.output_dir / f"{model_name}_timing_analysis.json"
        
        stats = self.get_timing_statistics(model_name)
        
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Timing analysis saved to: {output_file}")
    
    def save_batch_times(self, model_name: str, output_file: Optional[str] = None):
        """Save batch times to CSV file"""
        if output_file is None:
            output_file = self.output_dir / f"{model_name}_batch_times.csv"
        
        if model_name in self.timing_data and self.timing_data[model_name]['batch_times']:
            df = pd.DataFrame({
                'batch_index': range(len(self.timing_data[model_name]['batch_times'])),
                'batch_time_ms': [t * 1000 for t in self.timing_data[model_name]['batch_times']]
            })
            df.to_csv(output_file, index=False)
            print(f"Batch times saved to: {output_file}")
    
    def save_epoch_times(self, model_name: str, output_file: Optional[str] = None):
        """Save epoch times to CSV file"""
        if output_file is None:
            output_file = self.output_dir / f"{model_name}_epoch_times.csv"
        
        if model_name in self.timing_data and self.timing_data[model_name]['epoch_times']:
            df = pd.DataFrame({
                'epoch': range(len(self.timing_data[model_name]['epoch_times'])),
                'epoch_time_seconds': self.timing_data[model_name]['epoch_times']
            })
            df.to_csv(output_file, index=False)
            print(f"Epoch times saved to: {output_file}")
    
    def compare_models(self, model_names: List[str]) -> pd.DataFrame:
        """
        Compare timing statistics across multiple models
        
        Args:
            model_names: List of model names to compare
            
        Returns:
            DataFrame with comparative statistics
        """
        comparison_data = []
        
        for model_name in model_names:
            stats = self.get_timing_statistics(model_name)
            if stats:
                comparison_data.append(stats)
        
        if not comparison_data:
            return pd.DataFrame()
        
        # Create comparison DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Select key metrics for comparison
        key_metrics = [
            'model_name', 'total_training_time_minutes', 'avg_epoch_time_seconds',
            'avg_batch_time_ms', 'avg_forward_time_ms', 'avg_backward_time_ms',
            'avg_validation_time_seconds'
        ]
        
        # Add memory metrics if available
        if 'memory_usage' in df.columns:
            df['peak_memory_mb'] = df['memory_usage'].apply(
                lambda x: x.get('peak_memory_mb', 0) if isinstance(x, dict) else 0
            )
            key_metrics.append('peak_memory_mb')
        
        return df[key_metrics]
    
    def generate_efficiency_report(self, model_names: List[str], accuracy_scores: Dict[str, float],
                                  output_file: Optional[str] = None) -> str:
        """
        Generate efficiency report comparing time vs accuracy
        
        Args:
            model_names: List of model names
            accuracy_scores: Dictionary mapping model names to accuracy scores
            output_file: Optional output file path
            
        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("TEMPORAL EFFICIENCY ANALYSIS")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Calculate efficiency metrics
        efficiency_data = []
        for model_name in model_names:
            stats = self.get_timing_statistics(model_name)
            if stats and model_name in accuracy_scores:
                accuracy = accuracy_scores[model_name]
                training_time = stats['total_training_time_minutes']
                inference_time = stats['avg_batch_time_ms']
                
                # Calculate efficiency scores
                time_efficiency = accuracy / training_time if training_time > 0 else 0
                inference_efficiency = accuracy / inference_time if inference_time > 0 else 0
                
                efficiency_data.append({
                    'model_name': model_name,
                    'accuracy': accuracy,
                    'training_time_min': training_time,
                    'inference_time_ms': inference_time,
                    'time_efficiency': time_efficiency,
                    'inference_efficiency': inference_efficiency,
                    'efficiency_score': (time_efficiency + inference_efficiency) / 2
                })
        
        # Sort by efficiency score
        efficiency_data.sort(key=lambda x: x['efficiency_score'], reverse=True)
        
        # Generate ranking
        report_lines.append("EFFICIENCY RANKING:")
        report_lines.append("-" * 30)
        for i, data in enumerate(efficiency_data, 1):
            report_lines.append(f"{i}. {data['model_name']}: Score = {data['efficiency_score']:.3f}")
        report_lines.append("")
        
        # Best in each category
        report_lines.append("BEST IN EACH CATEGORY:")
        report_lines.append("-" * 30)
        
        fastest_training = min(efficiency_data, key=lambda x: x['training_time_min'])
        fastest_inference = min(efficiency_data, key=lambda x: x['inference_time_ms'])
        most_accurate = max(efficiency_data, key=lambda x: x['accuracy'])
        most_time_efficient = max(efficiency_data, key=lambda x: x['time_efficiency'])
        most_inference_efficient = max(efficiency_data, key=lambda x: x['inference_efficiency'])
        most_efficient_overall = max(efficiency_data, key=lambda x: x['efficiency_score'])
        
        report_lines.append(f"Fastest training: {fastest_training['model_name']} ({fastest_training['training_time_min']:.1f} min)")
        report_lines.append(f"Fastest inference: {fastest_inference['model_name']} ({fastest_inference['inference_time_ms']:.1f} ms)")
        report_lines.append(f"Most accurate: {most_accurate['model_name']} ({most_accurate['accuracy']:.1f}%)")
        report_lines.append(f"Most time efficient: {most_time_efficient['model_name']} ({most_time_efficient['time_efficiency']:.3f} acc/min)")
        report_lines.append(f"Most inference efficient: {most_inference_efficient['model_name']} ({most_inference_efficient['inference_efficiency']:.3f} acc/ms)")
        report_lines.append(f"Best overall efficiency: {most_efficient_overall['model_name']} (Score: {most_efficient_overall['efficiency_score']:.3f})")
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Efficiency report saved to: {output_file}")
        
        return report_text
    
    def clear_session_data(self, model_name: str):
        """Clear timing data for a specific model"""
        if model_name in self.timing_data:
            del self.timing_data[model_name]
        print(f"Cleared timing data for: {model_name}")
    
    def clear_all_data(self):
        """Clear all timing data"""
        self.timing_data.clear()
        print("Cleared all timing data")
    
    def get_model_comparison_summary(self, model_names: List[str]) -> Dict[str, Any]:
        """Get a summary comparison of models"""
        comparison_data = {}
        
        for model_name in model_names:
            stats = self.get_timing_statistics(model_name)
            if stats:
                comparison_data[model_name] = {
                    'training_time_min': stats['total_training_time_minutes'],
                    'inference_time_ms': stats['avg_batch_time_ms'],
                    'memory_peak_mb': stats.get('memory_usage', {}).get('peak_memory_mb', 0),
                    'epoch_time_s': stats['avg_epoch_time_seconds']
                }
        
        return comparison_data

class TimingContext:
    """Context manager for timing operations"""
    
    def __init__(self, profiler: TimingProfiler, operation_type: str):
        self.profiler = profiler
        self.operation_type = operation_type
        self.start_method = getattr(profiler, f"start_{operation_type}_timing")
        self.end_method = getattr(profiler, f"end_{operation_type}_timing")
    
    def __enter__(self):
        self.start_method()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.end_method()

def create_timing_profiler(device: str = 'auto', output_dir: str = 'timing_results') -> TimingProfiler:
    """Create a timing profiler instance"""
    return TimingProfiler(device=device, output_dir=output_dir)