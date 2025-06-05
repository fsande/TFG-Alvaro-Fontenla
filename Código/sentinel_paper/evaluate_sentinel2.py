#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import yaml
import argparse
from tqdm import tqdm
import cv2
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import time
from collections import defaultdict

try:
    from model_adapter import get_msfanet, get_simple_msfanet
    MODEL_ADAPTER_AVAILABLE = True
except ImportError:
    MODEL_ADAPTER_AVAILABLE = False
    try:
        from msfanet_model import get_msfanet
        from simple_msfanet import get_simple_msfanet
    except ImportError:
        print("No se pudieron importar los modelos")

# Dataset
try:
    from sentinel2_processor_fixed import Sentinel2Dataset
except ImportError:
    from sentinel2_processor import Sentinel2Dataset

class Sentinel2Evaluator:
    def __init__(self, config_path, checkpoint_path, output_dir="evaluation_results"):
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cargar modelo
        self.model = self.load_model(checkpoint_path)
        if self.model is None:
            raise ValueError("No se pudo cargar el modelo")
        
        # Dataset de evaluación
        self.dataset = Sentinel2Dataset(self.config['data_dir'])
        
        # Crear división de datos
        torch.manual_seed(self.config.get('seed', 42))
        train_size = int(self.config.get('train_split', 0.8) * len(self.dataset))
        val_size = len(self.dataset) - train_size
        _, self.eval_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])
        
        print(f"Evaluador Sentinel-2 inicializado:")
        print(f"   Modelo: {self.model.__class__.__name__}")
        print(f"   Device: {self.device}")
        print(f"   Muestras evaluación: {len(self.eval_dataset)}")
        print(f"   Salida: {self.output_dir}")
        
        # Métricas acumuladas
        self.results = {
            'predictions': [],
            'targets': [],
            'metrics': {},
            'sample_results': []
        }
    
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_model(self, checkpoint_path):
        print(f"Cargando modelo desde: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            if 'config' in checkpoint:
                model_config = checkpoint['config']
            else:
                model_config = self.config
            
            if MODEL_ADAPTER_AVAILABLE:
                model_args = {
                    'num_classes': model_config['num_classes'],
                    'width': model_config.get('model_width', 32),
                    'rgb_channels': model_config.get('rgb_channels', 3),
                    'msi_channels': model_config.get('msi_channels', 10)
                }
                
                try:
                    model = get_msfanet(**model_args)
                except:
                    model = get_simple_msfanet(**model_args)
            else:
                model = get_msfanet(
                    num_classes=model_config['num_classes'],
                    width=model_config.get('model_width', 32)
                )
            
            # Cargar pesos
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            
            print(f"Modelo cargado exitosamente")
            if 'epoch' in checkpoint:
                print(f"   Época: {checkpoint['epoch']}")
            if 'best_miou' in checkpoint:
                print(f"   Mejor mIoU: {checkpoint['best_miou']:.2f}%")
            
            return model
            
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return None
    
    def predict_sample(self, rgb, msi):
        with torch.no_grad():
            rgb = rgb.to(self.device)
            msi = msi.to(self.device)
            
            # Predicción
            output = self.model(rgb, msi)
            
            # Convertir a probabilidades y clases
            probs = F.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1)
            
            return pred.cpu(), probs.cpu()
    
    def evaluate_dataset(self, max_samples=None):
        print("Iniciando evaluación...")
        
        dataloader = torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=False,
            num_workers=2
        )
        
        all_predictions = []
        all_targets = []
        all_probs = []
        sample_results = []
        
        total_time = 0
        num_samples = 0
        
        progress_bar = tqdm(dataloader, desc="Evaluando")
        
        for batch_idx, batch in enumerate(progress_bar):
            if max_samples and num_samples >= max_samples:
                break
            
            rgb = batch['rgb']
            msi = batch['ms']
            target = batch['mask']
            
            batch_size = rgb.size(0)
            
            start_time = time.time()
            pred, probs = self.predict_sample(rgb, msi)
            inference_time = time.time() - start_time
            
            total_time += inference_time
            num_samples += batch_size
            
            # Acumular resultados
            all_predictions.append(pred)
            all_targets.append(target)
            all_probs.append(probs)
            
            # Guardar resultados por muestra para visualización
            for i in range(batch_size):
                sample_results.append({
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'rgb': rgb[i].numpy(),
                    'msi': msi[i].numpy(),
                    'target': target[i].numpy(),
                    'prediction': pred[i].numpy(),
                    'probabilities': probs[i].numpy(),
                    'inference_time': inference_time / batch_size
                })
            
            # Actualizar progreso
            avg_time = total_time / num_samples
            progress_bar.set_postfix({
                'Tiempo/muestra': f'{avg_time*1000:.1f}ms',
                'Muestras': num_samples
            })
        
        # Concatenar todos los resultados
        self.results['predictions'] = torch.cat(all_predictions, dim=0)
        self.results['targets'] = torch.cat(all_targets, dim=0)
        self.results['probabilities'] = torch.cat(all_probs, dim=0)
        self.results['sample_results'] = sample_results
        self.results['inference_stats'] = {
            'total_time': total_time,
            'num_samples': num_samples,
            'avg_time_per_sample': total_time / num_samples,
            'fps': num_samples / total_time
        }
        
        print(f"Evaluación completada:")
        print(f"   Muestras procesadas: {num_samples}")
        print(f"   Tiempo promedio: {self.results['inference_stats']['avg_time_per_sample']*1000:.1f}ms/muestra")
        print(f"   FPS: {self.results['inference_stats']['fps']:.1f}")
    
    def calculate_metrics(self):
        print("Calculando métricas...")
        
        predictions = self.results['predictions'].numpy().flatten()
        targets = self.results['targets'].numpy().flatten()
        probs = self.results['probabilities']
        
        # Matriz de confusión
        cm = confusion_matrix(targets, predictions, labels=[0, 1])
        
        # Métricas: true negatives, false positives, false negatives, true positives
        tn, fp, fn, tp = cm.ravel()
        
        # Calcular métricas
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # IoU por clase
        iou_background = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0
        iou_road = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        miou = (iou_background + iou_road) / 2
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # Específico para carreteras
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Curvas ROC y PR para clase road (índice 1)
        road_probs = probs[:, 1].numpy().flatten()
        
        # ROC
        fpr, tpr, roc_thresholds = roc_curve(targets, road_probs)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(targets, road_probs)
        pr_auc = auc(pr_recall, pr_precision)
        
        # Balanceo de clases
        road_pixels = np.sum(targets == 1)
        background_pixels = np.sum(targets == 0)
        class_balance = road_pixels / (road_pixels + background_pixels)
        
        self.results['metrics'] = {
            'confusion_matrix': cm,
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'specificity': specificity * 100,
            'iou_background': iou_background * 100,
            'iou_road': iou_road * 100,
            'miou': miou * 100,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'class_balance': class_balance * 100,
            'road_pixels': int(road_pixels),
            'background_pixels': int(background_pixels),
            'curves': {
                'roc': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds},
                'pr': {'precision': pr_precision, 'recall': pr_recall, 'thresholds': pr_thresholds}
            }
        }
        
        # Reporte de clasificación
        class_report = classification_report(
            targets, predictions,
            target_names=['Background', 'Road'],
            output_dict=True
        )
        self.results['classification_report'] = class_report
        
        print(f"Métricas calculadas:")
        print(f"   Accuracy: {accuracy*100:.2f}%")
        print(f"   Precision (Road): {precision*100:.2f}%")
        print(f"   Recall (Road): {recall*100:.2f}%")
        print(f"   F1-Score (Road): {f1*100:.2f}%")
        print(f"   mIoU: {miou*100:.2f}%")
        print(f"   Road IoU: {iou_road*100:.2f}%")
        print(f"   ROC AUC: {roc_auc:.3f}")
        print(f"   Balance clases: {class_balance*100:.1f}% carreteras")
    
    def generate_visualizations(self, num_samples=20):
        print("Generando visualizaciones...")
        
        # 1. Métricas generales
        self.plot_metrics_summary()
        
        # 2. Matriz de confusión
        self.plot_confusion_matrix()
        
        # 3. Curvas ROC y PR
        self.plot_roc_pr_curves()
        
        # 4. Distribución de probabilidades
        self.plot_probability_distributions()
        
        # 5. Muestras de predicciones
        self.plot_prediction_samples(num_samples)
        
        # 6. Análisis de errores
        self.plot_error_analysis()
        
        print(f"Visualizaciones guardadas en: {self.output_dir}")
    
    def plot_metrics_summary(self):
        metrics = self.results['metrics']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Gráfico de barras con métricas principales
        main_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'miou']
        values = [metrics[m] for m in main_metrics]
        labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'mIoU']
        
        bars = ax1.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax1.set_ylabel('Porcentaje (%)')
        ax1.set_title('Métricas Principales de Evaluación')
        ax1.set_ylim(0, 100)
        
        # Agregar valores en las barras
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # IoU por clase
        iou_values = [metrics['iou_background'], metrics['iou_road']]
        iou_labels = ['Background', 'Road']
        colors = ['#8c564b', '#e377c2']
        
        bars2 = ax2.bar(iou_labels, iou_values, color=colors)
        ax2.set_ylabel('IoU (%)')
        ax2.set_title('IoU por Clase')
        ax2.set_ylim(0, 100)
        
        for bar, value in zip(bars2, iou_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Distribución de clases
        sizes = [metrics['background_pixels'], metrics['road_pixels']]
        labels_pie = ['Background', 'Road']
        colors_pie = ['#ff9999', '#66b3ff']
        
        ax3.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%')
        ax3.set_title('Distribución de Clases en Dataset')
        
        # Estadísticas de inferencia
        stats = self.results['inference_stats']
        stat_labels = ['Tiempo/muestra\n(ms)', 'FPS', 'Total muestras']
        stat_values = [stats['avg_time_per_sample']*1000, stats['fps'], stats['num_samples']]
        
        ax4.bar(stat_labels, stat_values, color=['#bcbd22', '#17becf', '#7f7f7f'])
        ax4.set_title('Estadísticas de Inferencia')
        
        for i, (label, value) in enumerate(zip(stat_labels, stat_values)):
            if 'ms' in label:
                text = f'{value:.1f}'
            elif 'FPS' in label:
                text = f'{value:.1f}'
            else:
                text = f'{int(value)}'
            ax4.text(i, value + max(stat_values)*0.02, text, ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self):
        cm = self.results['metrics']['confusion_matrix']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Matriz absoluta
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Background', 'Road'],
                   yticklabels=['Background', 'Road'], ax=ax1)
        ax1.set_title('Matriz de Confusión (Valores Absolutos)')
        ax1.set_ylabel('Verdadero')
        ax1.set_xlabel('Predicho')
        
        # Matriz normalizada
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=['Background', 'Road'],
                   yticklabels=['Background', 'Road'], ax=ax2)
        ax2.set_title('Matriz de Confusión (Normalizada)')
        ax2.set_ylabel('Verdadero')
        ax2.set_xlabel('Predicho')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_pr_curves(self):
        curves = self.results['metrics']['curves']
        roc_auc = self.results['metrics']['roc_auc']
        pr_auc = self.results['metrics']['pr_auc']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Curva ROC
        ax1.plot(curves['roc']['fpr'], curves['roc']['tpr'], 
                 label=f'ROC (AUC = {roc_auc:.3f})', linewidth=2)
        ax1.plot([0, 1], [0, 1], 'k--', label='Random')
        ax1.set_xlabel('Tasa de Falsos Positivos')
        ax1.set_ylabel('Tasa de Verdaderos Positivos')
        ax1.set_title('Curva ROC - Clase Road')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Curva Precision-Recall
        ax2.plot(curves['pr']['recall'], curves['pr']['precision'],
                 label=f'PR (AUC = {pr_auc:.3f})', linewidth=2)
        ax2.axhline(y=self.results['metrics']['class_balance']/100, 
                   color='k', linestyle='--', label='Random')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Curva Precision-Recall - Clase Road')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_pr_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_probability_distributions(self):
        probs = self.results['probabilities']
        targets = self.results['targets'].numpy().flatten()
        
        # Probabilidades para clase road
        road_probs = probs[:, 1].numpy().flatten()
        
        # Separar por clase verdadera
        bg_probs = road_probs[targets == 0]
        road_probs_true = road_probs[targets == 1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogramas superpuestos
        ax1.hist(bg_probs, bins=50, alpha=0.7, label='Background (True)', 
                 color='orange', density=True)
        ax1.hist(road_probs_true, bins=50, alpha=0.7, label='Road (True)', 
                 color='blue', density=True)
        ax1.set_xlabel('Probabilidad predicha para clase Road')
        ax1.set_ylabel('Densidad')
        ax1.set_title('Distribución de Probabilidades')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plots
        data_to_plot = [bg_probs, road_probs_true]
        labels = ['Background\n(True)', 'Road\n(True)']
        colors = ['orange', 'blue']
        
        box_plot = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Probabilidad predicha para clase Road')
        ax2.set_title('Box Plot de Probabilidades')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'probability_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_prediction_samples(self, num_samples=20):
        samples = self.results['sample_results'][:num_samples]
        
        # Crear grid de imágenes
        rows = min(5, len(samples))
        cols = 4  # RGB, Ground Truth, Prediction, Overlay
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, sample in enumerate(samples[:rows]):
            if i >= rows:
                break
                
            # RGB (transponer de CHW a HWC y recortar valores)
            rgb = sample['rgb'].transpose(1, 2, 0)
            rgb = np.clip(rgb, 0, 1)
            
            # Ground truth
            gt = sample['target']
            
            # Predicción
            pred = sample['prediction']
            
            # Overlay (RGB + predicción)
            overlay = rgb.copy()
            road_mask = pred == 1
            overlay[road_mask] = [1, 0, 0]  # Rojo para carreteras predichas
            
            # Mostrar imágenes
            axes[i, 0].imshow(rgb)
            axes[i, 0].set_title(f'Sample {i+1}: RGB')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(gt, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title('Overlay')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'prediction_samples.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_error_analysis(self):
        predictions = self.results['predictions'].numpy()
        targets = self.results['targets'].numpy()
        
        # Calcular tipos de errores por muestra
        error_stats = []
        
        for i, sample in enumerate(self.results['sample_results'][:50]):  # Primeras 50 muestras
            pred = sample['prediction']
            gt = sample['target']
            
            tp = np.sum((pred == 1) & (gt == 1))
            fp = np.sum((pred == 1) & (gt == 0))
            fn = np.sum((pred == 0) & (gt == 1))
            tn = np.sum((pred == 0) & (gt == 0))
            
            total_pixels = pred.size
            
            error_stats.append({
                'sample': i,
                'tp_rate': tp / total_pixels * 100,
                'fp_rate': fp / total_pixels * 100,
                'fn_rate': fn / total_pixels * 100,
                'accuracy': (tp + tn) / total_pixels * 100
            })
        
        # Crear gráficos
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distribución de accuracy por muestra
        accuracies = [s['accuracy'] for s in error_stats]
        ax1.hist(accuracies, bins=20, alpha=0.7, color='green')
        ax1.set_xlabel('Accuracy por muestra (%)')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title('Distribución de Accuracy por Muestra')
        ax1.axvline(np.mean(accuracies), color='red', linestyle='--', 
                   label=f'Media: {np.mean(accuracies):.1f}%')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Tasas de error
        samples_idx = [s['sample'] for s in error_stats]
        fp_rates = [s['fp_rate'] for s in error_stats]
        fn_rates = [s['fn_rate'] for s in error_stats]
        
        ax2.plot(samples_idx, fp_rates, 'ro-', alpha=0.7, label='False Positives')
        ax2.plot(samples_idx, fn_rates, 'bo-', alpha=0.7, label='False Negatives')
        ax2.set_xlabel('Muestra')
        ax2.set_ylabel('Tasa de error (%)')
        ax2.set_title('Evolución de Errores por Muestra')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Correlación FP vs FN
        ax3.scatter(fp_rates, fn_rates, alpha=0.7)
        ax3.set_xlabel('Tasa de False Positives (%)')
        ax3.set_ylabel('Tasa de False Negatives (%)')
        ax3.set_title('Correlación FP vs FN')
        ax3.grid(True, alpha=0.3)
        
        # Resumen de errores
        total_tp = sum(s['tp_rate'] for s in error_stats)
        total_fp = sum(s['fp_rate'] for s in error_stats)
        total_fn = sum(s['fn_rate'] for s in error_stats)
        
        error_types = ['True Positives', 'False Positives', 'False Negatives']
        error_values = [total_tp, total_fp, total_fn]
        colors = ['green', 'red', 'orange']
        
        ax4.bar(error_types, error_values, color=colors, alpha=0.7)
        ax4.set_ylabel('Tasa acumulada (%)')
        ax4.set_title('Resumen de Tipos de Predicción')
        ax4.tick_params(axis='x', rotation=45)
        
        for i, v in enumerate(error_values):
            ax4.text(i, v + max(error_values)*0.01, f'{v:.1f}%', 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_detailed_report(self):
        print("Guardando reporte detallado...")
        
        # Preparar datos para JSON (convertir numpy arrays)
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        report = {
            'evaluation_summary': {
                'dataset_size': len(self.eval_dataset),
                'model_config': self.config,
                'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'metrics': convert_numpy({
                k: v for k, v in self.results['metrics'].items()
                if k != 'curves'  # Excluir curvas (arrays grandes)
            }),
            'inference_stats': convert_numpy(self.results['inference_stats']),
            'classification_report': convert_numpy(self.results['classification_report'])
        }
        
        # Guardar como JSON
        with open(self.output_dir / 'evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Guardar métricas en CSV para análisis
        import pandas as pd
        
        metrics_df = pd.DataFrame([report['metrics']])
        metrics_df.to_csv(self.output_dir / 'metrics.csv', index=False)
        
        print(f"Reporte guardado en: {self.output_dir / 'evaluation_report.json'}")
    
    def run_complete_evaluation(self, max_samples=None, num_vis_samples=20):
        print("Iniciando evaluación completa del modelo...")
        
        # 1. Evaluar dataset
        self.evaluate_dataset(max_samples)
        
        # 2. Calcular métricas
        self.calculate_metrics()
        
        # 3. Generar visualizaciones
        self.generate_visualizations(num_vis_samples)
        
        # 4. Guardar reporte
        self.save_detailed_report()
        
        print("\nEvaluación completada.")
        print(f"Resultados en: {self.output_dir}")
        
        # Mostrar resumen final
        metrics = self.results['metrics']
        print(f"\nRESUMEN DE RESULTADOS:")
        print(f"   Accuracy: {metrics['accuracy']:.2f}%")
        print(f"   Road IoU: {metrics['iou_road']:.2f}%")
        print(f"   mIoU: {metrics['miou']:.2f}%")
        print(f"   F1-Score: {metrics['f1_score']:.2f}%")
        print(f"   ROC AUC: {metrics['roc_auc']:.3f}")
        print(f"   Velocidad: {self.results['inference_stats']['fps']:.1f} FPS")

def main():
    parser = argparse.ArgumentParser(description='Evaluación MSFANet Sentinel-2')
    parser.add_argument('--config', type=str, default='config_sentinel2.yaml',
                       help='Archivo de configuración')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_sentinel2/best_s2.pth',
                       help='Checkpoint del modelo entrenado')
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='Directorio de salida')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Máximo número de muestras a evaluar')
    parser.add_argument('--vis-samples', type=int, default=20,
                       help='Número de muestras para visualizar')
    
    args = parser.parse_args()
    
    try:
        # Crear evaluador
        evaluator = Sentinel2Evaluator(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            output_dir=args.output
        )
        
        # Ejecutar evaluación completa
        evaluator.run_complete_evaluation(
            max_samples=args.max_samples,
            num_vis_samples=args.vis_samples
        )
        
    except Exception as e:
        print(f"Error durante evaluación: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()