#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
import yaml

try:
    from msfanet_model import get_msfanet
    MODEL_AVAILABLE = True
except ImportError:
    from simple_msfanet import get_simple_msfanet
    MODEL_AVAILABLE = False

from final_spacenet_processor import WorkingSpaceNetDataset

def load_trained_model(checkpoint_path, config):
    if MODEL_AVAILABLE:
        model = get_msfanet(
            num_classes=config['num_classes'],
            width=config.get('model_width', 32)
        )
    else:
        model = get_simple_msfanet(
            num_classes=config['num_classes'],
            width=config.get('model_width', 32)
        )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Modelo cargado desde {checkpoint_path}")
    print(f"   Época: {checkpoint['epoch']}")
    print(f"   Mejor mIoU: {checkpoint['best_miou']:.2f}%")
    
    return model

def visualize_predictions(model, dataset, device, num_samples=8):
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples * 3))
    fig.suptitle('MSFANet Results: RGB | Ground Truth | Prediction | Overlay', fontsize=16)
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= num_samples:
                break
                
            rgb = data['rgb'].to(device)
            msi = data['ms'].to(device) 
            target = data['mask'].cpu().numpy()[0]
            
            # Predicción
            output = model(rgb, msi)
            pred = torch.argmax(output, dim=1).cpu().numpy()[0]
            
            # Preparar RGB para visualización
            rgb_vis = rgb[0].cpu().permute(1, 2, 0).numpy()
            rgb_vis = (rgb_vis - rgb_vis.min()) / (rgb_vis.max() - rgb_vis.min())
            
            # Plot
            row = i
            
            # RGB Image
            axes[row, 0].imshow(rgb_vis)
            axes[row, 0].set_title('RGB Image')
            axes[row, 0].axis('off')
            
            # Ground Truth
            axes[row, 1].imshow(target, cmap='gray', vmin=0, vmax=1)
            axes[row, 1].set_title('Ground Truth')
            axes[row, 1].axis('off')
            
            # Prediction
            axes[row, 2].imshow(pred, cmap='gray', vmin=0, vmax=1)
            axes[row, 2].set_title('Prediction')
            axes[row, 2].axis('off')
            
            # Overlay
            overlay = rgb_vis.copy()
            # Carreteras verdaderas en verde
            overlay[target == 1] = [0, 1, 0]  
            # Predicciones en rojo
            overlay[pred == 1] = overlay[pred == 1] * 0.7 + np.array([1, 0, 0]) * 0.3
            
            axes[row, 3].imshow(overlay)
            axes[row, 3].set_title('Overlay (GT=Green, Pred=Red)')
            axes[row, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('msfanet_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualización guardada como 'msfanet_results.png'")

def calculate_detailed_metrics(model, dataset, device):
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    total_samples = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    print("Calculando métricas detalladas...")
    
    with torch.no_grad():
        for data in loader:
            rgb = data['rgb'].to(device)
            msi = data['ms'].to(device)
            target = data['mask'].to(device)
            
            # Predicción
            output = model(rgb, msi)
            pred = torch.argmax(output, dim=1)
            
            # Métricas por batch
            tp = ((pred == 1) & (target == 1)).sum().item()
            fp = ((pred == 1) & (target == 0)).sum().item()
            fn = ((pred == 0) & (target == 1)).sum().item()
            tn = ((pred == 0) & (target == 0)).sum().item()
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
            total_samples += target.numel()
    
    # Calcular métricas finales
    precision = total_tp / (total_tp + total_fp + 1e-7)
    recall = total_tp / (total_tp + total_fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-7)
    accuracy = (total_tp + total_tn) / total_samples
    
    print("\nMÉTRICAS DETALLADAS:")
    print(f"   Total píxeles: {total_samples:,}")
    print(f"   Píxeles de carretera: {total_tp + total_fn:,} ({(total_tp + total_fn)/total_samples*100:.2f}%)")
    print(f"   Precisión: {precision*100:.2f}%")
    print(f"   Recall: {recall*100:.2f}%")
    print(f"   F1-Score: {f1*100:.2f}%")
    print(f"   IoU: {iou*100:.2f}%")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    
    return {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'iou': iou * 100,
        'accuracy': accuracy * 100
    }

def main():
    print("ANÁLISIS DE RESULTADOS MSFANet")
    print("="*50)
    
    # Cargar configuración
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Cargar modelo entrenado
    checkpoint_path = Path('checkpoints/best.pth')
    if not checkpoint_path.exists():
        checkpoint_path = Path('checkpoints/last.pth')
    
    if not checkpoint_path.exists():
        print("No se encontró checkpoint del modelo")
        return
    
    model = load_trained_model(checkpoint_path, config)
    model = model.to(device)
    
    # Cargar dataset
    dataset = WorkingSpaceNetDataset(config['data_dir'])
    print(f"Dataset: {len(dataset)} muestras")
    
    # Calcular métricas detalladas
    metrics = calculate_detailed_metrics(model, dataset, device)
    
    # Visualizar resultados
    print("\nGenerando visualizaciones...")
    visualize_predictions(model, dataset, device, num_samples=6)
    
    print("\n" + "="*50)
    print("RESUMEN FINAL:")
    print(f"   F1-Score: {metrics['f1']:.2f}%")
    print(f"   IoU: {metrics['iou']:.2f}%")
    print(f"   Precisión: {metrics['precision']:.2f}%")
    print(f"   Recall: {metrics['recall']:.2f}%")

if __name__ == "__main__":
    main()
