#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Union


class PerformanceMetrics:
    """
    Class to calculate comprehensive performance metrics for segmentation tasks.
    """
    
    def __init__(self, num_classes: int = 2, target_class: int = 1, ignore_index: int = -1):
        """
        Initialize the metrics calculator.
        
        Args:
            num_classes: Number of classes in the segmentation task
            target_class: Index of the target class (e.g., road class)
            ignore_index: Index to ignore in calculations (e.g., unlabeled pixels)
        """
        self.num_classes = num_classes
        self.target_class = target_class
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.total_samples = 0
        self.total_pixels = 0
        self.class_samples = np.zeros(self.num_classes)
    
    def update(self, pred: Union[torch.Tensor, np.ndarray], target: Union[torch.Tensor, np.ndarray]):
        """
        Update metrics with new predictions and targets.
        
        Args:
            pred: Predicted labels or logits
            target: Ground truth labels
        """
        # Convert to numpy if tensors
        if isinstance(pred, torch.Tensor):
            if pred.dim() > 2:  # If logits, convert to predictions
                pred = torch.argmax(pred, dim=1)
            pred = pred.cpu().numpy()
        
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        
        # Flatten arrays
        pred = pred.flatten()
        target = target.flatten()
        
        # Create mask for valid pixels (ignore ignore_index)
        mask = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        mask = mask & (pred >= 0) & (pred < self.num_classes)
        
        if not np.any(mask):
            return
        
        pred_valid = pred[mask]
        target_valid = target[mask]
        
        # Update confusion matrix
        hist = np.bincount(
            self.num_classes * target_valid.astype(int) + pred_valid.astype(int),
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += hist
        self.total_samples += len(pred_valid)
        self.total_pixels += len(pred_valid)
        
        # Update class samples
        for i in range(self.num_classes):
            self.class_samples[i] += np.sum(target_valid == i)
    
    def compute_iou(self, class_idx: int) -> float:
        """
        Compute IoU for a specific class.
        
        Args:
            class_idx: Index of the class
            
        Returns:
            IoU value for the class
        """
        if class_idx >= self.num_classes:
            return 0.0
        
        intersection = self.confusion_matrix[class_idx, class_idx]
        union = (self.confusion_matrix[class_idx, :].sum() + 
                self.confusion_matrix[:, class_idx].sum() - intersection)
        
        return intersection / (union + 1e-8)
    
    def compute_precision(self, class_idx: int) -> float:
        """
        Compute precision for a specific class.
        
        Args:
            class_idx: Index of the class
            
        Returns:
            Precision value for the class
        """
        if class_idx >= self.num_classes:
            return 0.0
        
        tp = self.confusion_matrix[class_idx, class_idx]
        fp = self.confusion_matrix[:, class_idx].sum() - tp
        
        return tp / (tp + fp + 1e-8)
    
    def compute_recall(self, class_idx: int) -> float:
        """
        Compute recall for a specific class.
        
        Args:
            class_idx: Index of the class
            
        Returns:
            Recall value for the class
        """
        if class_idx >= self.num_classes:
            return 0.0
        
        tp = self.confusion_matrix[class_idx, class_idx]
        fn = self.confusion_matrix[class_idx, :].sum() - tp
        
        return tp / (tp + fn + 1e-8)
    
    def compute_f1(self, class_idx: int) -> float:
        """
        Compute F1-score for a specific class.
        
        Args:
            class_idx: Index of the class
            
        Returns:
            F1-score value for the class
        """
        precision = self.compute_precision(class_idx)
        recall = self.compute_recall(class_idx)
        
        return 2 * precision * recall / (precision + recall + 1e-8)
    
    def compute_accuracy(self) -> float:
        """
        Compute overall pixel accuracy.
        
        Returns:
            Overall accuracy
        """
        if self.total_samples == 0:
            return 0.0
        
        correct = np.trace(self.confusion_matrix)
        return correct / self.total_samples
    
    def compute_mean_iou(self) -> float:
        """
        Compute mean IoU across all classes.
        
        Returns:
            Mean IoU value
        """
        ious = []
        for i in range(self.num_classes):
            iou = self.compute_iou(i)
            ious.append(iou)
        
        return np.mean(ious)
    
    def compute_weighted_iou(self) -> float:
        """
        Compute weighted IoU based on class frequencies.
        
        Returns:
            Weighted IoU value
        """
        if self.total_samples == 0:
            return 0.0
        
        weighted_iou = 0.0
        for i in range(self.num_classes):
            iou = self.compute_iou(i)
            weight = self.class_samples[i] / self.total_samples
            weighted_iou += iou * weight
        
        return weighted_iou
    
    def compute_dice_coefficient(self, class_idx: int) -> float:
        """
        Compute Dice coefficient for a specific class.
        
        Args:
            class_idx: Index of the class
            
        Returns:
            Dice coefficient value for the class
        """
        if class_idx >= self.num_classes:
            return 0.0
        
        tp = self.confusion_matrix[class_idx, class_idx]
        fp = self.confusion_matrix[:, class_idx].sum() - tp
        fn = self.confusion_matrix[class_idx, :].sum() - tp
        
        return 2 * tp / (2 * tp + fp + fn + 1e-8)
    
    def compute_specificity(self, class_idx: int) -> float:
        """
        Compute specificity for a specific class.
        
        Args:
            class_idx: Index of the class
            
        Returns:
            Specificity value for the class
        """
        if class_idx >= self.num_classes:
            return 0.0
        
        tp = self.confusion_matrix[class_idx, class_idx]
        fp = self.confusion_matrix[:, class_idx].sum() - tp
        tn = self.confusion_matrix.sum() - self.confusion_matrix[class_idx, :].sum() - fp
        
        return tn / (tn + fp + 1e-8)
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics and return as dictionary.
        
        Returns:
            Dictionary containing all computed metrics
        """
        if self.total_samples == 0:
            return self._empty_metrics()
        
        # Compute IoU for all classes
        all_ious = []
        for i in range(self.num_classes):
            iou = self.compute_iou(i)
            all_ious.append(iou)
        
        # Target class specific metrics
        target_iou = self.compute_iou(self.target_class)
        target_precision = self.compute_precision(self.target_class)
        target_recall = self.compute_recall(self.target_class)
        target_f1 = self.compute_f1(self.target_class)
        target_dice = self.compute_dice_coefficient(self.target_class)
        target_specificity = self.compute_specificity(self.target_class)
        
        metrics = {
            # Overall metrics
            'accuracy': self.compute_accuracy() * 100,
            'miou': self.compute_mean_iou() * 100,
            'weighted_iou': self.compute_weighted_iou() * 100,
            
            # Target class metrics (e.g., road class)
            'target_iou': target_iou * 100,
            'target_precision': target_precision * 100,
            'target_recall': target_recall * 100,
            'target_f1': target_f1 * 100,
            'target_dice': target_dice * 100,
            'target_specificity': target_specificity * 100,
            
            # Per-class IoU
            'all_ious': np.array(all_ious) * 100,
            
            # Additional info
            'total_samples': self.total_samples,
            'confusion_matrix': self.confusion_matrix.tolist(),
            'class_distribution': (self.class_samples / self.total_samples * 100).tolist()
        }
        
        return metrics
    
    def _empty_metrics(self) -> Dict[str, float]:
        """
        Return default metrics when no data is available.
        
        Returns:
            Dictionary with default metric values
        """
        return {
            'accuracy': 0.0,
            'miou': 0.0,
            'weighted_iou': 0.0,
            'target_iou': 0.0,
            'target_precision': 0.0,
            'target_recall': 0.0,
            'target_f1': 0.0,
            'target_dice': 0.0,
            'target_specificity': 0.0,
            'road_iou': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'all_ious': np.zeros(self.num_classes),
            'total_samples': 0,
            'confusion_matrix': np.zeros((self.num_classes, self.num_classes)).tolist(),
            'class_distribution': np.zeros(self.num_classes).tolist()
        }
    
    def print_metrics(self, detailed: bool = True):
        """
        Print computed metrics in a formatted way.
        
        Args:
            detailed: Whether to print detailed metrics
        """
        metrics = self.compute_metrics()
        
        print("=" * 60)
        print("SEGMENTATION METRICS")
        print("=" * 60)
        
        print(f"Overall Accuracy: {metrics['accuracy']:.2f}%")
        print(f"Mean IoU (mIoU): {metrics['miou']:.2f}%")
        print(f"Weighted IoU: {metrics['weighted_iou']:.2f}%")
        
        print(f"\nTarget Class Metrics (Class {self.target_class}):")
        print(f"  IoU: {metrics['target_iou']:.2f}%")
        print(f"  Precision: {metrics['target_precision']:.2f}%")
        print(f"  Recall: {metrics['target_recall']:.2f}%")
        print(f"  F1-Score: {metrics['target_f1']:.2f}%")
        print(f"  Dice Coefficient: {metrics['target_dice']:.2f}%")
        print(f"  Specificity: {metrics['target_specificity']:.2f}%")
        
        if detailed:
            print(f"\nPer-Class IoU:")
            for i, iou in enumerate(metrics['all_ious']):
                print(f"  Class {i}: {iou:.2f}%")
            
            print(f"\nClass Distribution:")
            for i, dist in enumerate(metrics['class_distribution']):
                print(f"  Class {i}: {dist:.2f}%")
            
            print(f"\nTotal Samples: {metrics['total_samples']:,}")
        
        print("=" * 60)


class SegmentationLoss:
    """
    Collection of loss functions for segmentation tasks.
    """
    
    @staticmethod
    def weighted_cross_entropy(weight: Optional[torch.Tensor] = None, 
                             ignore_index: int = -1) -> nn.CrossEntropyLoss:
        """
        Create weighted cross-entropy loss.
        
        Args:
            weight: Class weights tensor
            ignore_index: Index to ignore in loss calculation
            
        Returns:
            CrossEntropyLoss instance
        """
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    
    @staticmethod
    def focal_loss(alpha: float = 1.0, gamma: float = 2.0, 
                  ignore_index: int = -1) -> 'FocalLoss':
        """
        Create focal loss for handling class imbalance.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            ignore_index: Index to ignore in loss calculation
            
        Returns:
            FocalLoss instance
        """
        return FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
    
    @staticmethod
    def dice_loss(smooth: float = 1.0, ignore_index: int = -1) -> 'DiceLoss':
        """
        Create Dice loss for better boundary detection.
        
        Args:
            smooth: Smoothing factor
            ignore_index: Index to ignore in loss calculation
            
        Returns:
            DiceLoss instance
        """
        return DiceLoss(smooth=smooth, ignore_index=ignore_index)
    
    @staticmethod
    def combined_loss(ce_weight: float = 1.0, dice_weight: float = 1.0,
                     class_weights: Optional[torch.Tensor] = None,
                     ignore_index: int = -1) -> 'CombinedLoss':
        """
        Create combined Cross-Entropy + Dice loss.
        
        Args:
            ce_weight: Weight for cross-entropy loss
            dice_weight: Weight for dice loss
            class_weights: Class weights for cross-entropy
            ignore_index: Index to ignore in loss calculation
            
        Returns:
            CombinedLoss instance
        """
        return CombinedLoss(ce_weight=ce_weight, dice_weight=dice_weight,
                          class_weights=class_weights, ignore_index=ignore_index)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, ignore_index: int = -1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """
    Dice Loss for better boundary detection.
    """
    
    def __init__(self, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply softmax to get probabilities
        inputs = torch.softmax(inputs, dim=1)
        
        # Create mask for valid pixels
        mask = targets != self.ignore_index
        
        # One-hot encode targets
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Apply mask
        inputs = inputs * mask.unsqueeze(1).float()
        targets_one_hot = targets_one_hot * mask.unsqueeze(1).float()
        
        # Calculate Dice coefficient
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (
            inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3)) + self.smooth
        )
        
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combined Cross-Entropy and Dice Loss.
    """
    
    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0,
                 class_weights: Optional[torch.Tensor] = None, ignore_index: int = -1):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce_loss(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss