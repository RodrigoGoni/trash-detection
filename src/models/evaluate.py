"""
Model evaluation and metrics
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """
    Model evaluation utilities

    Args:
        model: PyTorch model
        device: Device to evaluate on
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions and ground truth labels

        Returns:
            predictions, labels
        """
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)

                outputs = self.model(images)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        return np.array(all_preds), np.array(all_labels)

    def predict_proba(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get prediction probabilities and ground truth labels

        Returns:
            probabilities, labels
        """
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)

                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())

        return np.array(all_probs), np.array(all_labels)

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Comprehensive model evaluation

        Returns:
            Dictionary of metrics
        """
        preds, labels = self.predict(dataloader)
        probs, _ = self.predict_proba(dataloader)

        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision_macro': precision_score(labels, preds, average='macro'),
            'precision_weighted': precision_score(labels, preds, average='weighted'),
            'recall_macro': recall_score(labels, preds, average='macro'),
            'recall_weighted': recall_score(labels, preds, average='weighted'),
            'f1_macro': f1_score(labels, preds, average='macro'),
            'f1_weighted': f1_score(labels, preds, average='weighted'),
        }

        # ROC AUC (for multi-class)
        try:
            metrics['roc_auc_ovr'] = roc_auc_score(
                labels, probs, multi_class='ovr', average='weighted'
            )
        except:
            pass

        return metrics

    def confusion_matrix(
        self,
        dataloader: DataLoader,
        class_names: List[str] = None,
        save_path: str = None
    ) -> np.ndarray:
        """
        Compute and plot confusion matrix

        Args:
            dataloader: DataLoader
            class_names: List of class names
            save_path: Path to save plot

        Returns:
            Confusion matrix
        """
        preds, labels = self.predict(dataloader)
        cm = confusion_matrix(labels, preds)

        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save_path:
            plt.savefig(save_path)

        plt.close()

        return cm

    def classification_report_dict(
        self,
        dataloader: DataLoader,
        class_names: List[str] = None
    ) -> Dict:
        """
        Get detailed classification report

        Args:
            dataloader: DataLoader
            class_names: List of class names

        Returns:
            Classification report as dictionary
        """
        preds, labels = self.predict(dataloader)

        report = classification_report(
            labels, preds,
            target_names=class_names,
            output_dict=True
        )

        return report

    def per_class_accuracy(
        self,
        dataloader: DataLoader,
        class_names: List[str] = None
    ) -> Dict[str, float]:
        """
        Calculate per-class accuracy

        Returns:
            Dictionary of class-wise accuracies
        """
        preds, labels = self.predict(dataloader)
        cm = confusion_matrix(labels, preds)

        per_class_acc = cm.diagonal() / cm.sum(axis=1)

        if class_names:
            return {name: acc for name, acc in zip(class_names, per_class_acc)}
        else:
            return {f"class_{i}": acc for i, acc in enumerate(per_class_acc)}


def calculate_mAP(predictions: List[Dict], ground_truths: List[Dict], iou_threshold: float = 0.5) -> float:
    """
    Calculate mean Average Precision for object detection

    Args:
        predictions: List of prediction dictionaries
        ground_truths: List of ground truth dictionaries
        iou_threshold: IoU threshold for matching

    Returns:
        mAP score
    """
    # This is a simplified version
    # For production, use libraries like pycocotools

    aps = []

    for pred, gt in zip(predictions, ground_truths):
        # Extract boxes and scores
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        pred_labels = pred['labels']

        gt_boxes = gt['boxes']
        gt_labels = gt['labels']

        # Calculate IoU matrix
        # ... (implementation details)

        # Calculate AP for this image
        # ... (implementation details)

        pass

    # Return mean AP
    return np.mean(aps) if aps else 0.0
