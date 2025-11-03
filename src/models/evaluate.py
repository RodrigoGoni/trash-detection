"""
Model evaluation and metrics for object detection
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


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


def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes
    
    Args:
        box1: Box in format [x1, y1, x2, y2]
        box2: Box in format [x1, y1, x2, y2]
    
    Returns:
        IoU score
    """
    # Calculate intersection
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    # Calculate union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def calculate_ap_per_class(
    pred_boxes: List[torch.Tensor],
    pred_scores: List[torch.Tensor],
    pred_labels: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    gt_labels: List[torch.Tensor],
    class_id: int,
    iou_threshold: float = 0.5
) -> float:
    """
    Calculate Average Precision for a single class
    
    Args:
        pred_boxes: List of predicted boxes per image
        pred_scores: List of prediction scores per image
        pred_labels: List of predicted labels per image
        gt_boxes: List of ground truth boxes per image
        gt_labels: List of ground truth labels per image
        class_id: Class ID to calculate AP for
        iou_threshold: IoU threshold for matching
    
    Returns:
        Average Precision for the class
    """
    # Collect all predictions and ground truths for this class
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_image_ids = []
    
    all_gt_boxes = []
    all_gt_image_ids = []
    
    for img_idx, (pb, ps, pl, gb, gl) in enumerate(
        zip(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
    ):
        # Filter predictions for this class
        class_mask_pred = pl == class_id
        if class_mask_pred.any():
            all_pred_boxes.extend(pb[class_mask_pred].cpu().numpy())
            all_pred_scores.extend(ps[class_mask_pred].cpu().numpy())
            all_pred_image_ids.extend([img_idx] * class_mask_pred.sum().item())
        
        # Filter ground truths for this class
        class_mask_gt = gl == class_id
        if class_mask_gt.any():
            all_gt_boxes.extend(gb[class_mask_gt].cpu().numpy())
            all_gt_image_ids.extend([img_idx] * class_mask_gt.sum().item())
    
    if len(all_gt_boxes) == 0:
        return 0.0
    
    if len(all_pred_boxes) == 0:
        return 0.0
    
    # Sort predictions by score (descending)
    sorted_indices = np.argsort(all_pred_scores)[::-1]
    all_pred_boxes = [all_pred_boxes[i] for i in sorted_indices]
    all_pred_scores = [all_pred_scores[i] for i in sorted_indices]
    all_pred_image_ids = [all_pred_image_ids[i] for i in sorted_indices]
    
    # Track which ground truths have been matched
    gt_matched = [False] * len(all_gt_boxes)
    
    # Calculate true positives and false positives
    tp = np.zeros(len(all_pred_boxes))
    fp = np.zeros(len(all_pred_boxes))
    
    for pred_idx, (pred_box, pred_img_id) in enumerate(
        zip(all_pred_boxes, all_pred_image_ids)
    ):
        # Find ground truths in the same image
        max_iou = 0.0
        max_gt_idx = -1
        
        for gt_idx, (gt_box, gt_img_id) in enumerate(
            zip(all_gt_boxes, all_gt_image_ids)
        ):
            if gt_img_id != pred_img_id:
                continue
            
            if gt_matched[gt_idx]:
                continue
            
            iou = calculate_iou(
                torch.tensor(pred_box),
                torch.tensor(gt_box)
            )
            
            if iou > max_iou:
                max_iou = iou
                max_gt_idx = gt_idx
        
        # Check if prediction matches a ground truth
        if max_iou >= iou_threshold and max_gt_idx >= 0:
            if not gt_matched[max_gt_idx]:
                tp[pred_idx] = 1
                gt_matched[max_gt_idx] = True
            else:
                fp[pred_idx] = 1
        else:
            fp[pred_idx] = 1
    
    # Calculate precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / len(all_gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # Calculate AP using 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    
    return ap


def calculate_mAP(
    pred_boxes: List[torch.Tensor],
    pred_scores: List[torch.Tensor],
    pred_labels: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    gt_labels: List[torch.Tensor],
    num_classes: int,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate mean Average Precision and per-class AP for object detection
    
    Args:
        pred_boxes: List of predicted boxes per image [N, 4]
        pred_scores: List of prediction scores per image [N]
        pred_labels: List of predicted labels per image [N]
        gt_boxes: List of ground truth boxes per image [M, 4]
        gt_labels: List of ground truth labels per image [M]
        num_classes: Total number of classes (excluding background)
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with mAP and per-class AP scores
    """
    aps = []
    per_class_ap = {}
    
    # Calculate AP for each class (skip background class 0)
    for class_id in range(1, num_classes):
        ap = calculate_ap_per_class(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels,
            class_id, iou_threshold
        )
        aps.append(ap)
        per_class_ap[f'AP_class_{class_id}'] = ap
    
    # Calculate mAP
    mAP = np.mean(aps) if aps else 0.0
    
    results = {
        'mAP': mAP,
        **per_class_ap
    }
    
    return results


def calculate_detection_metrics(
    pred_boxes: List[torch.Tensor],
    pred_scores: List[torch.Tensor],
    pred_labels: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    gt_labels: List[torch.Tensor],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 for object detection
    
    Args:
        pred_boxes: List of predicted boxes per image
        pred_scores: List of prediction scores per image
        pred_labels: List of predicted labels per image
        gt_boxes: List of ground truth boxes per image
        gt_labels: List of ground truth labels per image
        iou_threshold: IoU threshold for matching
        score_threshold: Score threshold for filtering predictions
    
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for pb, ps, pl, gb, gl in zip(
        pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels
    ):
        # Filter predictions by score threshold
        score_mask = ps >= score_threshold
        pb = pb[score_mask]
        ps = ps[score_mask]
        pl = pl[score_mask]
        
        # Track matched ground truths
        gt_matched = torch.zeros(len(gb), dtype=torch.bool)
        
        # For each prediction, find best matching ground truth
        for pred_box, pred_label in zip(pb, pl):
            max_iou = 0.0
            max_gt_idx = -1
            
            for gt_idx, (gt_box, gt_label) in enumerate(zip(gb, gl)):
                if pred_label != gt_label:
                    continue
                
                if gt_matched[gt_idx]:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            if max_iou >= iou_threshold and max_gt_idx >= 0:
                total_tp += 1
                gt_matched[max_gt_idx] = True
            else:
                total_fp += 1
        
        # Count false negatives (unmatched ground truths)
        total_fn += (~gt_matched).sum().item()
    
    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn
    }


class ObjectDetectionEvaluator:
    """
    Evaluator for object detection models
    
    Args:
        model: PyTorch detection model
        device: Device to evaluate on
        num_classes: Number of classes (including background)
        score_threshold: Score threshold for filtering predictions
        iou_threshold: IoU threshold for NMS and matching
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        num_classes: int = 61,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.5
    ):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.model.to(device)
        self.model.eval()
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Comprehensive evaluation with detection metrics
        
        Args:
            dataloader: DataLoader for evaluation
        
        Returns:
            Dictionary with all metrics
        """
        all_pred_boxes = []
        all_pred_scores = []
        all_pred_labels = []
        all_gt_boxes = []
        all_gt_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                images = batch['images'].to(self.device)
                
                # Get predictions
                predictions = self.model(images)
                
                # Collect predictions
                for pred in predictions:
                    all_pred_boxes.append(pred['boxes'].cpu())
                    all_pred_scores.append(pred['scores'].cpu())
                    all_pred_labels.append(pred['labels'].cpu())
                
                # Collect ground truths
                for i in range(len(batch['bboxes'])):
                    all_gt_boxes.append(batch['bboxes'][i].cpu())
                    all_gt_labels.append(batch['labels'][i].cpu())
        
        # Calculate mAP
        map_results = calculate_mAP(
            all_pred_boxes, all_pred_scores, all_pred_labels,
            all_gt_boxes, all_gt_labels,
            self.num_classes, self.iou_threshold
        )
        
        # Calculate precision, recall, F1
        detection_metrics = calculate_detection_metrics(
            all_pred_boxes, all_pred_scores, all_pred_labels,
            all_gt_boxes, all_gt_labels,
            self.iou_threshold, self.score_threshold
        )
        
        # Combine all metrics
        all_metrics = {
            **map_results,
            **detection_metrics
        }
        
        return all_metrics
    
    def predict(self, dataloader: DataLoader) -> Tuple[List, List]:
        """
        Get predictions and ground truths
        
        Returns:
            predictions, ground_truths
        """
        all_predictions = []
        all_ground_truths = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                images = batch['images'].to(self.device)
                predictions = self.model(images)
                
                for i, pred in enumerate(predictions):
                    all_predictions.append({
                        'boxes': pred['boxes'].cpu(),
                        'scores': pred['scores'].cpu(),
                        'labels': pred['labels'].cpu()
                    })
                    
                    all_ground_truths.append({
                        'boxes': batch['bboxes'][i].cpu(),
                        'labels': batch['labels'][i].cpu()
                    })
        
        return all_predictions, all_ground_truths
