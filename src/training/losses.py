"""
Specialized loss functions for imbalanced object detection.

Implements:
- Focal Loss: Down-weights easy examples, focuses on hard negatives
- Class-Balanced Focal Loss: Combines Focal Loss with effective number weighting
- Weighted smooth L1 loss for bounding box regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., ICCV 2017)
    https://arxiv.org/abs/1708.02002
    
    Args:
        alpha: Weighting factor in range (0,1) to balance positive/negative examples
               Can be a scalar or tensor of shape (num_classes,)
        gamma: Focusing parameter γ ≥ 0 (default: 2.0)
               - γ = 0: equivalent to CrossEntropyLoss
               - γ = 2: recommended default
               - Higher γ focuses more on hard examples
        reduction: 'none', 'mean', or 'sum'
    """
    
    def __init__(self, 
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions of shape (N, C) where C = num_classes
            targets: Ground truth labels of shape (N,) with class indices
        
        Returns:
            Focal loss value
        """
        # Compute softmax probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get class probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal loss
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # Alpha is a tensor of shape (num_classes,)
                # Move alpha to the same device as targets
                alpha_t = self.alpha.to(targets.device).gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss combining Focal Loss with effective number weighting.
    
    CB_FL(p_t) = (1-β)/(1-β^n) * FL(p_t)
    
    Paper: "Class-Balanced Loss Based on Effective Number of Samples" (Cui et al., CVPR 2019)
    https://arxiv.org/abs/1901.05555
    
    Args:
        class_counts: Dictionary or tensor with number of samples per class
        num_classes: Total number of classes
        beta: Hyperparameter for effective number (default: 0.9999)
              - For TACO: 0.999 - 0.9999 recommended
              - Higher beta = more emphasis on rare classes
        gamma: Focal loss focusing parameter (default: 2.0)
        reduction: 'none', 'mean', or 'sum'
    """
    
    def __init__(self,
                 class_counts: Dict[int, int],
                 num_classes: int,
                 beta: float = 0.9999,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        super(ClassBalancedFocalLoss, self).__init__()
        
        # Compute effective number weights
        from src.training.class_weights import compute_effective_number_weights
        self.weights = compute_effective_number_weights(class_counts, num_classes, beta)
        
        # Create Focal Loss with class weights
        self.focal_loss = FocalLoss(alpha=self.weights, gamma=gamma, reduction=reduction)
        self.num_classes = num_classes
        self.beta = beta
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions of shape (N, C)
            targets: Ground truth labels of shape (N,)
        
        Returns:
            Class-balanced focal loss value
        """
        # Ensure weights are on same device as inputs
        if self.weights.device != inputs.device:
            self.weights = self.weights.to(inputs.device)
        
        return self.focal_loss(inputs, targets)


class WeightedSmoothL1Loss(nn.Module):
    """
    Smooth L1 loss with optional class-dependent weighting for bbox regression.
    
    Useful for giving more importance to bounding boxes of rare classes.
    
    Args:
        class_weights: Optional tensor of shape (num_classes,) with class weights
        beta: Threshold for smooth L1 (default: 1.0)
        reduction: 'none', 'mean', or 'sum'
    """
    
    def __init__(self,
                 class_weights: Optional[torch.Tensor] = None,
                 beta: float = 1.0,
                 reduction: str = 'mean'):
        super(WeightedSmoothL1Loss, self).__init__()
        self.class_weights = class_weights
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, 
                inputs: torch.Tensor, 
                targets: torch.Tensor,
                class_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            inputs: Predicted bbox coordinates (N, 4)
            targets: Target bbox coordinates (N, 4)
            class_ids: Class IDs for each box (N,), used for weighting
        
        Returns:
            Weighted smooth L1 loss
        """
        loss = F.smooth_l1_loss(inputs, targets, reduction='none', beta=self.beta)
        loss = loss.sum(dim=1)  # Sum over bbox coordinates
        
        # Apply class-dependent weighting if provided
        if self.class_weights is not None and class_ids is not None:
            if self.class_weights.device != loss.device:
                self.class_weights = self.class_weights.to(loss.device)
            weights = self.class_weights[class_ids]
            loss = loss * weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DetectionLoss(nn.Module):
    """
    Combined detection loss for object detection with imbalanced classes.
    
    Combines:
    - Classification loss (Focal Loss or Class-Balanced Focal Loss)
    - Localization loss (Weighted Smooth L1)
    
    Args:
        class_counts: Dictionary with class frequencies
        num_classes: Total number of classes
        loss_type: 'focal' or 'cb_focal' for classification
        beta: Beta for effective number (CB Focal Loss)
        gamma: Gamma for focal loss
        bbox_loss_weight: Weight for bbox regression loss (default: 1.0)
        use_weighted_bbox: Whether to weight bbox loss by class (default: True)
    """
    
    def __init__(self,
                 class_counts: Dict[int, int],
                 num_classes: int,
                 loss_type: str = 'cb_focal',
                 beta: float = 0.9999,
                 gamma: float = 2.0,
                 bbox_loss_weight: float = 1.0,
                 use_weighted_bbox: bool = True):
        super(DetectionLoss, self).__init__()
        
        self.num_classes = num_classes
        self.bbox_loss_weight = bbox_loss_weight
        
        # Create classification loss
        if loss_type == 'focal':
            from src.training.class_weights import compute_effective_number_weights
            weights = compute_effective_number_weights(class_counts, num_classes, beta)
            self.classification_loss = FocalLoss(alpha=weights, gamma=gamma)
        elif loss_type == 'cb_focal':
            self.classification_loss = ClassBalancedFocalLoss(
                class_counts, num_classes, beta=beta, gamma=gamma
            )
        elif loss_type == 'ce':
            # Standard cross entropy with class weights
            from src.training.class_weights import compute_class_weights
            weights = compute_class_weights(class_counts, num_classes, method='effective', beta=beta)
            self.classification_loss = nn.CrossEntropyLoss(weight=weights)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        # Create bbox regression loss
        if use_weighted_bbox:
            from src.training.class_weights import compute_class_weights
            bbox_weights = compute_class_weights(class_counts, num_classes, method='effective', beta=beta)
            self.bbox_loss = WeightedSmoothL1Loss(class_weights=bbox_weights)
        else:
            self.bbox_loss = WeightedSmoothL1Loss(class_weights=None)
    
    def forward(self, 
                class_logits: torch.Tensor,
                box_regression: torch.Tensor,
                class_targets: torch.Tensor,
                box_targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            class_logits: Classification predictions (N, num_classes)
            box_regression: Bbox predictions (N, 4)
            class_targets: Target class labels (N,)
            box_targets: Target bbox coordinates (N, 4)
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # Classification loss
        cls_loss = self.classification_loss(class_logits, class_targets)
        
        # Bbox regression loss (only for positive samples)
        positive_mask = class_targets > 0  # Assuming 0 is background
        if positive_mask.sum() > 0:
            bbox_loss = self.bbox_loss(
                box_regression[positive_mask],
                box_targets[positive_mask],
                class_targets[positive_mask]
            )
        else:
            bbox_loss = torch.tensor(0.0, device=class_logits.device)
        
        # Combine losses
        total_loss = cls_loss + self.bbox_loss_weight * bbox_loss
        
        loss_dict = {
            'loss_classifier': cls_loss.item(),
            'loss_box_reg': bbox_loss.item(),
            'loss_total': total_loss.item()
        }
        
        return total_loss, loss_dict


def get_loss_function(config: dict, class_counts: Dict[int, int], num_classes: int):
    """
    Factory function to create loss function from config.
    
    Args:
        config: Training configuration dictionary
        class_counts: Dictionary with class frequencies
        num_classes: Total number of classes
    
    Returns:
        Loss function module
    
    Example config:
        loss:
          type: 'cb_focal'  # 'focal', 'cb_focal', or 'ce'
          beta: 0.9999
          gamma: 2.0
          bbox_loss_weight: 1.0
          use_weighted_bbox: true
    """
    loss_config = config.get('loss', {})
    loss_type = loss_config.get('type', 'cb_focal')
    
    if loss_type in ['focal', 'cb_focal', 'ce']:
        return DetectionLoss(
            class_counts=class_counts,
            num_classes=num_classes,
            loss_type=loss_type,
            beta=loss_config.get('beta', 0.9999),
            gamma=loss_config.get('gamma', 2.0),
            bbox_loss_weight=loss_config.get('bbox_loss_weight', 1.0),
            use_weighted_bbox=loss_config.get('use_weighted_bbox', True)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")