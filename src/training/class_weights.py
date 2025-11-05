"""
Class weights computation for imbalanced datasets.

Provides multiple strategies for computing class weights to handle class imbalance:
- Inverse frequency
- Effective number of samples (Class-Balanced weighting)
- Square root inverse frequency
"""

import torch
import numpy as np
from typing import Dict, List, Union
from collections import Counter


def compute_inverse_frequency_weights(class_counts: Dict[int, int], 
                                     num_classes: int,
                                     smooth: float = 1.0) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequency.
    
    Weight = total_samples / (num_classes * class_count + smooth)
    
    Args:
        class_counts: Dictionary mapping class_id -> count
        num_classes: Total number of classes
        smooth: Smoothing factor to avoid division by zero (default: 1.0)
    
    Returns:
        Tensor of shape (num_classes,) with class weights
    """
    total_samples = sum(class_counts.values())
    weights = torch.zeros(num_classes)
    
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 0)
        if count > 0:
            weights[class_id] = total_samples / (num_classes * (count + smooth))
        else:
            # Assign high weight for missing classes (if any)
            weights[class_id] = total_samples / (num_classes * smooth)
    
    return weights


def compute_effective_number_weights(class_counts: Dict[int, int],
                                    num_classes: int,
                                    beta: float = 0.9999) -> torch.Tensor:
    """
    Compute class weights using effective number of samples.
    
    Based on "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)
    https://arxiv.org/abs/1901.05555
    
    Effective number: E_n = (1 - β^n) / (1 - β)
    Weight = (1 - β) / (1 - β^n)
    
    Args:
        class_counts: Dictionary mapping class_id -> count
        num_classes: Total number of classes
        beta: Hyperparameter (default: 0.9999 for large datasets, 0.999 for small)
              - Higher beta gives more emphasis to rare classes
              - β ∈ [0, 1), typically 0.99-0.9999
    
    Returns:
        Tensor of shape (num_classes,) with class weights
    """
    weights = torch.zeros(num_classes)
    
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 0)
        if count > 0:
            effective_num = (1.0 - np.power(beta, count)) / (1.0 - beta)
            weights[class_id] = 1.0 / effective_num
        else:
            # Assign maximum weight for missing classes
            weights[class_id] = 1.0 / (1.0 - beta)
    
    # Normalize weights to have mean = 1.0
    weights = weights / weights.mean()
    
    return weights


def compute_sqrt_inverse_frequency_weights(class_counts: Dict[int, int],
                                          num_classes: int,
                                          smooth: float = 1.0) -> torch.Tensor:
    """
    Compute class weights using square root of inverse frequency.
    
    This is a middle ground between no weighting and full inverse frequency.
    Weight = sqrt(total_samples / class_count)
    
    Args:
        class_counts: Dictionary mapping class_id -> count
        num_classes: Total number of classes
        smooth: Smoothing factor (default: 1.0)
    
    Returns:
        Tensor of shape (num_classes,) with class weights
    """
    total_samples = sum(class_counts.values())
    weights = torch.zeros(num_classes)
    
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 0)
        if count > 0:
            weights[class_id] = np.sqrt(total_samples / (count + smooth))
        else:
            weights[class_id] = np.sqrt(total_samples / smooth)
    
    # Normalize weights to have mean = 1.0
    weights = weights / weights.mean()
    
    return weights


def load_class_counts_from_annotations(annotations: List[dict]) -> Dict[int, int]:
    """
    Extract class counts from COCO-format annotations.
    
    Args:
        annotations: List of annotation dictionaries with 'category_id' field
    
    Returns:
        Dictionary mapping class_id -> count
    """
    class_counts = Counter(ann['category_id'] for ann in annotations)
    return dict(class_counts)


def compute_class_weights(class_counts: Union[Dict[int, int], List[dict]],
                         num_classes: int,
                         method: str = 'inverse',
                         beta: float = 0.9999,
                         smooth: float = 1.0) -> torch.Tensor:
    """
    Unified interface for computing class weights.
    
    Args:
        class_counts: Either dict of {class_id: count} or list of annotations
        num_classes: Total number of classes
        method: Weighting method - 'inverse', 'effective', or 'sqrt'
        beta: Beta parameter for effective number method (default: 0.9999)
        smooth: Smoothing factor for inverse methods (default: 1.0)
    
    Returns:
        Tensor of shape (num_classes,) with class weights
    
    Example:
        >>> # From annotation file
        >>> import json
        >>> with open('train/annotations.json') as f:
        >>>     data = json.load(f)
        >>> weights = compute_class_weights(data['annotations'], num_classes=60, method='effective')
        >>> 
        >>> # Or from precomputed counts
        >>> counts = {0: 100, 1: 50, 2: 200}
        >>> weights = compute_class_weights(counts, num_classes=3, method='inverse')
    """
    # Handle list of annotations
    if isinstance(class_counts, list):
        class_counts = load_class_counts_from_annotations(class_counts)
    
    # Compute weights based on method
    if method == 'inverse':
        weights = compute_inverse_frequency_weights(class_counts, num_classes, smooth)
    elif method == 'effective':
        weights = compute_effective_number_weights(class_counts, num_classes, beta)
    elif method == 'sqrt':
        weights = compute_sqrt_inverse_frequency_weights(class_counts, num_classes, smooth)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'inverse', 'effective', or 'sqrt'")
    
    return weights


def print_class_weights_summary(weights: torch.Tensor, 
                               class_counts: Dict[int, int],
                               class_names: Dict[int, str] = None,
                               top_k: int = 10):
    """
    Print summary of class weights for analysis.
    
    Args:
        weights: Tensor of class weights
        class_counts: Dictionary of class counts
        class_names: Optional dictionary mapping class_id -> name
        top_k: Number of top weighted classes to show
    """
    print("\n" + "="*60)
    print("CLASS WEIGHTS SUMMARY")
    print("="*60)
    
    # Get weight-class pairs
    weight_pairs = [(i, w.item(), class_counts.get(i, 0)) for i, w in enumerate(weights)]
    weight_pairs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {top_k} highest weighted classes:")
    print(f"{'Class ID':<10} {'Count':<10} {'Weight':<10} {'Name':<30}")
    print("-" * 60)
    for i, (class_id, weight, count) in enumerate(weight_pairs[:top_k]):
        name = class_names.get(class_id, 'N/A') if class_names else 'N/A'
        print(f"{class_id:<10} {count:<10} {weight:<10.4f} {name:<30}")
    
    print(f"\nBottom {top_k} lowest weighted classes:")
    print(f"{'Class ID':<10} {'Count':<10} {'Weight':<10} {'Name':<30}")
    print("-" * 60)
    for i, (class_id, weight, count) in enumerate(weight_pairs[-top_k:]):
        name = class_names.get(class_id, 'N/A') if class_names else 'N/A'
        print(f"{class_id:<10} {count:<10} {weight:<10.4f} {name:<30}")
    
    print("\nStatistics:")
    print(f"  Mean weight: {weights.mean():.4f}")
    print(f"  Std weight:  {weights.std():.4f}")
    print(f"  Min weight:  {weights.min():.4f}")
    print(f"  Max weight:  {weights.max():.4f}")
    print(f"  Weight ratio (max/min): {(weights.max() / weights.min()).item():.2f}x")
    print("="*60)


if __name__ == '__main__':
    # Example usage
    import json
    from pathlib import Path
    
    # Load training annotations
    train_ann_path = Path('data/processed/train/annotations.json')
    if train_ann_path.exists():
        with open(train_ann_path) as f:
            data = json.load(f)
        
        # Get class names
        class_names = {cat['id']: cat['name'] for cat in data['categories']}
        num_classes = len(data['categories'])
        
        print("Computing class weights for TACO dataset...")
        print(f"Total classes: {num_classes}")
        print(f"Total annotations: {len(data['annotations'])}")
        
        # Test all methods
        for method in ['inverse', 'effective', 'sqrt']:
            print(f"\n\n{'='*60}")
            print(f"METHOD: {method.upper()}")
            print(f"{'='*60}")
            
            weights = compute_class_weights(
                data['annotations'],
                num_classes=num_classes,
                method=method
            )
            
            class_counts = load_class_counts_from_annotations(data['annotations'])
            print_class_weights_summary(weights, class_counts, class_names, top_k=15)
    else:
        print(f"Training annotations not found at {train_ann_path}")
        print("Please run prepare_data.py first")
