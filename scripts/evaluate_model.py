"""
Script to evaluate trained object detection model on test set.
Supports both classification and detection models.
"""

import sys
import json
import argparse
from pathlib import Path

import yaml
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.taco_dataloader import create_dataloader
from src.models.detector import TrashDetector
from src.models.evaluate import ObjectDetectionEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained detection model on test set')
    parser.add_argument('--checkpoint', '--model-path', type=str, 
                        default='models/checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, 
                        default='data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--score-threshold', type=float, default=0.5,
                        help='Score threshold for filtering predictions')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for matching')
    parser.add_argument('--output', '--output-dir', type=str, 
                        default='models/checkpoints/test_metrics.json',
                        help='Path to save metrics JSON')
    return parser.parse_args()




def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print(f"Checkpoint info:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    if 'val_loss' in checkpoint:
        print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    
    # Get model config
    if 'config' in checkpoint:
        config = checkpoint['config']
        num_classes = config['model']['num_classes']
        backbone = config['model']['backbone']
    else:
        # Default values
        num_classes = 61
        backbone = 'resnet50'
        print("Warning: Using default model configuration")
    
    # Create model
    print(f"\nCreating model (backbone={backbone}, num_classes={num_classes})...")
    model = TrashDetector(
        num_classes=num_classes,
        backbone=backbone
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")
    
    # Create dataloader
    print(f"\nLoading {args.split} dataset from: {args.data_dir}")
    dataloader = create_dataloader(
        processed_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    print(f"{args.split.capitalize()} dataset: {len(dataloader)} batches")
    
    # Create evaluator
    print("\nCreating evaluator...")
    evaluator = ObjectDetectionEvaluator(
        model=model,
        device=device,
        num_classes=num_classes,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold
    )
    
    # Evaluate
    print(f"\n{'='*60}")
    print(f"Evaluating on {args.split.upper()} Set")
    print(f"{'='*60}")
    print(f"Score threshold: {args.score_threshold}")
    print(f"IoU threshold: {args.iou_threshold}")
    print()
    
    metrics = evaluator.evaluate(dataloader)
    
    # Print metrics
    print(f"\n{'='*60}")
    print(f"{args.split.upper()} SET RESULTS")
    print(f"{'='*60}")
    print(f"\nOverall Metrics:")
    print(f"  mAP@{args.iou_threshold}: {metrics['mAP']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    
    print(f"\nDetection Counts:")
    print(f"  True Positives: {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    
    # Per-class AP
    per_class_ap = {k: v for k, v in metrics.items() if k.startswith('AP_class_')}
    if per_class_ap:
        sorted_ap = sorted(per_class_ap.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nPer-Class Average Precision:")
        print(f"  Best class: {sorted_ap[0][0]} = {sorted_ap[0][1]:.4f}")
        print(f"  Worst class: {sorted_ap[-1][0]} = {sorted_ap[-1][1]:.4f}")
        
        # Count classes with AP > 0.5
        good_classes = sum(1 for _, ap in per_class_ap.items() if ap > 0.5)
        print(f"  Classes with AP > 0.5: {good_classes}/{len(per_class_ap)}")
    
    print(f"{'='*60}")
    
    # Save metrics to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Convert to serializable format
        json_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                json_metrics[k] = float(v)
            else:
                json_metrics[k] = v
        
        json.dump(json_metrics, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {output_path}")
    
    # Print top 10 and bottom 10 classes if available
    if per_class_ap and len(per_class_ap) > 10:
        print(f"\nTop 10 Classes by AP:")
        for i, (class_name, ap) in enumerate(sorted_ap[:10], 1):
            print(f"  {i}. {class_name}: {ap:.4f}")
        
        print(f"\nBottom 10 Classes by AP:")
        for i, (class_name, ap) in enumerate(sorted_ap[-10:], 1):
            print(f"  {i}. {class_name}: {ap:.4f}")


if __name__ == '__main__':
    main()
