"""
Script to evaluate trained model
"""

from src.models.evaluate import ModelEvaluator
from src.models.classifier import TrashClassifier
from src.data.augmentation import get_val_transforms
from src.data.dataloader import get_dataloaders
import argparse
import yaml
import json
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate trash detection model')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to data directory (overrides config)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--output-dir', type=str, default='experiments',
                        help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main evaluation function"""
    # Parse arguments
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override config
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir

    # Setup device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}")
    model = TrashClassifier(
        num_classes=config['model']['num_classes'],
        backbone=config['model']['backbone'],
        pretrained=False
    )

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Get data transforms
    val_transforms = get_val_transforms(config['data']['image_size'])

    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        val_transform=val_transforms
    )

    # Select dataloader based on split
    if args.split == 'train':
        dataloader = train_loader
    elif args.split == 'val':
        dataloader = val_loader
    elif args.split == 'test':
        if test_loader is None:
            raise ValueError("Test set not found")
        dataloader = test_loader

    print(
        f"Evaluating on {args.split} set ({len(dataloader.dataset)} samples)")

    # Create evaluator
    evaluator = ModelEvaluator(model, device=device)

    # Evaluate
    print("Evaluating model...")
    metrics = evaluator.evaluate(dataloader)

    # Print metrics
    print("\nEvaluation Metrics:")
    print("-" * 50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # Get class names
    class_names = dataloader.dataset.classes

    # Generate classification report
    report = evaluator.classification_report_dict(dataloader, class_names)

    # Per-class accuracy
    per_class_acc = evaluator.per_class_accuracy(dataloader, class_names)

    print("\nPer-Class Accuracy:")
    print("-" * 50)
    for class_name, acc in per_class_acc.items():
        print(f"{class_name}: {acc:.4f}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    with open(output_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / 'classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    with open(output_dir / 'per_class_accuracy.json', 'w') as f:
        json.dump(per_class_acc, f, indent=2)

    # Generate confusion matrix
    cm = evaluator.confusion_matrix(
        dataloader,
        class_names=class_names,
        save_path=output_dir / 'confusion_matrix.png'
    )

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
