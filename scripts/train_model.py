"""
Training script for trash detection model
"""

from src.utils.mlflow_utils import setup_mlflow_tracking
from src.models.train import Trainer, get_optimizer, get_criterion
from src.models.classifier import TrashClassifier
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.data.dataloader import get_dataloaders
import argparse
import yaml
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train trash detection model')
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to data directory (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with command line arguments
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.device:
        config['training']['device'] = args.device

    # Setup MLflow tracking
    mlflow_tracker = setup_mlflow_tracking(config)

    # Start MLflow run
    run_name = config['experiment'].get('name', 'trash_detection_run')
    mlflow_tracker.start_run(run_name=run_name)

    # Log config
    mlflow_tracker.log_config(config)
    mlflow_tracker.log_params({
        'backbone': config['model']['backbone'],
        'num_classes': config['model']['num_classes'],
        'batch_size': config['data']['batch_size'],
        'learning_rate': config['training']['optimizer']['lr'],
        'num_epochs': config['training']['num_epochs'],
    })

    # Set tags
    tags = config['experiment'].get('tags', [])
    mlflow_tracker.set_tags({f'tag_{i}': tag for i, tag in enumerate(tags)})

    # Setup device
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print(f"Using device: {device}")

    # Get data transforms
    train_transforms = get_train_transforms(config['data']['image_size'])
    val_transforms = get_val_transforms(config['data']['image_size'])

    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        train_transform=train_transforms,
        val_transform=val_transforms
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Create model
    print("Creating model...")
    model = TrashClassifier(
        num_classes=config['model']['num_classes'],
        backbone=config['model']['backbone'],
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout']
    )

    # Get optimizer and criterion
    optimizer = get_optimizer(model, config['training']['optimizer'])
    criterion = get_criterion(config['training']['criterion'])

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=config['training']
    )

    # Train model
    print("Starting training...")
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        save_dir=config['training']['save_dir']
    )

    # Log best model
    mlflow_tracker.log_model(model, "best_model")
    mlflow_tracker.log_metrics({
        'best_val_accuracy': trainer.best_val_acc,
        'best_val_loss': trainer.best_val_loss
    })

    # End MLflow run
    mlflow_tracker.end_run()

    print("Training completed!")


if __name__ == '__main__':
    main()
