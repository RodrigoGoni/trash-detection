"""
Training script for TACO object detection.
Trains Faster R-CNN model with MLflow tracking and versioning.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.taco_dataloader import create_dataloader
from src.models.detector import TrashDetector


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train TACO object detection model')
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
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_mlflow(config: dict):
    """Setup MLflow experiment tracking."""
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])


def log_config_to_mlflow(config: dict):
    """Log all configuration to MLflow."""
    # Log main parameters
    mlflow.log_param("model_name", config['model']['name'])
    mlflow.log_param("backbone", config['model']['backbone'])
    mlflow.log_param("num_classes", config['model']['num_classes'])
    mlflow.log_param("batch_size", config['data']['batch_size'])
    mlflow.log_param("num_epochs", config['training']['num_epochs'])
    mlflow.log_param("optimizer", config['training']['optimizer']['type'])
    mlflow.log_param("learning_rate", config['training']['optimizer']['lr'])
    mlflow.log_param("dataset_version", config['data']['dataset_version'])
    
    # Log preprocessing parameters
    for key, value in config['data']['preprocessing'].items():
        mlflow.log_param(f"preprocess_{key}", value)
    
    # Log augmentation parameters
    for key, value in config['data']['augmentation'].items():
        mlflow.log_param(f"aug_{key}", value)
    
    # Log scheduler parameters
    for key, value in config['training']['scheduler'].items():
        mlflow.log_param(f"scheduler_{key}", value)


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['images'].to(device)
        
        # Convert to list of dicts for Faster R-CNN
        targets = []
        for i in range(len(batch['bboxes'])):
            target = {
                'boxes': batch['bboxes'][i].to(device),
                'labels': batch['labels'][i].to(device)
            }
            targets.append(target)
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        num_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {losses.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, device):
    """Validate the model."""
    model.train()  # Keep in train mode to get loss
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            
            # Convert to list of dicts
            targets = []
            for i in range(len(batch['bboxes'])):
                target = {
                    'boxes': batch['bboxes'][i].to(device),
                    'labels': batch['labels'][i].to(device)
                }
                targets.append(target)
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            total_loss += losses.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data']['processed_dir'] = args.data_dir
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader = create_dataloader(
        processed_dir=config['data']['processed_dir'],
        split='train',
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    val_loader = create_dataloader(
        processed_dir=config['data']['processed_dir'],
        split='val',
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print("Creating model...")
    model = TrashDetector(
        num_classes=config['model']['num_classes'],
        backbone=config['model']['backbone']
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        momentum=config['training']['optimizer']['momentum'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['scheduler']['step_size'],
        gamma=config['training']['scheduler']['gamma']
    )
    
    # Setup MLflow
    setup_mlflow(config)
    
    # Start MLflow run
    run_name = f"faster_rcnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # Log configuration
        log_config_to_mlflow(config)
        
        # Log dataset statistics
        dataset_stats_path = Path(config['data']['processed_dir']) / 'dataset_stats.json'
        if dataset_stats_path.exists():
            mlflow.log_artifact(str(dataset_stats_path))
        
        # Log config file
        mlflow.log_artifact(args.config)
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        checkpoint_dir = Path('models/checkpoints')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nStarting training for {config['training']['num_epochs']} epochs...")
        
        for epoch in range(1, config['training']['num_epochs'] + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{config['training']['num_epochs']}")
            print(f"{'='*60}")
            
            # Train
            train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = validate(model, val_loader, device)
            print(f"Val Loss: {val_loss:.4f}")
            
            # Log metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)
            
            # Learning rate scheduler step
            scheduler.step()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                best_model_path = checkpoint_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config
                }, best_model_path)
                
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
                
                # Log to MLflow
                mlflow.log_artifact(str(best_model_path))
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        # Log final metrics
        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("total_epochs", epoch)
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best model saved to: {best_model_path}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
