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
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.taco_dataloader import create_dataloader
from src.models.detector import TrashDetector
from src.models.evaluate import ObjectDetectionEvaluator


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
    """Setup MLflow experiment tracking with hierarchical structure."""
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    
    # Create hierarchical experiment name for better organization
    # Format: taco-detection/{model_type}/{backbone}/experiments
    experiment_name = f"taco-detection/{config['model']['name'].lower()}/{config['model']['backbone']}/experiments"
    mlflow.set_experiment(experiment_name)


def log_config_to_mlflow(config: dict):
    """Log all configuration to MLflow with organized parameters and tags for benchmarking."""
    mlflow.set_tags({
        # Model identification
        "model.type": config['model']['name'].lower(),
        "model.backbone": config['model']['backbone'],
        "model.variant": "baseline" if all(v == 0.0 for v in config['data']['augmentation'].values()) else "augmented",
        
        # Dataset information
        "data.version": config['data']['dataset_version'],
        "data.size": config['data']['img_size'],
        
        # Training type
        "training.hardware": "gpu" if torch.cuda.is_available() else "cpu",
        "training.batch_size": config['data']['batch_size'],
        "training.optimizer": config['training']['optimizer']['type'],
        
        # Experiment categorization
        "experiment.type": "benchmark",
        "experiment.phase": "development"
    })
    
    mlflow.log_params({
        # Model parameters
        "model.num_classes": config['model']['num_classes'],
        "model.pretrained": config['model'].get('pretrained', True),
        
        # Training parameters
        "train.epochs": config['training']['num_epochs'],
        "train.learning_rate": config['training']['optimizer']['lr'],
        "train.momentum": config['training']['optimizer']['momentum'],
        "train.weight_decay": config['training']['optimizer']['weight_decay'],
        
        # Scheduler parameters
        "scheduler.type": config['training']['scheduler']['type'],
        "scheduler.step_size": config['training']['scheduler']['step_size'],
        "scheduler.gamma": config['training']['scheduler']['gamma']
    })
    
    # Log preprocessing configuration as a single parameter (for reference)
    mlflow.log_param("preprocessing_config", config['data']['preprocessing'])
    
    # Log augmentation configuration as a single parameter
    mlflow.log_param("augmentation_config", config['data']['augmentation'])


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


def evaluate_with_metrics(model, dataloader, device, num_classes, split='val', verbose=True):
    """
    Evaluate model with detection metrics (mAP, Precision, Recall).
    
    Args:
        model: Detection model
        dataloader: DataLoader for evaluation
        device: Device to use
        num_classes: Number of classes
        split: Dataset split name for logging
        verbose: Whether to print detailed output
    
    Returns:
        Dictionary with all metrics
    """
    if verbose:
        print(f"Evaluating on {split} set with detection metrics...")
    
    # Use lower threshold during training to see if model is learning
    score_threshold = 0.05 if split == 'val' else 0.5
    
    evaluator = ObjectDetectionEvaluator(
        model=model,
        device=device,
        num_classes=num_classes,
        score_threshold=score_threshold,
        iou_threshold=0.5
    )
    
    metrics = evaluator.evaluate(dataloader)
    
    return metrics


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
    
    # Get augmentation config for training
    augmentation_config = config['data']['augmentation']
    
    train_loader = create_dataloader(
        processed_dir=config['data']['processed_dir'],
        split='train',
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        augmentation_config=augmentation_config
    )
    
    val_loader = create_dataloader(
        processed_dir=config['data']['processed_dir'],
        split='val',
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        augmentation_config=None  # No augmentation for validation
    )
    
    test_loader = create_dataloader(
        processed_dir=config['data']['processed_dir'],
        split='test',
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        augmentation_config=None  # No augmentation for test
    )
    
    print(f"Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader)}, "
          f"Test batches: {len(test_loader)}")
    
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
            print(f"\nTrain Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = validate(model, val_loader, device)
            print(f"Val Loss: {val_loss:.4f}")
            
            # Log basic metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)
            
            # Evaluate with detection metrics every 5 epochs
            if epoch % 5 == 0 or epoch == config['training']['num_epochs']:
                print(f"\n{'-'*60}")
                print(f"Computing Detection Metrics (Epoch {epoch})...")
                print(f"{'-'*60}")
                
                val_metrics = evaluate_with_metrics(
                    model, val_loader, device, 
                    config['model']['num_classes'], 
                    split='val',
                    verbose=False
                )
                
                # Display metrics in a nice table format
                print(f"\n{'='*60}")
                print(f"VALIDATION METRICS (Epoch {epoch}) [score_threshold=0.05]")
                print(f"{'='*60}")
                print(f"  mAP@0.5    : {val_metrics['mAP']:.4f}")
                print(f"  Precision  : {val_metrics['precision']:.4f}")
                print(f"  Recall     : {val_metrics['recall']:.4f}")
                print(f"  F1-Score   : {val_metrics['f1_score']:.4f}")
                print(f"  TP/FP/FN   : {val_metrics['true_positives']}/{val_metrics['false_positives']}/{val_metrics['false_negatives']}")
                print(f"{'='*60}")
                
                # Log detection metrics to MLflow
                mlflow.log_metric("val_mAP", val_metrics['mAP'], step=epoch)
                mlflow.log_metric("val_precision", val_metrics['precision'], step=epoch)
                mlflow.log_metric("val_recall", val_metrics['recall'], step=epoch)
                mlflow.log_metric("val_f1_score", val_metrics['f1_score'], step=epoch)
            
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
                
                print(f">> Saved best model (val_loss: {val_loss:.4f})")
                
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
        
        # Final evaluation on test set
        print(f"\n{'='*60}")
        print("Final Evaluation on Test Set")
        print(f"{'='*60}")
        
        # Load best model
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate on test set with detection metrics
        print("Evaluating best model on test set...")
        test_metrics = evaluate_with_metrics(
            model, test_loader, device,
            config['model']['num_classes'],
            split='test',
            verbose=True  # Show detailed output for final evaluation
        )
        
        # Print detailed test metrics
        print(f"\n{'='*60}")
        print(f"FINAL TEST SET METRICS")
        print(f"{'='*60}")
        print(f"  mAP@0.5    : {test_metrics['mAP']:.4f}")
        print(f"  Precision  : {test_metrics['precision']:.4f}")
        print(f"  Recall     : {test_metrics['recall']:.4f}")
        print(f"  F1-Score   : {test_metrics['f1_score']:.4f}")
        print(f"  TP/FP/FN   : {test_metrics['true_positives']}/{test_metrics['false_positives']}/{test_metrics['false_negatives']}")
        print(f"{'='*60}")
        
        # Log test metrics to MLflow
        mlflow.log_metric("test_mAP", test_metrics['mAP'])
        mlflow.log_metric("test_precision", test_metrics['precision'])
        mlflow.log_metric("test_recall", test_metrics['recall'])
        mlflow.log_metric("test_f1_score", test_metrics['f1_score'])
        mlflow.log_metric("test_true_positives", test_metrics['true_positives'])
        mlflow.log_metric("test_false_positives", test_metrics['false_positives'])
        mlflow.log_metric("test_false_negatives", test_metrics['false_negatives'])
        
        mlflow.log_metrics({
            "benchmark.map50": test_metrics['mAP'],
            "benchmark.f1": test_metrics['f1_score'],
            "benchmark.precision": test_metrics['precision'],
            "benchmark.recall": test_metrics['recall'],
            "benchmark.true_positives": test_metrics['true_positives'],
            "benchmark.false_positives": test_metrics['false_positives'],
            "benchmark.false_negatives": test_metrics['false_negatives']
        })
        
        # Save test metrics to JSON for detailed analysis
        test_metrics_path = checkpoint_dir / 'test_metrics.json'
        with open(test_metrics_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in test_metrics.items()}
            json.dump(json_metrics, f, indent=2)
        
        mlflow.log_artifact(str(test_metrics_path))
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"\nTraining Summary:")
        print(f"  Total Epochs     : {epoch}")
        print(f"  Best Val Loss    : {best_val_loss:.4f}")
        print(f"\nFinal Test Set Performance:")
        print(f"  mAP@0.5          : {test_metrics['mAP']:.4f}")
        print(f"  Precision        : {test_metrics['precision']:.4f}")
        print(f"  Recall           : {test_metrics['recall']:.4f}")
        print(f"  F1-Score         : {test_metrics['f1_score']:.4f}")
        print(f"\nSaved Artifacts:")
        print(f"  Best model       : {best_model_path}")
        print(f"  Test metrics     : {test_metrics_path}")
        print(f"\nView results in MLflow:")
        print(f"  mlflow ui")
        print(f"  http://localhost:5000")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
