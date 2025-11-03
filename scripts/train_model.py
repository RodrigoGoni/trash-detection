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
from src.utils.debug_utils import (
    save_batch_with_boxes, 
    save_predictions_comparison,
    analyze_batch_statistics
)


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


def train_one_epoch(model, dataloader, optimizer, device, epoch, debug=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Debug: Save first batch of first epoch
        if debug and batch_idx == 0 and epoch == 1:
            print("\n=== DEBUG: Analyzing training batch ===")
            analyze_batch_statistics(batch['images'], batch['bboxes'], batch['labels'])
            save_batch_with_boxes(
                batch['images'], 
                batch['bboxes'], 
                batch['labels'],
                save_dir='debug_images/train_input',
                prefix='train_batch',
                epoch=epoch
            )
            print("=== DEBUG: Done ===\n")
        
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


def evaluate_with_metrics(model, dataloader, device, num_classes, split='val', verbose=True, debug=False, epoch=None):
    """
    Evaluate model with detection metrics (mAP, Precision, Recall).
    
    Args:
        model: Detection model
        dataloader: DataLoader for evaluation
        device: Device to use
        num_classes: Number of classes
        split: Dataset split name for logging
        verbose: Whether to print detailed output
        debug: Whether to save debug visualizations
        epoch: Current epoch number
    
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
    
    # Debug: Save predictions for first batch
    if debug and split == 'val':
        print("\n=== DEBUG: Analyzing validation predictions ===")
        batch = next(iter(dataloader))
        
        # Analyze input
        print("Input batch:")
        analyze_batch_statistics(batch['images'], batch['bboxes'], batch['labels'])
        
        # Save input
        save_batch_with_boxes(
            batch['images'],
            batch['bboxes'],
            batch['labels'],
            save_dir='debug_images/val_input',
            prefix='val_input',
            epoch=epoch
        )
        
        # Get predictions
        images = batch['images'].to(device)
        model.eval()
        with torch.no_grad():
            predictions = model(images)
        
        # Analyze predictions
        print("\nPredictions:")
        for i, pred in enumerate(predictions):
            print(f"  Image {i}: {len(pred['boxes'])} boxes, "
                  f"scores range [{pred['scores'].min():.4f}, {pred['scores'].max():.4f}]"
                  f" (>{score_threshold}: {(pred['scores'] >= score_threshold).sum()})")
        
        # Save comparison
        save_predictions_comparison(
            batch['images'],
            [p['boxes'] for p in predictions],
            [p['scores'] for p in predictions],
            [p['labels'] for p in predictions],
            batch['bboxes'],
            batch['labels'],
            save_dir='debug_images/val_predictions',
            prefix='val_pred',
            epoch=epoch,
            score_threshold=score_threshold
        )
        print("=== DEBUG: Done ===\n")
    
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
            
            # Train (enable debug for first epoch)
            debug_mode = (epoch == 1)
            train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, debug=debug_mode)
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
                    verbose=False,  # Don't print detailed output during training
                    debug=True,  # Enable debug visualizations
                    epoch=epoch
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
        
        # Save test metrics to JSON
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
