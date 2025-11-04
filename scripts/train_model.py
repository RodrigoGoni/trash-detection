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
import mlflow
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.taco_dataloader import create_dataloader
from src.models.detector import TrashDetector
from src.models.evaluate import ObjectDetectionEvaluator
from src.training.optimizers import create_optimizer, create_scheduler
from src.utils.system_info import get_system_info, log_gpu_metrics_to_mlflow
from src.utils.training_logger import TrainingLogger
from src.utils.mlflow_utils import (
    setup_mlflow,
    get_dataset_info,
    log_system_tags,
    log_experiment_tags,
    log_dataset_tags,
    register_dataset_in_mlflow,
    log_augmentation_tags,
    log_config_parameters,
    log_training_tags,
    log_system_parameters,
    log_dataset_parameters,
    register_model_in_mlflow,
    log_epoch_metrics,
    log_validation_detection_metrics,
    log_test_metrics,
    log_final_training_metrics
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


def train_one_epoch(model, dataloader, optimizer, device, epoch, scaler=None, grad_clip_norm=None):
    """Train for one epoch with optional AMP and gradient clipping."""
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
        
        optimizer.zero_grad()
        
        # Forward pass with optional AMP
        if scaler is not None:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass with gradient scaling
            scaler.scale(losses).backward()
            
            # Gradient clipping (unscale first)
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            losses.backward()
            
            # Gradient clipping
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            
            # Optimizer step
            optimizer.step()
        
        total_loss += losses.item()
        num_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            TrainingLogger.print_batch_progress(
                epoch, batch_idx, len(dataloader), losses.item(), frequency=50
            )
    
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
    
    TrainingLogger.print_dataloader_info(len(train_loader), len(val_loader), len(test_loader))
    
    # Create model
    print("Creating model...")
    model = TrashDetector(
        num_classes=config['model']['num_classes'],
        backbone=config['model']['backbone']
    ).to(device)
    
    # Create optimizer using factory function
    optimizer = create_optimizer(model.parameters(), config['training']['optimizer'])
    
    # Create scheduler using factory function
    scheduler = create_scheduler(
        optimizer, 
        config['training']['scheduler'], 
        config['training']['num_epochs']
    )
    
    # Setup mixed precision training (AMP)
    use_amp = config['training'].get('use_amp', False)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Setup MLflow
    setup_mlflow(config)
    
    # Get system and dataset info
    system_info = get_system_info()
    dataset_info = get_dataset_info(config['data']['processed_dir'])
    
    # Print setup information
    TrainingLogger.print_setup_info(config, system_info, dataset_info)
    
    # Create comprehensive run name
    dataset_version = config['data'].get('dataset_version', 'unknown')
    run_name = f"faster_rcnn_{dataset_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # ========================================
        # ENABLE SYSTEM METRICS LOGGING
        # ========================================
        # This will automatically log CPU, RAM, disk, and network usage
        mlflow.enable_system_metrics_logging()
        
        # ========================================
        # LOG ALL TAGS (System, Experiment, Dataset, Training, Augmentation)
        # ========================================
        log_system_tags(system_info)
        log_experiment_tags(config)
        log_training_tags(config)
        
        # Register dataset in MLflow and log tags
        log_dataset_tags(dataset_info, dataset_version)
        register_dataset_in_mlflow(dataset_info, dataset_version, config)
        
        # Log augmentation tags
        aug_config = config['data']['augmentation']
        enabled_augs = log_augmentation_tags(aug_config)
        

        # Log configuration
        log_config_parameters(config)
        log_system_parameters(system_info)
        log_dataset_parameters(dataset_info)
        
        # Log dataset statistics
        dataset_stats_path = Path(config['data']['processed_dir']) / 'dataset_stats.json'
        if dataset_stats_path.exists():
            mlflow.log_artifact(str(dataset_stats_path))
        
        # Log config file
        mlflow.log_artifact(args.config)
        
        # Create checkpoint directory
        checkpoint_dir = Path('models/checkpoints')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Log system info as artifact
        system_info_path = checkpoint_dir / 'system_info.json'
        with open(system_info_path, 'w') as f:
            json.dump(system_info, f, indent=2)
        mlflow.log_artifact(str(system_info_path))
        
        # Print MLflow tracking info
        TrainingLogger.print_mlflow_info(
            run_id=mlflow.active_run().info.run_id,
            experiment_name=config['experiment']['name'],
            dataset_version=dataset_version,
            dataset_hash=dataset_info.get('dataset_hash', 'unknown')
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        TrainingLogger.print_section(f"TRAINING ({config['training']['num_epochs']} epochs)")
        
        for epoch in range(1, config['training']['num_epochs'] + 1):
            TrainingLogger.print_epoch_header(epoch, config['training']['num_epochs'])
            
            # Get gradient clipping value
            grad_clip_norm = config['training'].get('grad_clip_norm', None)
            
            # Train with AMP and gradient clipping support
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device, epoch,
                scaler=scaler,
                grad_clip_norm=grad_clip_norm
            )
            
            # Validate
            val_loss = validate(model, val_loader, device)
            
            # Log basic metrics
            log_epoch_metrics(
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=optimizer.param_groups[0]['lr'],
                step=epoch
            )
            
            # Log GPU metrics
            log_gpu_metrics_to_mlflow(step=epoch)
            
            # Evaluate with detection metrics based on val_frequency
            val_frequency = config['training'].get('val_frequency', 1)
            if epoch % val_frequency == 0 or epoch == config['training']['num_epochs']:
                val_metrics = evaluate_with_metrics(
                    model, val_loader, device, 
                    config['model']['num_classes'], 
                    split='val',
                    verbose=False
                )
                
                # Log detection metrics to MLflow
                log_validation_detection_metrics(val_metrics, step=epoch, prefix="val")
                
                # Print compact validation metrics
                TrainingLogger.print_validation_metrics(epoch, val_metrics, threshold=0.05)
            
            # Learning rate scheduler step
            scheduler.step()
            
            # Save best model
            is_best = val_loss < best_val_loss
            if is_best:
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
                
                # Log to MLflow
                mlflow.log_artifact(str(best_model_path))
            else:
                patience_counter += 1
            
            # Print epoch summary
            patience_info = f"Early stopping: {patience_counter}/{patience}" if patience_counter > 0 else None
            TrainingLogger.print_epoch_summary(
                epoch, train_loss, val_loss, 
                optimizer.param_groups[0]['lr'],
                best_model=is_best,
                patience_info=patience_info
            )
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                break
        
        # Log final metrics
        log_final_training_metrics(best_val_loss, epoch)
        
        # Final evaluation on test set
        TrainingLogger.print_section("FINAL EVALUATION")
        
        # Load best model
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate on test set with detection metrics
        test_metrics = evaluate_with_metrics(
            model, test_loader, device,
            config['model']['num_classes'],
            split='test',
            verbose=False
        )
        
        # Print test metrics
        TrainingLogger.print_test_metrics(test_metrics, threshold=0.5)
        
        # Log test metrics to MLflow
        log_test_metrics(test_metrics)
        
        # Save test metrics to JSON for detailed analysis
        test_metrics_path = checkpoint_dir / 'test_metrics.json'
        with open(test_metrics_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in test_metrics.items()}
            json.dump(json_metrics, f, indent=2)
        
        mlflow.log_artifact(str(test_metrics_path))
        
        # Register model in MLflow
        TrainingLogger.print_subsection("Model Registration")
        
        version = register_model_in_mlflow(
            model=model,
            config=config,
            test_metrics=test_metrics,
            dataset_version=dataset_version,
            dataset_info=dataset_info,
            enabled_augs=enabled_augs,
            epoch=epoch,
            use_amp=use_amp
        )
        
        if version:
            model_name = f"taco-{config['model']['name'].lower()}-{config['model']['backbone']}"
            TrainingLogger.print_model_registration(
                model_name=model_name,
                version=version,
                metrics=test_metrics,
                dataset_version=dataset_version,
                optimizer=config['training']['optimizer']['type'].upper(),
                scheduler=config['training']['scheduler']['type']
            )
        
        # Print final summary
        TrainingLogger.print_training_complete(
            epoch=epoch,
            best_val_loss=best_val_loss,
            test_metrics=test_metrics,
            paths={
                'model': best_model_path,
                'metrics': test_metrics_path
            }
        )


if __name__ == '__main__':
    main()
