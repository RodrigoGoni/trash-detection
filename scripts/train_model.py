"""
Training script for TACO object detection.
Trains Faster R-CNN model with MLflow tracking and versioning.
"""

import sys
import json
import argparse
import platform
import hashlib
import psutil
from pathlib import Path
from datetime import datetime

import yaml
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd

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


def get_system_info():
    """Get comprehensive system information for reproducibility."""
    info = {
        # Python & Libraries
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'numpy_version': np.__version__,
        
        # System
        'platform': platform.platform(),
        'hostname': platform.node(),
        'processor': platform.processor(),
        
        # CPU
        'cpu_count': psutil.cpu_count(logical=False),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        
        # Memory
        'ram_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'ram_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
        
        # Disk
        'disk_total_gb': round(psutil.disk_usage('/').total / (1024**3), 2),
        'disk_free_gb': round(psutil.disk_usage('/').free / (1024**3), 2),
        
        # CUDA
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_name'] = torch.cuda.get_device_name(0)
        
        # GPU Memory
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        info['gpu_memory_gb'] = round(gpu_mem / (1024**3), 2)
    
    return info


def get_dataset_info(data_dir: Path):
    """Get dataset information and generate version hash."""
    data_dir = Path(data_dir)
    
    info = {
        'path': str(data_dir.resolve()),
        'exists': data_dir.exists()
    }
    
    if not data_dir.exists():
        return info
    
    # Count files per split
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split / 'images'
        if split_dir.exists():
            images = list(split_dir.rglob('*.jpg')) + list(split_dir.rglob('*.png'))
            info[f'{split}_images'] = len(images)
            info[f'{split}_size_mb'] = round(
                sum(img.stat().st_size for img in images) / (1024**2), 2
            )
    
    # Load dataset stats if available
    stats_file = data_dir / 'dataset_stats.json'
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
            info['num_classes'] = stats.get('num_classes', 0)
            info['total_annotations'] = stats.get('total_annotations', 0)
    
    # Generate dataset hash based on annotations files
    hash_md5 = hashlib.md5()
    for split in ['train', 'val', 'test']:
        ann_file = data_dir / split / 'annotations.json'
        if ann_file.exists():
            with open(ann_file, 'rb') as f:
                hash_md5.update(f.read())
    
    info['dataset_hash'] = hash_md5.hexdigest()[:12]  # Short hash
    
    # Creation time
    if (data_dir / 'train').exists():
        info['created_timestamp'] = datetime.fromtimestamp(
            (data_dir / 'train').stat().st_mtime
        ).isoformat()
    
    return info


def log_gpu_metrics(step: int):
    """Log GPU metrics to MLflow if CUDA is available."""
    if not torch.cuda.is_available():
        return
    
    try:
        # GPU memory metrics
        for i in range(torch.cuda.device_count()):
            # Memory allocated by PyTorch
            allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(i) / (1024**3)    # GB
            max_allocated = torch.cuda.max_memory_allocated(i) / (1024**3)  # GB
            
            mlflow.log_metric(f"gpu_{i}_memory_allocated_gb", allocated, step=step)
            mlflow.log_metric(f"gpu_{i}_memory_reserved_gb", reserved, step=step)
            mlflow.log_metric(f"gpu_{i}_memory_max_allocated_gb", max_allocated, step=step)
            
            # GPU utilization (if available)
            try:
                utilization = torch.cuda.utilization(i)
                mlflow.log_metric(f"gpu_{i}_utilization_percent", utilization, step=step)
            except:
                pass  # Not all CUDA versions support this
    except Exception as e:
        # Don't fail training if GPU metrics logging fails
        pass


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
    
    # Create optimizer (support SGD, Adam, AdamW)
    optimizer_type = config['training']['optimizer']['type'].lower()
    
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['training']['optimizer']['lr'],
            momentum=config['training']['optimizer'].get('momentum', 0.9),
            weight_decay=config['training']['optimizer'].get('weight_decay', 0.0005)
        )
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['optimizer']['lr'],
            betas=config['training']['optimizer'].get('betas', [0.9, 0.999]),
            weight_decay=config['training']['optimizer'].get('weight_decay', 0.0001),
            eps=config['training']['optimizer'].get('eps', 1e-8)
        )
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['optimizer']['lr'],
            betas=config['training']['optimizer'].get('betas', [0.9, 0.999]),
            weight_decay=config['training']['optimizer'].get('weight_decay', 0.01),
            eps=config['training']['optimizer'].get('eps', 1e-8)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    print(f"Using optimizer: {optimizer_type.upper()}")
    
    # Create scheduler (support StepLR, CosineAnnealing, CosineAnnealingWarmup)
    scheduler_type = config['training']['scheduler']['type'].lower()
    
    if scheduler_type == 'step_lr':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['scheduler']['step_size'],
            gamma=config['training']['scheduler']['gamma']
        )
        warmup_scheduler = None
    elif scheduler_type == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs'],
            eta_min=config['training']['scheduler'].get('min_lr', 1e-6)
        )
        warmup_scheduler = None
    elif scheduler_type == 'cosine_annealing_warmup':
        warmup_epochs = config['training']['scheduler'].get('warmup_epochs', 5)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs'] - warmup_epochs,
            eta_min=config['training']['scheduler'].get('min_lr', 1e-6)
        )
        # Linear warmup
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    print(f"Using scheduler: {scheduler_type}")
    
    # Setup mixed precision training (AMP)
    use_amp = config['training'].get('use_amp', False)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Using Automatic Mixed Precision (AMP) training")
    
    # Setup MLflow
    setup_mlflow(config)
    
    # Get system and dataset info
    print("\nCollecting system information...")
    system_info = get_system_info()
    
    print("Collecting dataset information...")
    dataset_info = get_dataset_info(config['data']['processed_dir'])
    
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
        # SYSTEM TAGS
        # ========================================
        mlflow.set_tag("system.hostname", system_info['hostname'])
        mlflow.set_tag("system.platform", system_info['platform'])
        mlflow.set_tag("system.python_version", system_info['python_version'])
        mlflow.set_tag("system.pytorch_version", system_info['pytorch_version'])
        mlflow.set_tag("system.cuda_available", system_info['cuda_available'])
        
        if system_info['cuda_available']:
            mlflow.set_tag("system.gpu_name", system_info['gpu_name'])
            mlflow.set_tag("system.cuda_version", system_info['cuda_version'])
            mlflow.set_tag("system.gpu_memory_gb", system_info['gpu_memory_gb'])
        
        # ========================================
        # EXPERIMENT TAGS (from config)
        # ========================================
        if 'tags' in config['experiment']:
            for tag in config['experiment']['tags']:
                mlflow.set_tag(f"experiment.{tag}", "true")
        
        if 'description' in config['experiment']:
            mlflow.set_tag("mlflow.note.content", config['experiment']['description'])
        
        # ========================================
        # DATASET TAGS & REGISTRATION
        # ========================================
        mlflow.set_tag("dataset.version", dataset_version)
        mlflow.set_tag("dataset.hash", dataset_info.get('dataset_hash', 'unknown'))
        mlflow.set_tag("dataset.path", dataset_info['path'])
        
        if 'train_images' in dataset_info:
            mlflow.set_tag("dataset.train_images", dataset_info['train_images'])
            mlflow.set_tag("dataset.val_images", dataset_info['val_images'])
            mlflow.set_tag("dataset.test_images", dataset_info['test_images'])
        
        # Register dataset in MLflow (appears in Datasets tab)
        dataset_source = mlflow.data.from_pandas(
            pd.DataFrame([{
                'name': 'TACO-trash-detection',
                'version': dataset_version,
                'hash': dataset_info.get('dataset_hash', 'unknown'),
                'train_images': dataset_info.get('train_images', 0),
                'val_images': dataset_info.get('val_images', 0),
                'test_images': dataset_info.get('test_images', 0),
                'num_classes': dataset_info.get('num_classes', config['model']['num_classes']),
                'total_annotations': dataset_info.get('total_annotations', 0),
                'path': dataset_info['path']
            }]),
            source=dataset_info['path'],
            name="TACO-dataset"
        )
        mlflow.log_input(dataset_source, context="training")
        
        aug_config = config['data']['augmentation']
        enabled_augs = [k for k, v in aug_config.items() if k.endswith('_p') and v > 0]
        if enabled_augs:
            mlflow.set_tag("augmentation.enabled", ", ".join(enabled_augs))
            mlflow.set_tag("augmentation.status", "enabled")
        else:
            mlflow.set_tag("augmentation.status", "disabled")
        

        # Log configuration
        log_config_to_mlflow(config)
        
        # Log system info as params
        for key, value in system_info.items():
            mlflow.log_param(f"sys_{key}", value)
        
        # Log dataset info as params
        for key, value in dataset_info.items():
            if key not in ['path']:  # path is already in tags
                mlflow.log_param(f"data_{key}", value)
        
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
        
        print(f"\n{'='*60}")
        print(f"MLflow Run: {mlflow.active_run().info.run_id}")
        print(f"Experiment: {config['experiment']['name']}")
        print(f"Dataset: {dataset_version} (hash: {dataset_info.get('dataset_hash', 'unknown')})")
        print(f"GPU: {system_info.get('gpu_name', 'CPU only')}")
        print(f"{'='*60}")
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        print(f"\nStarting training for {config['training']['num_epochs']} epochs...")
        
        for epoch in range(1, config['training']['num_epochs'] + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{config['training']['num_epochs']}")
            print(f"{'='*60}")
            
            # Get gradient clipping value
            grad_clip_norm = config['training'].get('grad_clip_norm', None)
            
            # Train with AMP and gradient clipping support
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device, epoch,
                scaler=scaler,
                grad_clip_norm=grad_clip_norm
            )
            print(f"\nTrain Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = validate(model, val_loader, device)
            print(f"Val Loss: {val_loss:.4f}")
            
            # Log basic metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)
            
            # Log GPU metrics
            log_gpu_metrics(step=epoch)
            
            # Evaluate with detection metrics based on val_frequency
            val_frequency = config['training'].get('val_frequency', 1)
            if epoch % val_frequency == 0 or epoch == config['training']['num_epochs']:
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
        print("Registering model in MLflow Model Registry...")
        print(f"{'='*60}")
        
        # Log the model (will auto-register if registered_model_name is provided)
        model_name = f"taco-{config['model']['name'].lower()}-{config['model']['backbone']}"
        
        # Log model to MLflow
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )
        
        # Get model version tags
        client = mlflow.tracking.MlflowClient()
        
        # Get all versions and find the latest one (modern API)
        model_versions = client.search_model_versions(f"name='{model_name}'")
        if model_versions:
            # Sort by version number and get the latest
            latest_version = sorted(model_versions, key=lambda x: int(x.version), reverse=True)[0]
            version = latest_version.version
            
            # Set version tags
            client.set_model_version_tag(
                name=model_name,
                version=version,
                key="dataset_version",
                value=dataset_version
            )
            client.set_model_version_tag(
                name=model_name,
                version=version,
                key="dataset_hash",
                value=dataset_info.get('dataset_hash', 'unknown')
            )
            client.set_model_version_tag(
                name=model_name,
                version=version,
                key="test_mAP",
                value=f"{test_metrics['mAP']:.4f}"
            )
            client.set_model_version_tag(
                name=model_name,
                version=version,
                key="test_precision",
                value=f"{test_metrics['precision']:.4f}"
            )
            client.set_model_version_tag(
                name=model_name,
                version=version,
                key="test_recall",
                value=f"{test_metrics['recall']:.4f}"
            )
            client.set_model_version_tag(
                name=model_name,
                version=version,
                key="augmentation_status",
                value="disabled" if not enabled_augs else "enabled"
            )
            client.set_model_version_tag(
                name=model_name,
                version=version,
                key="optimizer",
                value=config['training']['optimizer']['type']
            )
            client.set_model_version_tag(
                name=model_name,
                version=version,
                key="scheduler",
                value=config['training']['scheduler']['type']
            )
            
            # Set model version description
            client.update_model_version(
                name=model_name,
                version=version,
                description=f"Faster R-CNN {config['model']['backbone']} trained on TACO dataset "
                           f"(v{dataset_version}). "
                           f"Optimizer: {config['training']['optimizer']['type'].upper()}, "
                           f"Scheduler: {config['training']['scheduler']['type']}. "
                           f"Test mAP@0.5: {test_metrics['mAP']:.4f}, "
                           f"Precision: {test_metrics['precision']:.4f}, "
                           f"Recall: {test_metrics['recall']:.4f}. "
                           f"Trained for {epoch} epochs with {'AMP' if use_amp else 'FP32'}."
            )
            
            print(f"✓ Model registered: {model_name} (version {version})")
            print(f"  Test mAP@0.5: {test_metrics['mAP']:.4f}")
            print(f"  Precision: {test_metrics['precision']:.4f}")
            print(f"  Recall: {test_metrics['recall']:.4f}")
            print(f"  Dataset: {dataset_version} (hash: {dataset_info.get('dataset_hash', 'unknown')})")
            print(f"  Optimizer: {config['training']['optimizer']['type'].upper()}")
            print(f"  Scheduler: {config['training']['scheduler']['type']}")
            print(f"  Test mAP@0.5: {test_metrics['mAP']:.4f}")
            print(f"  Dataset: {dataset_version} (hash: {dataset_info.get('dataset_hash', 'unknown')})")
        
        print(f"{'='*60}")
        
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
