"""
MLflow utilities for experiment tracking and model registry.
Provides functions to setup MLflow, log configurations, and manage datasets.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import mlflow
import mlflow.pytorch
import pandas as pd


def setup_mlflow(config: dict):
    """
    Setup MLflow experiment tracking with hierarchical structure.
    
    Args:
        config: Configuration dictionary with MLflow settings
    """
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    
    # Create hierarchical experiment name for better organization
    # Format: taco-detection/{model_type}/{backbone or model_size}/experiments
    model_name = config['model']['name'].lower()
    
    # For YOLO models, use model_size; for Faster R-CNN, use backbone
    if 'yolo' in model_name:
        model_variant = config['model'].get('model_size', 's')
        experiment_name = f"taco-detection/{model_name}/{model_name}{model_variant}/experiments"
    else:
        backbone = config['model'].get('backbone', 'unknown')
        experiment_name = f"taco-detection/{model_name}/{backbone}/experiments"
    
    # Handle deleted experiments
    try:
        mlflow.set_experiment(experiment_name)
    except mlflow.exceptions.MlflowException as e:
        if "deleted experiment" in str(e).lower():
            print(f"⚠️  Experiment '{experiment_name}' was previously deleted.")
            print("   Attempting to restore or create new experiment...")
            
            # Try to restore the experiment
            client = mlflow.tracking.MlflowClient()
            try:
                # Get experiment by name (even if deleted)
                experiment = client.get_experiment_by_name(experiment_name)
                if experiment and experiment.lifecycle_stage == "deleted":
                    # Restore the experiment
                    client.restore_experiment(experiment.experiment_id)
                    print(f"✓  Restored experiment: {experiment_name}")
                    mlflow.set_experiment(experiment_name)
                else:
                    raise e
            except Exception as restore_error:
                print(f"⚠️  Could not restore experiment: {restore_error}")
                print("   Please run one of the following commands:")
                print(f"   1. Restore: mlflow experiments restore --experiment-name '{experiment_name}'")
                print(f"   2. Delete permanently and create new: mlflow gc --backend-store-uri mlruns")
                raise
        else:
            raise


def get_dataset_info(data_dir: Path) -> dict:
    """
    Get dataset information and generate version hash.
    
    Args:
        data_dir: Path to the dataset directory
        
    Returns:
        Dictionary with dataset information including paths, counts, and hash
    """
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


def log_system_tags(system_info: dict):
    """
    Log system information as MLflow tags.
    
    Args:
        system_info: Dictionary with system information
    """
    mlflow.set_tag("system.hostname", system_info['hostname'])
    mlflow.set_tag("system.platform", system_info['platform'])
    mlflow.set_tag("system.python_version", system_info['python_version'])
    mlflow.set_tag("system.pytorch_version", system_info['pytorch_version'])
    mlflow.set_tag("system.cuda_available", system_info['cuda_available'])
    
    if system_info['cuda_available']:
        mlflow.set_tag("system.gpu_name", system_info['gpu_name'])
        mlflow.set_tag("system.cuda_version", system_info['cuda_version'])
        mlflow.set_tag("system.gpu_memory_gb", system_info['gpu_memory_gb'])


def log_experiment_tags(config: dict):
    """
    Log experiment-specific tags from configuration.
    
    Args:
        config: Configuration dictionary with experiment settings
    """
    if 'tags' in config.get('experiment', {}):
        for tag in config['experiment']['tags']:
            mlflow.set_tag(f"experiment.{tag}", "true")
    
    if 'description' in config.get('experiment', {}):
        mlflow.set_tag("mlflow.note.content", config['experiment']['description'])


def log_dataset_tags(dataset_info: dict, dataset_version: str):
    """
    Log dataset information as MLflow tags.
    
    Args:
        dataset_info: Dictionary with dataset information
        dataset_version: Dataset version string
    """
    mlflow.set_tag("dataset.version", dataset_version)
    mlflow.set_tag("dataset.hash", dataset_info.get('dataset_hash', 'unknown'))
    mlflow.set_tag("dataset.path", dataset_info['path'])
    
    if 'train_images' in dataset_info:
        mlflow.set_tag("dataset.train_images", dataset_info['train_images'])
        mlflow.set_tag("dataset.val_images", dataset_info['val_images'])
        mlflow.set_tag("dataset.test_images", dataset_info['test_images'])


def register_dataset_in_mlflow(dataset_info: dict, dataset_version: str, config: dict):
    """
    Register dataset in MLflow (appears in Datasets tab).
    
    Args:
        dataset_info: Dictionary with dataset information
        dataset_version: Dataset version string
        config: Configuration dictionary
    """
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


def log_augmentation_tags(augmentation_config: dict):
    """
    Log augmentation configuration as tags.
    
    Args:
        augmentation_config: Dictionary with augmentation settings
        
    Returns:
        List of enabled augmentations
    """
    enabled_augs = [k for k, v in augmentation_config.items() if k.endswith('_p') and v > 0]
    
    if enabled_augs:
        mlflow.set_tag("augmentation.enabled", ", ".join(enabled_augs))
        mlflow.set_tag("augmentation.status", "enabled")
    else:
        mlflow.set_tag("augmentation.status", "disabled")
    
    return enabled_augs


def log_config_parameters(config: dict):
    """
    Log training configuration as MLflow parameters.
    
    Args:
        config: Configuration dictionary
    """
    # Model parameters
    mlflow.log_params({
        "model.num_classes": config['model']['num_classes'],
        "model.pretrained": config['model'].get('pretrained', True),
    })
    
    # Training parameters
    mlflow.log_params({
        "train.epochs": config['training']['num_epochs'],
        "train.learning_rate": config['training']['optimizer']['lr'],
        "train.momentum": config['training']['optimizer'].get('momentum', 0.9),
        "train.weight_decay": config['training']['optimizer'].get('weight_decay', 0.0005),
    })
    
    # Scheduler parameters
    mlflow.log_params({
        "scheduler.type": config['training']['scheduler']['type'],
        "scheduler.step_size": config['training']['scheduler'].get('step_size', 0),
        "scheduler.gamma": config['training']['scheduler'].get('gamma', 0),
    })
    
    # Log preprocessing and augmentation as single parameters
    mlflow.log_param("preprocessing_config", str(config['data']['preprocessing']))
    mlflow.log_param("augmentation_config", str(config['data']['augmentation']))


def log_training_tags(config: dict):
    """
    Log training-related tags for benchmarking.
    
    Args:
        config: Configuration dictionary
    """
    mlflow.set_tags({
        # Model identification
        "model.type": config['model']['name'].lower(),
        "model.backbone": config['model']['backbone'],
        "model.variant": "baseline" if all(
            v == 0.0 for v in config['data']['augmentation'].values()
        ) else "augmented",
        
        # Dataset information
        "data.version": config['data'].get('dataset_version', 'unknown'),
        "data.size": config['data']['img_size'],
        
        # Training type
        "training.hardware": "gpu" if torch.cuda.is_available() else "cpu",
        "training.batch_size": config['data']['batch_size'],
        "training.optimizer": config['training']['optimizer']['type'],
        
        # Experiment categorization
        "experiment.type": "benchmark",
        "experiment.phase": "development"
    })


def log_system_parameters(system_info: dict):
    """
    Log system information as MLflow parameters.
    
    Args:
        system_info: Dictionary with system information
    """
    for key, value in system_info.items():
        mlflow.log_param(f"sys_{key}", value)


def log_dataset_parameters(dataset_info: dict):
    """
    Log dataset information as MLflow parameters.
    
    Args:
        dataset_info: Dictionary with dataset information
    """
    for key, value in dataset_info.items():
        if key not in ['path']:  # path is already in tags
            mlflow.log_param(f"data_{key}", value)


def register_model_in_mlflow(model, config: dict, test_metrics: dict, 
                             dataset_version: str, dataset_info: dict,
                             enabled_augs: list, epoch: int, use_amp: bool) -> str:
    """
    Register trained model in MLflow Model Registry.
    
    Args:
        model: Trained PyTorch model
        config: Configuration dictionary
        test_metrics: Dictionary with test set metrics
        dataset_version: Dataset version string
        dataset_info: Dictionary with dataset information
        enabled_augs: List of enabled augmentations
        epoch: Final training epoch
        use_amp: Whether AMP was used
        
    Returns:
        Model version number as string
    """
    model_name = f"taco-{config['model']['name'].lower()}-{config['model']['backbone']}"
    
    # Log model to MLflow
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name=model_name,
    )
    
    # Get model version and set tags
    client = mlflow.tracking.MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    
    if not model_versions:
        return None
    
    # Sort by version number and get the latest
    latest_version = sorted(model_versions, key=lambda x: int(x.version), reverse=True)[0]
    version = latest_version.version
    
    # Set version tags
    version_tags = {
        "dataset_version": dataset_version,
        "dataset_hash": dataset_info.get('dataset_hash', 'unknown'),
        "test_mAP": f"{test_metrics['mAP']:.4f}",
        "test_precision": f"{test_metrics['precision']:.4f}",
        "test_recall": f"{test_metrics['recall']:.4f}",
        "augmentation_status": "disabled" if not enabled_augs else "enabled",
        "optimizer": config['training']['optimizer']['type'],
        "scheduler": config['training']['scheduler']['type'],
    }
    
    for key, value in version_tags.items():
        client.set_model_version_tag(
            name=model_name,
            version=version,
            key=key,
            value=str(value)
        )
    
    # Set model version description
    client.update_model_version(
        name=model_name,
        version=version,
        description=(
            f"Faster R-CNN {config['model']['backbone']} trained on TACO dataset "
            f"(v{dataset_version}). "
            f"Optimizer: {config['training']['optimizer']['type'].upper()}, "
            f"Scheduler: {config['training']['scheduler']['type']}. "
            f"Test mAP@0.5: {test_metrics['mAP']:.4f}, "
            f"Precision: {test_metrics['precision']:.4f}, "
            f"Recall: {test_metrics['recall']:.4f}. "
            f"Trained for {epoch} epochs with {'AMP' if use_amp else 'FP32'}."
        )
    )
    
    return version


def log_epoch_metrics(train_loss: float, val_loss: float, learning_rate: float, step: int):
    """
    Log basic training metrics for an epoch.
    
    Args:
        train_loss: Training loss value
        val_loss: Validation loss value
        learning_rate: Current learning rate
        step: Epoch number
    """
    mlflow.log_metrics({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": learning_rate
    }, step=step)


def log_validation_detection_metrics(metrics: dict, step: int, prefix: str = "val"):
    """
    Log validation detection metrics (mAP, precision, recall, F1).
    
    Args:
        metrics: Dictionary with detection metrics
        step: Epoch number
        prefix: Metric prefix (default: "val")
    """
    mlflow.log_metrics({
        f"{prefix}_mAP": metrics['mAP'],
        f"{prefix}_precision": metrics['precision'],
        f"{prefix}_recall": metrics['recall'],
        f"{prefix}_f1_score": metrics['f1_score']
    }, step=step)


def log_test_metrics(test_metrics: dict):
    """
    Log comprehensive test set metrics including benchmark metrics.
    
    Args:
        test_metrics: Dictionary with all test metrics
    """
    # Log individual test metrics
    mlflow.log_metrics({
        "test_mAP": test_metrics['mAP'],
        "test_precision": test_metrics['precision'],
        "test_recall": test_metrics['recall'],
        "test_f1_score": test_metrics['f1_score'],
        "test_true_positives": test_metrics['true_positives'],
        "test_false_positives": test_metrics['false_positives'],
        "test_false_negatives": test_metrics['false_negatives']
    })
    
    # Log benchmark metrics (for easy comparison)
    mlflow.log_metrics({
        "benchmark.map50": test_metrics['mAP'],
        "benchmark.f1": test_metrics['f1_score'],
        "benchmark.precision": test_metrics['precision'],
        "benchmark.recall": test_metrics['recall'],
        "benchmark.true_positives": test_metrics['true_positives'],
        "benchmark.false_positives": test_metrics['false_positives'],
        "benchmark.false_negatives": test_metrics['false_negatives']
    })


def log_final_training_metrics(best_val_loss: float, total_epochs: int):
    """
    Log final training summary metrics.
    
    Args:
        best_val_loss: Best validation loss achieved
        total_epochs: Total number of epochs trained
    """
    mlflow.log_metrics({
        "best_val_loss": best_val_loss,
        "total_epochs": total_epochs
    })


# ============================================================================
# YOLO-SPECIFIC MLFLOW FUNCTIONS
# ============================================================================

def log_yolo_training_config(config: dict, system_info: dict, dataset_info: dict, 
                              dataset_version: str, device: str):
    """
    Log YOLO-specific training configuration to MLflow.
    
    Args:
        config: Configuration dictionary
        system_info: System information dictionary
        dataset_info: Dataset information dictionary
        dataset_version: Dataset version string
        device: Device used for training
    """
    # Enable system metrics logging
    mlflow.enable_system_metrics_logging()
    
    # Log model configuration
    model_name = config['model']['name']
    model_size = config['model']['model_size']
    
    # Check if using new yolo_training config structure
    yolo_config = config.get('yolo_training', {})
    training_config = config.get('training', {})
    
    # Get optimizer info from yolo_training or training section
    if yolo_config:
        optimizer_type = yolo_config.get('optimizer', 'auto')
        learning_rate = yolo_config.get('lr0', 0.01)
        scheduler_type = 'cosine' if yolo_config.get('cos_lr', False) else 'none'
        patience = yolo_config.get('patience', 100)
    else:
        optimizer_config = training_config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'auto')
        learning_rate = optimizer_config.get('lr', 0.01)
        scheduler_config = training_config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'none')
        patience = training_config.get('patience', 100)
    
    mlflow.log_params({
        'model_name': model_name,
        'model_size': model_size,
        'num_classes': config['model']['num_classes'],
        'epochs': training_config.get('num_epochs', 100),
        'batch_size': config['data']['batch_size'],
        'optimizer': optimizer_type,
        'learning_rate': learning_rate,
        'scheduler': scheduler_type,
        'image_size': config['data'].get('img_size', 640),
        'patience': patience
    })
    
    # Log YOLO-specific parameters if available
    if yolo_config:
        # Log key YOLO parameters
        yolo_params = {
            'yolo.dropout': yolo_config.get('dropout', 0.0),
            'yolo.weight_decay': yolo_config.get('weight_decay', 0.0005),
            'yolo.momentum': yolo_config.get('momentum', 0.937),
            'yolo.warmup_epochs': yolo_config.get('warmup_epochs', 3.0),
            'yolo.cos_lr': yolo_config.get('cos_lr', False),
            'yolo.lrf': yolo_config.get('lrf', 0.01),
            'yolo.box_weight': yolo_config.get('box', 7.5),
            'yolo.cls_weight': yolo_config.get('cls', 0.5),
            'yolo.dfl_weight': yolo_config.get('dfl', 1.5),
            'yolo.multi_scale': yolo_config.get('multi_scale', False),
            'yolo.rect': yolo_config.get('rect', False),
            'yolo.compile': yolo_config.get('compile', False),
            'yolo.val': yolo_config.get('val', True),
            'yolo.plots': yolo_config.get('plots', True),
            'yolo.save': yolo_config.get('save', True),
        }
        
        # Log classes parameter if specified
        classes_param = yolo_config.get('classes')
        if classes_param is not None:
            mlflow.log_param('yolo.classes', str(classes_param))
        
        mlflow.log_params(yolo_params)
        
        # Log augmentation parameters
        if 'augmentation' in yolo_config:
            aug_config = yolo_config['augmentation']
            aug_params = {
                f'yolo.aug.{k}': v 
                for k, v in aug_config.items()
            }
            mlflow.log_params(aug_params)
    else:
        # Log optimizer parameters (old structure)
        optimizer_config = training_config.get('optimizer', {})
        if optimizer_config.get('type', '').lower() == 'adamw':
            mlflow.log_param('weight_decay', optimizer_config.get('weight_decay', 0.0005))
        elif optimizer_config.get('type', '').lower() == 'sgd':
            mlflow.log_param('momentum', optimizer_config.get('momentum', 0.937))
            mlflow.log_param('weight_decay', optimizer_config.get('weight_decay', 0.0005))
    
    # Log loss configuration if present (old structure)
    if 'loss' in training_config:
        loss_config = training_config['loss']
        mlflow.set_tag("loss.type", loss_config.get('type', 'yolo_default'))
        mlflow.set_tag("loss.use_class_weights", loss_config.get('use_class_weights', False))
        mlflow.set_tag("loss.class_weight_method", loss_config.get('class_weight_method', 'none'))
        
        if 'beta' in loss_config:
            mlflow.set_tag("loss.beta", loss_config.get('beta'))
        if 'gamma' in loss_config:
            mlflow.set_tag("loss.gamma", loss_config.get('gamma'))
    
    # Log system information
    mlflow.log_params({
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'python_version': system_info.get('python_version', 'unknown'),
        'hostname': system_info.get('hostname', 'unknown')
    })
    
    # Set tags
    mlflow.set_tag("model.type", "YOLOv11")
    mlflow.set_tag("model.variant", model_size)
    mlflow.set_tag("training.framework", "ultralytics")
    
    # Register dataset
    register_dataset_in_mlflow(dataset_info, dataset_version, config)


def log_yolo_results(results_dir: Path, config: dict) -> tuple[dict, dict]:
    """
    Parse and log YOLO training results to MLflow.
    
    Args:
        results_dir: Path to YOLO results directory
        config: Configuration dictionary
        
    Returns:
        Tuple of (final_metrics, best_metrics) dictionaries
    """
    final_metrics = {}
    best_metrics = {}
    
    # Log results CSV if available
    results_csv = results_dir / 'results.csv'
    if not results_csv.exists():
        print(f"⚠ Warning: Results CSV not found at {results_csv}")
        return final_metrics, best_metrics
    
    import pandas as pd
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
    # Log metrics for each epoch
    for idx, row in df.iterrows():
        epoch = idx + 1
        
        # Log training losses
        if 'train/box_loss' in row:
            mlflow.log_metric('train_box_loss', float(row['train/box_loss']), step=epoch)
        if 'train/cls_loss' in row:
            mlflow.log_metric('train_cls_loss', float(row['train/cls_loss']), step=epoch)
        if 'train/dfl_loss' in row:
            mlflow.log_metric('train_dfl_loss', float(row['train/dfl_loss']), step=epoch)
        
        # Calculate and log total train loss
        if all(k in row for k in ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']):
            total_train_loss = float(row['train/box_loss']) + float(row['train/cls_loss']) + float(row['train/dfl_loss'])
            mlflow.log_metric('train_loss', total_train_loss, step=epoch)
        
        # Log validation losses
        if 'val/box_loss' in row:
            mlflow.log_metric('val_box_loss', float(row['val/box_loss']), step=epoch)
        if 'val/cls_loss' in row:
            mlflow.log_metric('val_cls_loss', float(row['val/cls_loss']), step=epoch)
        if 'val/dfl_loss' in row:
            mlflow.log_metric('val_dfl_loss', float(row['val/dfl_loss']), step=epoch)
        
        # Calculate and log total validation loss
        if all(k in row for k in ['val/box_loss', 'val/cls_loss', 'val/dfl_loss']):
            total_val_loss = float(row['val/box_loss']) + float(row['val/cls_loss']) + float(row['val/dfl_loss'])
            mlflow.log_metric('val_loss', total_val_loss, step=epoch)
        
        # Log validation metrics
        if 'metrics/mAP50(B)' in row:
            mlflow.log_metric('val_mAP50', float(row['metrics/mAP50(B)']), step=epoch)
        if 'metrics/mAP50-95(B)' in row:
            mlflow.log_metric('val_mAP50_95', float(row['metrics/mAP50-95(B)']), step=epoch)
        if 'metrics/precision(B)' in row:
            mlflow.log_metric('val_precision', float(row['metrics/precision(B)']), step=epoch)
        if 'metrics/recall(B)' in row:
            mlflow.log_metric('val_recall', float(row['metrics/recall(B)']), step=epoch)
    
    print(f"✓ Logged {len(df)} epochs of training metrics from results.csv")
    
    # Extract final metrics (last epoch)
    if len(df) > 0:
        last_row = df.iloc[-1]
        
        if 'metrics/mAP50(B)' in last_row:
            final_metrics['final_mAP50'] = float(last_row['metrics/mAP50(B)'])
        if 'metrics/mAP50-95(B)' in last_row:
            final_metrics['final_mAP50_95'] = float(last_row['metrics/mAP50-95(B)'])
        if 'metrics/precision(B)' in last_row:
            final_metrics['final_precision'] = float(last_row['metrics/precision(B)'])
        if 'metrics/recall(B)' in last_row:
            final_metrics['final_recall'] = float(last_row['metrics/recall(B)'])
        
        # Extract best metrics (across all epochs)
        if 'val/box_loss' in df.columns and 'val/cls_loss' in df.columns and 'val/dfl_loss' in df.columns:
            df['total_val_loss'] = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']
            best_val_loss = float(df['total_val_loss'].min())
            best_val_loss_epoch = int(df['total_val_loss'].idxmin()) + 1
            best_metrics['best_val_loss'] = best_val_loss
            best_metrics['best_val_loss_epoch'] = best_val_loss_epoch
            mlflow.log_metric('best_val_loss', best_val_loss)
            mlflow.set_tag('best_val_loss_epoch', best_val_loss_epoch)
        
        if 'metrics/mAP50(B)' in df.columns:
            best_mAP50 = float(df['metrics/mAP50(B)'].max())
            best_mAP50_epoch = int(df['metrics/mAP50(B)'].idxmax()) + 1
            best_metrics['best_mAP50'] = best_mAP50
            best_metrics['best_mAP50_epoch'] = best_mAP50_epoch
            mlflow.log_metric('best_mAP50', best_mAP50)
            mlflow.set_tag('best_mAP50_epoch', best_mAP50_epoch)
        
        if 'metrics/mAP50-95(B)' in df.columns:
            best_mAP50_95 = float(df['metrics/mAP50-95(B)'].max())
            best_mAP50_95_epoch = int(df['metrics/mAP50-95(B)'].idxmax()) + 1
            best_metrics['best_mAP50_95'] = best_mAP50_95
            best_metrics['best_mAP50_95_epoch'] = best_mAP50_95_epoch
            mlflow.log_metric('best_mAP50_95', best_mAP50_95)
            mlflow.set_tag('best_mAP50_95_epoch', best_mAP50_95_epoch)
        
        # Log final metrics
        for key, value in final_metrics.items():
            mlflow.log_metric(key, value)
    
    return final_metrics, best_metrics


def log_yolo_artifacts(results_dir: Path):
    """
    Log YOLO training artifacts (weights, plots, etc.) to MLflow.
    
    Args:
        results_dir: Path to YOLO results directory
    """
    # Log model weights
    best_weights = results_dir / 'weights' / 'best.pt'
    if best_weights.exists():
        mlflow.log_artifact(str(best_weights), artifact_path="model")
        print(f"✓ Logged best model weights: {best_weights.name}")
    
    last_weights = results_dir / 'weights' / 'last.pt'
    if last_weights.exists():
        mlflow.log_artifact(str(last_weights), artifact_path="model")
        print(f"✓ Logged last model weights: {last_weights.name}")
    
    # Log training plots
    plot_count = 0
    for plot_file in results_dir.glob('*.png'):
        mlflow.log_artifact(str(plot_file), artifact_path="plots")
        plot_count += 1
    
    if plot_count > 0:
        print(f"✓ Logged {plot_count} training plots")
    
    # Log confusion matrix if available
    confusion_matrix = results_dir / 'confusion_matrix.png'
    if confusion_matrix.exists():
        print(f"✓ Logged confusion matrix")


def log_yolo_training_summary(results_dir: Path, config: dict, 
                               final_metrics: dict, best_metrics: dict):
    """
    Log final YOLO training summary tags to MLflow.
    
    Args:
        results_dir: Path to YOLO results directory
        config: Configuration dictionary
        final_metrics: Final metrics dictionary
        best_metrics: Best metrics dictionary
    """
    model_size = config['model']['model_size']
    patience = config['training'].get('patience', 100)
    
    # Model size to parameters mapping (approximate)
    model_params = {
        'n': '2.6M',
        's': '9.4M',
        'm': '20.5M',
        'l': '25.3M',
        'x': '68.2M'
    }
    
    mlflow.set_tag("model.parameters", model_params.get(model_size, 'unknown'))
    mlflow.set_tag("training.patience", patience)
    
    # Log total epochs trained
    results_csv = results_dir / 'results.csv'
    if results_csv.exists():
        import pandas as pd
        df = pd.read_csv(results_csv)
        total_epochs = len(df)
        mlflow.set_tag("training.total_epochs", total_epochs)
        mlflow.log_metric("total_epochs", total_epochs)
