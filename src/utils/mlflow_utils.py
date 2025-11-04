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
    # Format: taco-detection/{model_type}/{backbone}/experiments
    experiment_name = (
        f"taco-detection/{config['model']['name'].lower()}/"
        f"{config['model']['backbone']}/experiments"
    )
    mlflow.set_experiment(experiment_name)


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
