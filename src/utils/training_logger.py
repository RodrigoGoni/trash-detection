"""
Training logger utilities for clean and organized console output.
Provides formatted logging functions to reduce console clutter.
"""

from typing import Dict, Any


class TrainingLogger:
    """Manages training output with organized formatting."""
    
    @staticmethod
    def print_section(title: str, width: int = 60):
        """Print a section header."""
        print(f"\n{'='*width}")
        print(f"{title}")
        print(f"{'='*width}")
    
    @staticmethod
    def print_subsection(title: str, width: int = 60):
        """Print a subsection header."""
        print(f"\n{'-'*width}")
        print(f"{title}")
        print(f"{'-'*width}")
    
    @staticmethod
    def print_setup_info(config: dict, system_info: dict, dataset_info: dict):
        """Print compact setup information."""
        TrainingLogger.print_section("TRAINING SETUP")
        
        # Model & Data
        print(f"Model:      {config['model']['name']} ({config['model']['backbone']})")
        print(f"Classes:    {config['model']['num_classes']}")
        print(f"Dataset:    {config['data'].get('dataset_version', 'unknown')}")
        
        # Training config
        print(f"\nEpochs:     {config['training']['num_epochs']}")
        print(f"Batch Size: {config['data']['batch_size']}")
        print(f"Optimizer:  {config['training']['optimizer']['type'].upper()} (lr={config['training']['optimizer']['lr']})")
        print(f"Scheduler:  {config['training']['scheduler']['type']}")
        
        # Hardware
        gpu_name = system_info.get('gpu_name', 'CPU only')
        print(f"\nHardware:   {gpu_name}")
        if system_info.get('cuda_available'):
            print(f"GPU Memory: {system_info.get('gpu_memory_gb', 'N/A')} GB")
        
        # Dataset
        if 'train_images' in dataset_info:
            print(f"\nData Split: Train={dataset_info['train_images']}, "
                  f"Val={dataset_info['val_images']}, "
                  f"Test={dataset_info['test_images']}")
    
    @staticmethod
    def print_epoch_header(epoch: int, total_epochs: int):
        """Print compact epoch header."""
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{total_epochs}")
        print(f"{'='*60}")
    
    @staticmethod
    def print_epoch_summary(epoch: int, train_loss: float, val_loss: float, 
                           learning_rate: float, best_model: bool = False,
                           patience_info: str = None):
        """Print compact epoch summary."""
        status = "✓ BEST" if best_model else ""
        print(f"\nEpoch {epoch}: Loss(train={train_loss:.4f}, val={val_loss:.4f}) "
              f"LR={learning_rate:.2e} {status}")
        
        if patience_info:
            print(f"  {patience_info}")
    
    @staticmethod
    def print_validation_metrics(epoch: int, metrics: dict, threshold: float = 0.05):
        """Print validation metrics in compact format."""
        print(f"\n  Validation [threshold={threshold}]: "
              f"mAP={metrics['mAP']:.3f} | "
              f"P={metrics['precision']:.3f} | "
              f"R={metrics['recall']:.3f} | "
              f"F1={metrics['f1_score']:.3f}")
    
    @staticmethod
    def print_test_metrics(metrics: dict, threshold: float = 0.5):
        """Print detailed test metrics."""
        TrainingLogger.print_section("FINAL TEST RESULTS")
        print(f"Threshold:    {threshold}")
        print(f"\nMetrics:")
        print(f"  mAP@0.5     {metrics['mAP']:.4f}")
        print(f"  Precision   {metrics['precision']:.4f}")
        print(f"  Recall      {metrics['recall']:.4f}")
        print(f"  F1-Score    {metrics['f1_score']:.4f}")
        print(f"\nDetections:")
        print(f"  TP: {metrics['true_positives']} | "
              f"FP: {metrics['false_positives']} | "
              f"FN: {metrics['false_negatives']}")
    
    @staticmethod
    def print_training_complete(epoch: int, best_val_loss: float, 
                               test_metrics: dict, paths: dict):
        """Print final training summary."""
        TrainingLogger.print_section("TRAINING COMPLETED")
        
        print(f"\nSummary:")
        print(f"  Total Epochs:     {epoch}")
        print(f"  Best Val Loss:    {best_val_loss:.4f}")
        print(f"  Test mAP@0.5:     {test_metrics['mAP']:.4f}")
        print(f"  Test Precision:   {test_metrics['precision']:.4f}")
        print(f"  Test Recall:      {test_metrics['recall']:.4f}")
        
        print(f"\nArtifacts:")
        print(f"  Model:      {paths.get('model', 'N/A')}")
        print(f"  Metrics:    {paths.get('metrics', 'N/A')}")
        
        print(f"\nMLflow UI:  http://localhost:5000")
        print(f"{'='*60}\n")
    
    @staticmethod
    def print_mlflow_info(run_id: str, experiment_name: str, dataset_version: str, 
                         dataset_hash: str):
        """Print MLflow tracking info."""
        TrainingLogger.print_subsection("MLflow Tracking")
        print(f"Run ID:     {run_id}")
        print(f"Experiment: {experiment_name}")
        print(f"Dataset:    {dataset_version} (hash: {dataset_hash})")
    
    @staticmethod
    def print_model_registration(model_name: str, version: str, metrics: dict,
                                dataset_version: str, optimizer: str, scheduler: str):
        """Print model registration info."""
        TrainingLogger.print_subsection("Model Registered")
        print(f"Name:       {model_name}")
        print(f"Version:    {version}")
        print(f"Test mAP:   {metrics['mAP']:.4f}")
        print(f"Dataset:    {dataset_version}")
        print(f"Optimizer:  {optimizer}")
        print(f"Scheduler:  {scheduler}")
    
    @staticmethod
    def print_batch_progress(epoch: int, batch_idx: int, total_batches: int, 
                            loss: float, frequency: int = 50):
        """Print batch progress (less frequently)."""
        if (batch_idx + 1) % frequency == 0:
            progress = (batch_idx + 1) / total_batches * 100
            print(f"  [{progress:5.1f}%] Batch {batch_idx+1}/{total_batches} | Loss: {loss:.4f}")
    
    @staticmethod
    def print_dataloader_info(train_batches: int, val_batches: int, test_batches: int):
        """Print dataloader information."""
        print(f"Dataloaders: Train={train_batches}, Val={val_batches}, Test={test_batches} batches")


# Convenience functions for backward compatibility
def log_section(title: str):
    """Print a section header."""
    TrainingLogger.print_section(title)


def log_info(message: str):
    """Print an info message."""
    print(f"  {message}")
