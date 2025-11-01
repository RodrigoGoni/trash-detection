"""
MLflow experiment tracking utilities
"""

import mlflow
import mlflow.pytorch
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import torch.nn as nn


class MLflowTracker:
    """
    MLflow experiment tracking wrapper

    Args:
        tracking_uri: MLflow tracking server URI
        experiment_name: Name of the experiment
        run_name: Name of the run (optional)
    """

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "trash-detection",
        run_name: Optional[str] = None
    ):
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Set or create experiment
        mlflow.set_experiment(experiment_name)

        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run = None

    def start_run(self, run_name: Optional[str] = None):
        """Start a new MLflow run"""
        self.run = mlflow.start_run(run_name=run_name or self.run_name)
        return self.run

    def end_run(self):
        """End the current MLflow run"""
        mlflow.end_run()

    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric"""
        mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str):
        """Log an artifact (file)"""
        mlflow.log_artifact(local_path)

    def log_artifacts(self, local_dir: str):
        """Log artifacts from a directory"""
        mlflow.log_artifacts(local_dir)

    def log_model(self, model: nn.Module, artifact_path: str = "model"):
        """Log PyTorch model"""
        mlflow.pytorch.log_model(model, artifact_path)

    def log_config(self, config: Dict[str, Any], filename: str = "config.yaml"):
        """Log configuration file"""
        config_path = Path(filename)
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        mlflow.log_artifact(str(config_path))
        config_path.unlink()  # Remove temp file

    def set_tags(self, tags: Dict[str, str]):
        """Set tags for the run"""
        mlflow.set_tags(tags)

    def log_figure(self, figure, artifact_file: str):
        """Log matplotlib figure"""
        figure.savefig(artifact_file)
        mlflow.log_artifact(artifact_file)
        Path(artifact_file).unlink()  # Remove temp file

    @staticmethod
    def load_model(model_uri: str):
        """Load a model from MLflow"""
        return mlflow.pytorch.load_model(model_uri)


def setup_mlflow_tracking(config: Dict[str, Any]) -> MLflowTracker:
    """
    Setup MLflow tracking from config

    Args:
        config: Configuration dictionary

    Returns:
        MLflowTracker instance
    """
    mlflow_config = config.get('mlflow', {})

    tracker = MLflowTracker(
        tracking_uri=mlflow_config.get(
            'tracking_uri', 'http://localhost:5000'),
        experiment_name=mlflow_config.get(
            'experiment_name', 'trash-detection'),
        run_name=mlflow_config.get('run_name', None)
    )

    return tracker
