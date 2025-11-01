"""
DVC (Data Version Control) utilities
"""

import subprocess
from pathlib import Path
from typing import List, Optional


class DVCManager:
    """
    DVC management utilities

    Args:
        repo_path: Path to the repository root
    """

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)

    def init(self):
        """Initialize DVC in the repository"""
        subprocess.run(["dvc", "init"], cwd=self.repo_path, check=True)

    def add(self, path: str):
        """
        Add a file or directory to DVC tracking

        Args:
            path: Path to file or directory
        """
        subprocess.run(["dvc", "add", path], cwd=self.repo_path, check=True)

    def push(self, remote: Optional[str] = None):
        """
        Push tracked files to remote storage

        Args:
            remote: Remote name (optional)
        """
        cmd = ["dvc", "push"]
        if remote:
            cmd.extend(["-r", remote])

        subprocess.run(cmd, cwd=self.repo_path, check=True)

    def pull(self, remote: Optional[str] = None):
        """
        Pull tracked files from remote storage

        Args:
            remote: Remote name (optional)
        """
        cmd = ["dvc", "pull"]
        if remote:
            cmd.extend(["-r", remote])

        subprocess.run(cmd, cwd=self.repo_path, check=True)

    def checkout(self, target: Optional[str] = None):
        """
        Checkout data files

        Args:
            target: Specific target to checkout (optional)
        """
        cmd = ["dvc", "checkout"]
        if target:
            cmd.append(target)

        subprocess.run(cmd, cwd=self.repo_path, check=True)

    def add_remote(self, name: str, url: str, default: bool = False):
        """
        Add a DVC remote

        Args:
            name: Remote name
            url: Remote URL
            default: Set as default remote
        """
        subprocess.run(
            ["dvc", "remote", "add", name, url],
            cwd=self.repo_path,
            check=True
        )

        if default:
            subprocess.run(
                ["dvc", "remote", "default", name],
                cwd=self.repo_path,
                check=True
            )

    def run_pipeline(self, pipeline_file: str = "dvc.yaml"):
        """
        Run DVC pipeline

        Args:
            pipeline_file: Path to pipeline file
        """
        subprocess.run(
            ["dvc", "repro", pipeline_file],
            cwd=self.repo_path,
            check=True
        )

    def status(self):
        """Check DVC status"""
        result = subprocess.run(
            ["dvc", "status"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        return result.stdout


def create_dvc_pipeline(config_path: str = "dvc.yaml"):
    """
    Create a DVC pipeline configuration

    This is a template that should be customized for your project
    """
    pipeline_config = """
stages:
  prepare_data:
    cmd: python scripts/prepare_data.py
    deps:
      - data/raw
      - scripts/prepare_data.py
    outs:
      - data/processed
    params:
      - train_config.yaml:
          - data.image_size
          - data.train_split

  train:
    cmd: python scripts/train_model.py
    deps:
      - data/processed
      - src/models
      - scripts/train_model.py
    outs:
      - models/checkpoints
    params:
      - train_config.yaml:
          - training
          - model
    metrics:
      - experiments/metrics.json:
          cache: false

  evaluate:
    cmd: python scripts/evaluate_model.py
    deps:
      - models/checkpoints
      - data/processed
      - scripts/evaluate_model.py
    metrics:
      - experiments/evaluation_metrics.json:
          cache: false
    plots:
      - experiments/confusion_matrix.png:
          cache: false
"""

    with open(config_path, 'w') as f:
        f.write(pipeline_config)

    print(f"DVC pipeline configuration created at {config_path}")
