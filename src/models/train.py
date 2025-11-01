"""
Training utilities and trainer class
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
from tqdm import tqdm
import numpy as np
from pathlib import Path

# MLflow for experiment tracking
import mlflow
import mlflow.pytorch


class Trainer:
    """
    Training manager for deep learning models

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        config: Training configuration dict
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str = 'cuda',
        config: Optional[Dict] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config or {}

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        # Scheduler (optional)
        self.scheduler = None
        if 'scheduler' in self.config:
            self.scheduler = self._get_scheduler()

        # Early stopping
        self.patience = self.config.get('patience', 10)
        self.patience_counter = 0

    def _get_scheduler(self):
        """Get learning rate scheduler"""
        scheduler_type = self.config['scheduler']['type']
        scheduler_params = self.config['scheduler'].get('params', {})

        if scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, **scheduler_params)
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **scheduler_params)
        elif scheduler_type == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **scheduler_params)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader,
                    desc=f'Epoch {self.current_epoch + 1} [Train]')

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return {'loss': epoch_loss, 'acc': epoch_acc}

    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader,
                        desc=f'Epoch {self.current_epoch + 1} [Val]')

            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update progress bar
                pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        return {'loss': epoch_loss, 'acc': epoch_acc}

    def train(self, num_epochs: int, save_dir: str = './models/checkpoints'):
        """
        Train the model for multiple epochs

        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])

            # Log to MLflow
            mlflow.log_metrics({
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['acc'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['acc'],
            }, step=epoch)

            # Learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # Save best model
            if val_metrics['acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['acc']
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(save_dir / 'best_model.pth')
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    save_dir / f'checkpoint_epoch_{epoch + 1}.pth')

        print(
            f"Training completed. Best validation accuracy: {self.best_val_acc:.2f}%")

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']


def get_optimizer(model: nn.Module, config: Dict) -> optim.Optimizer:
    """Get optimizer from config"""
    optimizer_type = config.get('type', 'adam')
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 0.0)

    if optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        momentum = config.get('momentum', 0.9)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def get_criterion(config: Dict) -> nn.Module:
    """Get loss function from config"""
    criterion_type = config.get('type', 'cross_entropy')

    if criterion_type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif criterion_type == 'bce':
        return nn.BCELoss()
    elif criterion_type == 'bce_with_logits':
        return nn.BCEWithLogitsLoss()
    elif criterion_type == 'focal':
        # Custom focal loss would go here
        raise NotImplementedError("Focal loss not implemented yet")
    else:
        raise ValueError(f"Unknown criterion: {criterion_type}")
