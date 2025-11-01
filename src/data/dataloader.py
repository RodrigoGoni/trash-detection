"""
Data loading and dataset classes for trash detection
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image


class TrashDataset(Dataset):
    """
    Custom Dataset for trash detection

    Args:
        data_dir: Path to dataset directory
        split: 'train', 'val', or 'test'
        transform: Optional transforms to apply
        target_size: Target size for images (height, width)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform=None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size

        # Load image paths and labels
        self.images, self.labels = self._load_data()

        # Get class names
        self.classes = sorted(set(self.labels))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def _load_data(self) -> Tuple[List[str], List[str]]:
        """Load image paths and labels from directory structure"""
        images = []
        labels = []

        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")

        # Assuming structure: data_dir/split/class_name/image.jpg
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                for img_path in class_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        images.append(str(img_path))
                        labels.append(class_name)

        return images, labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample"""
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        else:
            image = np.array(image)
            image = cv2.resize(image, self.target_size)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Convert label to index
        label_idx = self.class_to_idx[label]

        return image, label_idx


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform=None,
    val_transform=None,
    test_transform=None
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create dataloaders for train, validation, and test sets

    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        train_transform: Transforms for training data
        val_transform: Transforms for validation data
        test_transform: Transforms for test data

    Returns:
        train_loader, val_loader, test_loader (if exists)
    """

    # Create datasets
    train_dataset = TrashDataset(
        data_dir=data_dir,
        split='train',
        transform=train_transform
    )

    val_dataset = TrashDataset(
        data_dir=data_dir,
        split='val',
        transform=val_transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Check if test set exists
    test_loader = None
    test_path = Path(data_dir) / 'test'
    if test_path.exists():
        test_dataset = TrashDataset(
            data_dir=data_dir,
            split='test',
            transform=test_transform or val_transform
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return train_loader, val_loader, test_loader
