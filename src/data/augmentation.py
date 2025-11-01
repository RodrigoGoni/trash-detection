"""
Data augmentation transforms using Albumentations
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 224):
    """
    Get training data augmentation pipeline

    Args:
        image_size: Target image size

    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        # Geometric transforms
        A.Resize(image_size, image_size),
        A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=30, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                           rotate_limit=15, p=0.5),

        # Color transforms
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20,
                             sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15,
                   b_shift_limit=15, p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=0.3),

        # Weather and lighting
        A.OneOf([
            A.RandomRain(p=1.0),
            A.RandomFog(p=1.0),
            A.RandomSunFlare(p=1.0),
        ], p=0.2),

        # Noise
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.ISONoise(p=1.0),
        ], p=0.2),

        # Normalization and conversion
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224):
    """
    Get validation/test data transforms (no augmentation)

    Args:
        image_size: Target image size

    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_test_time_augmentation():
    """
    Get test-time augmentation transforms

    Returns:
        List of transform pipelines for TTA
    """
    base_transforms = [
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ]

    tta_transforms = [
        # Original
        A.Compose(base_transforms),

        # Horizontal flip
        A.Compose([A.HorizontalFlip(p=1.0)] + base_transforms),

        # Vertical flip
        A.Compose([A.VerticalFlip(p=1.0)] + base_transforms),

        # Rotate 90
        A.Compose([A.Rotate(limit=90, p=1.0)] + base_transforms),

        # Brightness adjustment
        A.Compose([A.RandomBrightnessContrast(
            brightness_limit=0.2, p=1.0)] + base_transforms),
    ]

    return tta_transforms


def get_object_detection_transforms(image_size: int = 640):
    """
    Get transforms for object detection (preserves bounding boxes)

    Args:
        image_size: Target image size

    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Blur(blur_limit=3, p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
