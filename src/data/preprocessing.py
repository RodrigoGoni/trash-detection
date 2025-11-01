"""
Object Detection Preprocessing Pipeline for TACO Dataset

This module provides preprocessing utilities specifically designed for object detection
tasks using the TACO dataset. It handles variable image sizes, quality issues, and
correctly transforms bounding boxes using the Albumentations library.

Key Features:
- Letterboxing (resize with aspect ratio preservation + padding)
- Data augmentation with bbox-aware transformations
- Photometric adjustments for variable lighting/quality
- Proper bbox format conversion and validation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(
    mode: str = 'train',
    img_size: int = 640,
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> A.Compose:
    """
    Get preprocessing and augmentation pipeline for object detection.

    This function returns an Albumentations pipeline that:
    1. Resizes images while preserving aspect ratio (letterboxing)
    2. Applies data augmentation (only in train mode)
    3. Normalizes pixel values
    4. Automatically handles bounding box transformations

    Args:
        mode: 'train' for augmentation or 'val'/'test' for inference only
        img_size: Target image size (square). Default: 640
        normalize: Whether to normalize images with mean/std
        mean: RGB mean values for normalization (ImageNet defaults)
        std: RGB std values for normalization (ImageNet defaults)

    Returns:
        Albumentations Compose object with the full pipeline

    Example:
        >>> transform = get_transforms(mode='train', img_size=640)
        >>> transformed = transform(image=image, bboxes=bboxes, labels=labels)
        >>> img_transformed = transformed['image']
        >>> bboxes_transformed = transformed['bboxes']
    """

    if mode == 'train':
        # Training pipeline with augmentation
        transforms = [
            # 1. Letterbox resize (maintains aspect ratio, pads to square)
            A.LongestMaxSize(max_size=img_size, p=1.0),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114),  # Gray padding
                p=1.0
            ),

            # 2. Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
                p=0.5
            ),

            # 3. Photometric augmentations (address brightness/contrast variability)
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5
            ),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.3),

            # Add noise simulation (to make model robust to noisy images)
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(p=1.0),
            ], p=0.3),

            # Color augmentations
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),

            # Cutout augmentation (simulate occlusions)
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=114,
                p=0.3
            ),

            # 4. Normalization and tensor conversion
            A.Normalize(mean=mean, std=std) if normalize else A.NoOp(),
        ]

    else:  # val or test mode
        # Validation/Test pipeline (no augmentation, only preprocessing)
        transforms = [
            A.LongestMaxSize(max_size=img_size, p=1.0),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
                p=1.0
            ),
            A.Normalize(mean=mean, std=std) if normalize else A.NoOp(),
        ]

    # Compose with bbox parameters
    # bbox_params specifies format and minimum visibility threshold
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',  # (x_min, y_min, x_max, y_max)
            label_fields=['labels'],  # Field containing class labels
            min_area=25,  # Remove boxes smaller than 25 pixels²
            min_visibility=0.3,  # Remove boxes with <30% visible after transforms
        )
    )


def get_inference_transforms(
    img_size: int = 640,
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> A.Compose:
    """
    Get minimal preprocessing pipeline for inference (no augmentation).

    Alias for get_transforms with mode='test' for clarity in production code.

    Args:
        img_size: Target image size
        normalize: Whether to normalize
        mean: Normalization mean
        std: Normalization std

    Returns:
        Albumentations Compose for inference
    """
    return get_transforms(
        mode='test',
        img_size=img_size,
        normalize=normalize,
        mean=mean,
        std=std
    )


def convert_coco_to_pascal(
    bbox: List[float],
    img_width: int,
    img_height: int
) -> List[float]:
    """
    Convert COCO bbox format to Pascal VOC format.

    COCO format: [x, y, width, height] (top-left corner + dimensions)
    Pascal VOC format: [x_min, y_min, x_max, y_max] (top-left + bottom-right)

    Args:
        bbox: Bounding box in COCO format [x, y, w, h]
        img_width: Image width for clipping
        img_height: Image height for clipping

    Returns:
        Bounding box in Pascal VOC format [x_min, y_min, x_max, y_max]
    """
    x, y, w, h = bbox
    x_min = max(0, x)
    y_min = max(0, y)
    x_max = min(img_width, x + w)
    y_max = min(img_height, y + h)
    return [x_min, y_min, x_max, y_max]


def convert_pascal_to_coco(bbox: List[float]) -> List[float]:
    """
    Convert Pascal VOC bbox format to COCO format.

    Args:
        bbox: Bounding box in Pascal VOC format [x_min, y_min, x_max, y_max]

    Returns:
        Bounding box in COCO format [x, y, width, height]
    """
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    return [x_min, y_min, width, height]


def convert_pascal_to_yolo(
    bbox: List[float],
    img_width: int,
    img_height: int
) -> List[float]:
    """
    Convert Pascal VOC bbox format to YOLO format (normalized center coordinates).

    YOLO format: [x_center, y_center, width, height] (normalized 0-1)

    Args:
        bbox: Bounding box in Pascal VOC format [x_min, y_min, x_max, y_max]
        img_width: Image width
        img_height: Image height

    Returns:
        Bounding box in YOLO format [x_center, y_center, w, h] (normalized)
    """
    x_min, y_min, x_max, y_max = bbox

    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    return [x_center, y_center, width, height]


def validate_bbox(
    bbox: List[float],
    img_width: int,
    img_height: int,
    min_size: int = 1
) -> bool:
    """
    Validate if a bounding box is valid.

    Args:
        bbox: Bounding box in Pascal VOC format [x_min, y_min, x_max, y_max]
        img_width: Image width
        img_height: Image height
        min_size: Minimum bbox width/height in pixels

    Returns:
        True if bbox is valid, False otherwise
    """
    x_min, y_min, x_max, y_max = bbox

    # Check if coordinates are within image bounds
    if x_min < 0 or y_min < 0 or x_max > img_width or y_max > img_height:
        return False

    # Check if bbox has positive area
    width = x_max - x_min
    height = y_max - y_min

    if width < min_size or height < min_size:
        return False

    return True


class TACOPreprocessor:
    """
    Complete preprocessing pipeline for TACO dataset.

    This class provides a convenient interface for loading COCO-format
    annotations and applying transformations to both images and bounding boxes.

    Example:
        >>> preprocessor = TACOPreprocessor(mode='train', img_size=640)
        >>> result = preprocessor(image, bboxes, labels)
        >>> transformed_image = result['image']
        >>> transformed_bboxes = result['bboxes']
    """

    def __init__(
        self,
        mode: str = 'train',
        img_size: int = 640,
        normalize: bool = True,
        bbox_format: str = 'pascal_voc'
    ):
        """
        Initialize preprocessor.

        Args:
            mode: 'train', 'val', or 'test'
            img_size: Target image size
            normalize: Whether to normalize images
            bbox_format: Output bbox format ('pascal_voc', 'coco', 'yolo')
        """
        self.mode = mode
        self.img_size = img_size
        self.normalize = normalize
        self.bbox_format = bbox_format
        self.transform = get_transforms(
            mode=mode, img_size=img_size, normalize=normalize)

    def __call__(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        labels: List[int]
    ) -> Dict[str, Union[np.ndarray, List]]:
        """
        Apply preprocessing pipeline to image and annotations.

        Args:
            image: Input image (H, W, C) in RGB format
            bboxes: List of bounding boxes in Pascal VOC format
            labels: List of class labels (one per bbox)

        Returns:
            Dictionary with:
                - 'image': Transformed image
                - 'bboxes': Transformed bounding boxes
                - 'labels': Labels (unchanged)
        """
        # Apply transformations
        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            labels=labels
        )

        # Convert bbox format if needed
        if self.bbox_format == 'yolo':
            transformed_bboxes = [
                convert_pascal_to_yolo(bbox, self.img_size, self.img_size)
                for bbox in transformed['bboxes']
            ]
            transformed['bboxes'] = transformed_bboxes
        elif self.bbox_format == 'coco':
            transformed_bboxes = [
                convert_pascal_to_coco(bbox)
                for bbox in transformed['bboxes']
            ]
            transformed['bboxes'] = transformed_bboxes

        return transformed

    def preprocess_image_only(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image without bounding boxes (for inference on full images).

        Args:
            image: Input image (H, W, C) in RGB format

        Returns:
            Transformed image
        """
        # Create a simple transform without bbox handling
        transform = A.Compose([
            A.LongestMaxSize(max_size=self.img_size, p=1.0),
            A.PadIfNeeded(
                min_height=self.img_size,
                min_width=self.img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
                p=1.0
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ) if self.normalize else A.NoOp(),
        ])

        transformed = transform(image=image)
        return transformed['image']
