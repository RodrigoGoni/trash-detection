"""
Object detection models for trash detection
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Optional, Dict, List


class TrashDetector(nn.Module):
    """
    Object detection model for trash detection using Faster R-CNN

    Args:
        num_classes: Number of object classes (including background)
        backbone: Backbone architecture
        pretrained: Whether to use pretrained weights
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = 'resnet50',
        pretrained: bool = True
    ):
        super().__init__()

        if backbone == 'resnet50':
            # Load pretrained Faster R-CNN
            self.model = fasterrcnn_resnet50_fpn(pretrained=pretrained)

            # Replace the classifier with a new one
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, images, targets=None):
        """
        Args:
            images: List of images (Tensors)
            targets: List of targets (Dicts) - only during training

        Returns:
            During training: losses dict
            During inference: predictions list
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        return self.model(images, targets)


class YOLOv5Wrapper(nn.Module):
    """
    Wrapper for YOLOv5 model
    Note: Requires ultralytics package
    """

    def __init__(
        self,
        num_classes: int,
        model_size: str = 'yolov5s',
        pretrained: bool = True
    ):
        super().__init__()

        try:
            import ultralytics
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Please install ultralytics: pip install ultralytics")

        # Load model
        if pretrained:
            self.model = YOLO(f'{model_size}.pt')
        else:
            self.model = YOLO(f'{model_size}.yaml')

        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

    def train_model(self, data_yaml: str, epochs: int = 100, **kwargs):
        """Train the model"""
        return self.model.train(data=data_yaml, epochs=epochs, **kwargs)

    def predict(self, source, **kwargs):
        """Make predictions"""
        return self.model.predict(source, **kwargs)


class RetinaNetDetector(nn.Module):
    """
    RetinaNet object detection model
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = 'resnet50',
        pretrained: bool = True
    ):
        super().__init__()

        if backbone == 'resnet50':
            # Load pretrained RetinaNet
            self.model = torchvision.models.detection.retinanet_resnet50_fpn(
                pretrained=pretrained,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        return self.model(images, targets)


def non_max_suppression(
    predictions: List[Dict],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5
) -> List[Dict]:
    """
    Apply Non-Maximum Suppression to predictions

    Args:
        predictions: List of prediction dictionaries
        iou_threshold: IoU threshold for NMS
        score_threshold: Score threshold for filtering

    Returns:
        Filtered predictions
    """
    filtered_predictions = []

    for pred in predictions:
        # Filter by score threshold
        keep_mask = pred['scores'] > score_threshold

        boxes = pred['boxes'][keep_mask]
        scores = pred['scores'][keep_mask]
        labels = pred['labels'][keep_mask]

        # Apply NMS
        keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)

        filtered_predictions.append({
            'boxes': boxes[keep_indices],
            'scores': scores[keep_indices],
            'labels': labels[keep_indices]
        })

    return filtered_predictions
