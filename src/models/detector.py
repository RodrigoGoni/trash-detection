import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.detection.roi_heads as roi_heads_module
from typing import Optional, Dict, List

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, 
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_mobilenet_v3_large_320_fpn
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ultralytics import YOLO, settings as ultralytics_settings


def custom_fastrcnn_loss(class_logits, box_regression, labels, regression_targets, 
                        custom_loss_fn=None):
    """Custom loss function for Faster R-CNN with support for focal loss"""
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    
    # Classification loss with custom loss function support
    if custom_loss_fn is not None:
        # Use only the classification loss component from DetectionLoss
        classification_loss = custom_loss_fn.classification_loss(class_logits, labels)
        
        # Compute bounding box regression loss using DetectionLoss's bbox loss
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        if sampled_pos_inds_subset.numel() > 0:
            labels_pos = labels[sampled_pos_inds_subset]
            N, num_classes = class_logits.shape
            box_regression_reshaped = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
            
            box_loss = custom_loss_fn.bbox_loss(
                box_regression_reshaped[sampled_pos_inds_subset, labels_pos],
                regression_targets[sampled_pos_inds_subset],
                labels_pos
            )
        else:
            box_loss = torch.tensor(0.0, device=class_logits.device)
    else:
        classification_loss = F.cross_entropy(class_logits, labels)
        
        # Bounding box regression loss
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        N, num_classes = class_logits.shape
        box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction="sum",
        )
        box_loss = box_loss / labels.numel()

    return classification_loss, box_loss

class TrashDetector(nn.Module):
    """Faster R-CNN model for trash detection
    
    Args:
        num_classes: Number of classes including background
        backbone: Model backbone (resnet50, mobilenet_v3_large, mobilenet_v3_large_320)
        pretrained: Use pretrained weights
        custom_loss_fn: Optional custom loss function
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        custom_loss_fn = None
    ):
        super().__init__()

        if backbone == 'resnet50':
            self.model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
        elif backbone == 'mobilenet_v3_large':
            self.model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=pretrained)
        elif backbone == 'mobilenet_v3_large_320':
            self.model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Replace classifier head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        self.custom_loss_fn = custom_loss_fn
        
        # Apply custom loss if provided
        if custom_loss_fn is not None:
            self._apply_custom_loss(custom_loss_fn)

    def _apply_custom_loss(self, custom_loss_fn):
        """Replace default loss with custom loss function"""
        original_loss = roi_heads_module.fastrcnn_loss
        
        def wrapped_loss(class_logits, box_regression, labels, regression_targets):
            return custom_fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets,
                custom_loss_fn=custom_loss_fn
            )
        
        roi_heads_module.fastrcnn_loss = wrapped_loss
        self._original_fastrcnn_loss = original_loss

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("Targets required in training mode")
        return self.model(images, targets)


class YOLOv11Detector(nn.Module):
    """YOLOv11 detector for trash detection
    
    Args:
        num_classes: Number of classes without background
        model_size: Model variant (n, s, m, l, x)
        pretrained: Use pretrained weights
        img_size: Input image size
    """

    def __init__(
        self,
        num_classes: int,
        model_size: str = 's',
        pretrained: bool = True,
        img_size: int = 640
    ):
        super().__init__()

        valid_sizes = ['n', 's', 'm', 'l', 'x']
        if model_size not in valid_sizes:
            raise ValueError(f"Invalid model_size: {model_size}. Use one of {valid_sizes}")

        model_name = f'yolo11{model_size}.pt' if pretrained else f'yolo11{model_size}.yaml'
        
        self.model = YOLO(model_name)
        self.num_classes = num_classes
        self.img_size = img_size
        self.model_size = model_size
        
        print(f"Loaded YOLOv11-{model_size.upper()} {'with pretrained weights' if pretrained else 'from scratch'}")

    def forward(self, x):
        return self.model(x)

    def train_yolo(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        device: str = 'cuda',
        project: str = './runs/detect',
        name: str = 'exp',
        yolo_config: Optional[Dict] = None,
        **kwargs
    ):
        """Train YOLO model with configuration
        
        Args:
            data_yaml: Path to data YAML file
            epochs: Number of epochs
            batch_size: Batch size
            device: Device to use
            project: Project directory
            name: Experiment name
            yolo_config: YOLO configuration dict from config file
            **kwargs: Additional arguments override yolo_config
        """
        from src.training.yolo_trainer import train_yolo_model
        
        ultralytics_settings.update({
            'mlflow': False,
            'comet': False,
            'tensorboard': False,
            'wandb': False
        })
        
        return train_yolo_model(
            model=self.model,
            data_yaml=data_yaml,
            epochs=epochs,
            batch_size=batch_size,
            img_size=self.img_size,
            device=device,
            project=project,
            name=name,
            yolo_config=yolo_config,
            **kwargs
        )

    def predict(self, source, **kwargs):
        return self.model.predict(source, imgsz=self.img_size, **kwargs)
    
    def val(self, data_yaml: str, **kwargs):
        return self.model.val(data=data_yaml, imgsz=self.img_size, **kwargs)


class YOLOv11SegmentationDetector(nn.Module):
    """YOLOv11 Instance Segmentation detector for trash detection
    
    This model extends YOLOv11 for instance segmentation tasks, providing
    both bounding box detection and pixel-level segmentation masks.
    
    Args:
        num_classes: Number of classes without background
        model_size: Model variant (n, s, m, l, x)
        pretrained: Use pretrained weights (COCO-seg pretrained)
        img_size: Input image size
    """

    def __init__(
        self,
        num_classes: int,
        model_size: str = 's',
        pretrained: bool = True,
        img_size: int = 640
    ):
        super().__init__()

        valid_sizes = ['n', 's', 'm', 'l', 'x']
        if model_size not in valid_sizes:
            raise ValueError(f"Invalid model_size: {model_size}. Use one of {valid_sizes}")

        # Use segmentation model variant
        model_name = f'yolo11{model_size}-seg.pt' if pretrained else f'yolo11{model_size}-seg.yaml'
        
        self.model = YOLO(model_name)
        self.num_classes = num_classes
        self.img_size = img_size
        self.model_size = model_size
        self.task = 'segment'
        
        print(f"Loaded YOLOv11-{model_size.upper()}-seg {'with pretrained weights' if pretrained else 'from scratch'}")

    def forward(self, x):
        return self.model(x)

    def train_yolo_segmentation(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        device: str = 'cuda',
        project: str = './runs/segment',
        name: str = 'exp',
        yolo_config: Optional[Dict] = None,
        **kwargs
    ):
        """Train YOLO segmentation model with configuration
        
        Args:
            data_yaml: Path to data YAML file (should have task: segment)
            epochs: Number of epochs
            batch_size: Batch size
            device: Device to use
            project: Project directory (default: ./runs/segment)
            name: Experiment name
            yolo_config: YOLO configuration dict from config file
            **kwargs: Additional arguments override yolo_config
        
        Returns:
            Training results
        """
        from src.training.yolo_trainer import train_yolo_segmentation_model
        
        # Disable external integrations (we use MLflow manually)
        ultralytics_settings.update({
            'mlflow': False,
            'comet': False,
            'tensorboard': False,
            'wandb': False
        })
        
        return train_yolo_segmentation_model(
            model=self.model,
            data_yaml=data_yaml,
            epochs=epochs,
            batch_size=batch_size,
            img_size=self.img_size,
            device=device,
            project=project,
            name=name,
            yolo_config=yolo_config,
            **kwargs
        )

    def predict(self, source, **kwargs):
        """Run prediction on source (image/video/directory)"""
        return self.model.predict(source, imgsz=self.img_size, **kwargs)
    
    def val(self, data_yaml: str, **kwargs):
        """Run validation on dataset"""
        return self.model.val(data=data_yaml, imgsz=self.img_size, **kwargs)

