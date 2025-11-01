"""
Classification models for trash detection
"""

import torch
import torch.nn as nn
from typing import Optional
from .backbone import get_backbone


class TrashClassifier(nn.Module):
    """
    Image classification model for trash detection

    Args:
        num_classes: Number of output classes
        backbone: Name of backbone network
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        super().__init__()

        self.backbone = get_backbone(backbone, pretrained=pretrained)
        feature_dim = self.backbone.feature_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) == 4:  # If backbone returns 4D tensor
            features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output

    def extract_features(self, x):
        """Extract features without classification"""
        features = self.backbone(x)
        if len(features.shape) == 4:
            features = features.view(features.size(0), -1)
        return features


class MultiLabelTrashClassifier(nn.Module):
    """
    Multi-label classification model for trash detection
    (Can detect multiple types of trash in one image)
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        super().__init__()

        self.backbone = get_backbone(backbone, pretrained=pretrained)
        feature_dim = self.backbone.feature_dim

        # Multi-label classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
            nn.Sigmoid()  # Sigmoid for multi-label
        )

    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) == 4:
            features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output


class EnsembleClassifier(nn.Module):
    """
    Ensemble of multiple classifiers
    """

    def __init__(
        self,
        models: list,
        num_classes: int,
        ensemble_method: str = 'average'
    ):
        super().__init__()

        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        self.num_classes = num_classes

    def forward(self, x):
        outputs = []

        for model in self.models:
            output = model(x)
            outputs.append(output)

        # Stack outputs
        stacked = torch.stack(outputs)

        if self.ensemble_method == 'average':
            return stacked.mean(dim=0)
        elif self.ensemble_method == 'max':
            return stacked.max(dim=0)[0]
        elif self.ensemble_method == 'voting':
            # Get predictions
            preds = torch.argmax(stacked, dim=-1)
            # Mode voting
            return torch.mode(preds, dim=0)[0]
        else:
            raise ValueError(
                f"Unknown ensemble method: {self.ensemble_method}")
