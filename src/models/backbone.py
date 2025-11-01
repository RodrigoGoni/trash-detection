"""
Backbone networks for feature extraction
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


def get_backbone(
    name: str = 'resnet50',
    pretrained: bool = True,
    freeze_layers: int = 0
) -> nn.Module:
    """
    Get a backbone network for feature extraction

    Args:
        name: Backbone name (resnet50, efficientnet_b0, mobilenet_v3, vit_b_16, etc.)
        pretrained: Whether to use pretrained weights
        freeze_layers: Number of layers to freeze (0 = none)

    Returns:
        Backbone model
    """

    if name == 'resnet50':
        backbone = models.resnet50(pretrained=pretrained)
        feature_dim = backbone.fc.in_features
        backbone = nn.Sequential(*list(backbone.children())[:-1])

    elif name == 'resnet101':
        backbone = models.resnet101(pretrained=pretrained)
        feature_dim = backbone.fc.in_features
        backbone = nn.Sequential(*list(backbone.children())[:-1])

    elif name == 'efficientnet_b0':
        backbone = models.efficientnet_b0(pretrained=pretrained)
        feature_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

    elif name == 'efficientnet_b3':
        backbone = models.efficientnet_b3(pretrained=pretrained)
        feature_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

    elif name == 'mobilenet_v3_large':
        backbone = models.mobilenet_v3_large(pretrained=pretrained)
        feature_dim = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()

    elif name == 'vit_b_16':
        backbone = models.vit_b_16(pretrained=pretrained)
        feature_dim = backbone.heads.head.in_features
        backbone.heads = nn.Identity()

    elif name == 'convnext_tiny':
        backbone = models.convnext_tiny(pretrained=pretrained)
        feature_dim = backbone.classifier[2].in_features
        backbone.classifier = nn.Identity()

    else:
        raise ValueError(f"Unknown backbone: {name}")

    # Freeze layers if specified
    if freeze_layers > 0:
        count = 0
        for child in backbone.children():
            count += 1
            if count <= freeze_layers:
                for param in child.parameters():
                    param.requires_grad = False

    # Add feature_dim as attribute
    backbone.feature_dim = feature_dim

    return backbone


class CustomBackbone(nn.Module):
    """Custom CNN backbone"""

    def __init__(self, num_classes: int = 1000):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 512

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
