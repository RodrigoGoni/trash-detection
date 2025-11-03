"""Model architecture modules"""

from .detector import TrashDetector
from .backbone import get_backbone

__all__ = [
    'TrashClassifier',
    'TrashDetector',
    'get_backbone',
]
