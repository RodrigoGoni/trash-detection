"""
Image preprocessing utilities
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from PIL import Image


class ImagePreprocessor:
    """Image preprocessing pipeline"""

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize preprocessor

        Args:
            target_size: Target image size (height, width)
            normalize: Whether to normalize images
            mean: Mean values for normalization (ImageNet defaults)
            std: Std values for normalization (ImageNet defaults)
        """
        self.target_size = target_size
        self.normalize = normalize
        self.mean = np.array(mean).reshape(1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 3)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Process a single image"""
        # Resize
        image = cv2.resize(image, self.target_size)

        # Convert to float and scale to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Normalize
        if self.normalize:
            image = (image - self.mean) / self.std

        return image

    def process_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Process a batch of images"""
        return np.array([self(img) for img in images])

    @staticmethod
    def remove_background(image: np.ndarray, method: str = 'grabcut') -> np.ndarray:
        """
        Remove background from image

        Args:
            image: Input image
            method: Background removal method ('grabcut', 'threshold')
        """
        if method == 'grabcut':
            mask = np.zeros(image.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            # Define ROI
            rect = (10, 10, image.shape[1] - 10, image.shape[0] - 10)

            # Apply GrabCut
            cv2.grabCut(image, mask, rect, bgd_model,
                        fgd_model, 5, cv2.GC_INIT_WITH_RECT)

            # Create mask
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

            # Apply mask
            result = image * mask2[:, :, np.newaxis]

            return result

        elif method == 'threshold':
            # Simple thresholding
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Apply mask
            result = cv2.bitwise_and(image, image, mask=thresh)

            return result

        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def denoise(image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
        """Apply denoising to image"""
        if method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'median':
            return cv2.medianBlur(image, 5)
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def enhance_contrast(image: np.ndarray, method: str = 'clahe') -> np.ndarray:
        """Enhance image contrast"""
        if method == 'clahe':
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            # Merge channels
            enhanced = cv2.merge([l, a, b])

            # Convert back to BGR
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        elif method == 'histogram':
            # Histogram equalization
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        else:
            raise ValueError(f"Unknown method: {method}")
