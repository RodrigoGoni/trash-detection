"""
TACO Object Detection Dataset and DataLoader.

Provides PyTorch Dataset for loading preprocessed TACO dataset in COCO format.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))
from data.preprocessing import TACOPreprocessor, convert_coco_to_pascal, validate_bbox


class TACODetectionDataset(Dataset):
    """
    PyTorch Dataset for TACO object detection.
    
    Args:
        processed_dir: Path to processed dataset directory
        split: 'train', 'val', or 'test'
        img_size: Target image size (default: 640)
        normalize: Whether to normalize images (default: True)
        bbox_format: Output bbox format (default: 'pascal_voc')
    """
    
    def __init__(self, processed_dir: str, split: str = 'train', img_size: int = 640,
                 normalize: bool = True, bbox_format: str = 'pascal_voc'):
        self.processed_dir = Path(processed_dir)
        self.split = split
        
        # Load annotations
        annotations_file = self.processed_dir / split / 'annotations.json'
        if not annotations_file.exists():
            raise FileNotFoundError(f"Annotations not found: {annotations_file}")
        
        with open(annotations_file, 'r') as f:
            self.data = json.load(f)
        
        self.images = self.data['images']
        self.annotations = self.data['annotations']
        self.categories = self.data['categories']
        
        # Create mappings
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.categories}
        
        # Filter valid images
        self.valid_images = [img for img in self.images if img['id'] in self.img_to_anns]
        
        # Preprocessor
        self.preprocessor = TACOPreprocessor(mode=split, img_size=img_size,
                                            normalize=normalize, bbox_format=bbox_format)
    
    def __len__(self) -> int:
        return len(self.valid_images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                current_idx = (idx + attempt) % len(self)
                img_info = self.valid_images[current_idx]
                img_filename = Path(img_info['file_name']).name
                img_path = self.processed_dir / self.split / 'images' / img_filename
                
                # Load image
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                orig_h, orig_w = image.shape[:2]
                
                # Get annotations
                annotations = self.img_to_anns.get(img_info['id'], [])
                bboxes = []
                labels = []
                
                for ann in annotations:
                    bbox_coco = ann['bbox']
                    bbox_pascal = convert_coco_to_pascal(bbox_coco, img_info['width'], 
                                                         img_info['height'])
                    if validate_bbox(bbox_pascal, img_info['width'], img_info['height']):
                        # Clip bbox to image boundaries to prevent Albumentations errors
                        x_min, y_min, x_max, y_max = bbox_pascal
                        x_min = max(0, min(x_min, orig_w - 1))
                        y_min = max(0, min(y_min, orig_h - 1))
                        x_max = max(x_min + 1, min(x_max, orig_w))
                        y_max = max(y_min + 1, min(y_max, orig_h))
                        
                        # Only keep if bbox still has valid area
                        if (x_max - x_min) > 1 and (y_max - y_min) > 1:
                            bboxes.append([x_min, y_min, x_max, y_max])
                            labels.append(ann['category_id'])
                
                # Skip if no valid bboxes
                if len(bboxes) == 0:
                    continue
                
                # Apply preprocessing
                result = self.preprocessor(image, bboxes, labels)
                
                # Skip if all bboxes were filtered out during augmentation
                if len(result['bboxes']) == 0:
                    continue
                
                image_tensor = torch.from_numpy(result['image']).permute(2, 0, 1).float()
                bboxes_tensor = torch.FloatTensor(result['bboxes'])
                labels_tensor = torch.LongTensor(result['labels'])
                
                return {
                    'images': image_tensor,
                    'bboxes': bboxes_tensor,
                    'labels': labels_tensor
                }
            except Exception as e:
                print(f"Error loading sample {current_idx}: {e}")
                continue
        
        # If all attempts failed, return a dummy sample with one small bbox
        print(f"Warning: Could not load valid sample after {max_attempts} attempts, returning dummy")
        dummy_image = torch.zeros((3, 640, 640), dtype=torch.float32)
        dummy_bbox = torch.FloatTensor([[10, 10, 20, 20]])
        dummy_label = torch.LongTensor([1])
        
        return {
            'images': dummy_image,
            'bboxes': dummy_bbox,
            'labels': dummy_label
        }
    
    def get_category_name(self, cat_id: int) -> str:
        """Get category name from ID."""
        return self.cat_id_to_name.get(cat_id, f'unknown_{cat_id}')


def collate_fn(batch: List[Dict]) -> Dict[str, List]:
    """
    Custom collate function for variable-size bboxes.
    
    Returns batched data with images stacked and bboxes/labels as lists.
    """
    # Handle both 'image' and 'images' keys for compatibility
    if 'images' in batch[0]:
        images = torch.stack([sample['images'] for sample in batch], dim=0)
    else:
        images = torch.stack([sample['image'] for sample in batch], dim=0)
    
    bboxes = [sample['bboxes'] for sample in batch]
    labels = [sample['labels'] for sample in batch]
    
    return {
        'images': images,
        'bboxes': bboxes,
        'labels': labels
    }


def create_dataloader(processed_dir: str, split: str = 'train', batch_size: int = 16,
                     img_size: int = 640, num_workers: int = 4, 
                     shuffle: Optional[bool] = None, pin_memory: bool = True) -> DataLoader:
    """
    Create a DataLoader for TACO dataset.
    
    Args:
        processed_dir: Path to processed dataset
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        img_size: Image size for preprocessing
        num_workers: Number of worker processes
        shuffle: Whether to shuffle (None = auto based on split)
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        PyTorch DataLoader
    """
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataset = TACODetectionDataset(processed_dir=processed_dir, split=split, img_size=img_size)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=(split == 'train')
    )
    
    return dataloader

