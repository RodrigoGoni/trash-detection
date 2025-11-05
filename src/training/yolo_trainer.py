"""
YOLO Trainer utilities for trash detection.

Provides functions for training YOLO models and data format conversion.
"""

import os
from ultralytics import YOLO, settings
from typing import Dict, Optional
import json
from pathlib import Path
import yaml

def train_yolo_model(
    model,
    data_yaml: str,
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = 'cuda',
    project: str = './runs/detect',
    name: str = 'exp',
    yolo_config: Optional[Dict] = None,
    **kwargs
):
    """Train YOLO model with configuration parameters.
    
    Args:
        model: YOLO model instance
        data_yaml: Path to data YAML configuration
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Image size
        device: Device to train on
        project: Project directory
        name: Experiment name
        yolo_config: Dictionary with YOLO training parameters from config file
        **kwargs: Additional YOLO training arguments (override yolo_config)
        
    Returns:
        Training results
    """
    
    # Base training arguments
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'project': project,
        'name': name,
        'verbose': True,
        'exist_ok': True,
    }
    
    # Disable YOLO's built-in integrations (we use MLflow)
    os.environ['MLFLOW_TRACKING_URI'] = ''
    try:
        settings.update({'mlflow': False})
    except:
        pass
    
    # Apply YOLO-specific configuration if provided
    if yolo_config:
        # Direct training parameters
        direct_params = [
            'dropout', 'val', 'plots', 'save', 'lr0', 'lrf', 'momentum', 
            'weight_decay', 'warmup_epochs', 'warmup_momentum', 
            'warmup_bias_lr', 'box', 'cls', 'dfl', 'optimizer', 
            'seed', 'deterministic', 'single_cls', 'classes', 'rect', 
            'multi_scale', 'compile', 'cos_lr', 'close_mosaic', 'resume', 
            'patience'
        ]
        
        for param in direct_params:
            if param in yolo_config:
                train_args[param] = yolo_config[param]
        
        # Augmentation parameters
        if 'augmentation' in yolo_config:
            aug_config = yolo_config['augmentation']
            aug_params = [
                'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate',
                'scale', 'shear', 'perspective', 'flipud', 'fliplr',
                'mosaic', 'mixup', 'copy_paste'
            ]
            
            for param in aug_params:
                if param in aug_config:
                    train_args[param] = aug_config[param]
    
    # Merge additional kwargs (these override yolo_config)
    train_args.update(kwargs)
    
    # Train model
    results = model.train(**train_args)
    
    return results


def convert_coco_to_yolo_format(
    coco_json_path: str,
    output_dir: str,
    images_dir: str,
    class_names: list
):
    """
    Convert COCO format annotations to YOLO format.
    
    Args:
        coco_json_path: Path to COCO JSON file
        output_dir: Output directory for YOLO labels
        images_dir: Directory containing images
        class_names: List of class names
    """

    
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create category mapping
    cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    
    # Group annotations by image
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    # Convert each image's annotations
    for img in coco_data['images']:
        img_id = img['id']
        if img_id not in img_to_anns:
            continue
        
        img_width = img['width']
        img_height = img['height']
        
        # Get image filename without extension
        img_filename = Path(img['file_name']).stem
        
        # Create YOLO label file
        label_file = output_path / f"{img_filename}.txt"
        
        with open(label_file, 'w') as f:
            for ann in img_to_anns[img_id]:
                # Get category index
                cat_idx = cat_id_to_idx[ann['category_id']]
                
                # Convert COCO bbox [x, y, w, h] to YOLO [x_center, y_center, w, h] (normalized)
                x, y, w, h = ann['bbox']
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                w_norm = w / img_width
                h_norm = h / img_height
                
                # Write YOLO format: class x_center y_center width height
                f.write(f"{cat_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    
    print(f" Converted {len(img_to_anns)} images to YOLO format")
    print(f"  Output: {output_dir}")


def create_yolo_data_yaml(
    output_path: str,
    train_images: str,
    val_images: str,
    test_images: str,
    class_names: list,
    num_classes: int
):
    """
    Create YOLO data.yaml configuration file.
    
    Args:
        output_path: Path to save data.yaml
        train_images: Path to training images
        val_images: Path to validation images
        test_images: Path to test images (optional)
        class_names: List of class names
        num_classes: Number of classes
    """
    
    data_config = {
        'path': str(Path(train_images).parent.parent.absolute()),
        'train': str(Path(train_images).relative_to(Path(train_images).parent.parent)),
        'val': str(Path(val_images).relative_to(Path(val_images).parent.parent)),
        'test': str(Path(test_images).relative_to(Path(test_images).parent.parent)) if test_images else '',
        'nc': num_classes,
        'names': class_names
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    
    print(f" Created YOLO data configuration: {output_path}")
