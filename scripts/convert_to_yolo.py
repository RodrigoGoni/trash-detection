"""
Convert TACO dataset from COCO format to YOLO format for YOLOv11 training.

This script converts the preprocessed COCO annotations to YOLO format
and creates the necessary data.yaml configuration file.
"""

import json
import yaml
from pathlib import Path
import shutil
from tqdm import tqdm
import argparse


def convert_bbox_coco_to_yolo(bbox, img_width, img_height):
    """
    Convert COCO bbox [x, y, w, h] to YOLO [x_center, y_center, w, h] normalized.
    
    Args:
        bbox: COCO bbox [x_min, y_min, width, height]
        img_width: Image width
        img_height: Image height
    
    Returns:
        YOLO bbox [x_center_norm, y_center_norm, width_norm, height_norm]
    """
    x_min, y_min, width, height = bbox
    
    # Calculate center
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    
    # Normalize by image dimensions
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    # Clip to [0, 1]
    x_center_norm = max(0, min(1, x_center_norm))
    y_center_norm = max(0, min(1, y_center_norm))
    width_norm = max(0, min(1, width_norm))
    height_norm = max(0, min(1, height_norm))
    
    return [x_center_norm, y_center_norm, width_norm, height_norm]


def convert_split(
    annotations_file: Path,
    images_dir: Path,
    output_labels_dir: Path,
    output_images_dir: Path,
    class_mapping: dict,
    copy_images: bool = True  # Cambiado a True por defecto
):
    """
    Convert one split (train/val/test) from COCO to YOLO format.
    
    Handles TACO dataset structure with batch_X subdirectories.
    
    Args:
        annotations_file: Path to annotations.json
        images_dir: Path to images directory (may contain batch_X subdirs)
        output_labels_dir: Output directory for YOLO labels
        output_images_dir: Output directory for images
        class_mapping: Dict mapping category_id to class_index (0-based)
        copy_images: Whether to copy images to output directory
    
    Returns:
        Number of images processed
    """
    # Load COCO annotations
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Group annotations by image
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    # Process each image
    num_processed = 0
    num_skipped = 0
    
    for img_info in tqdm(coco_data['images'], desc=f"Converting {annotations_file.parent.name}"):
        img_id = img_info['id']
        
        # Skip images without annotations
        if img_id not in img_to_anns:
            num_skipped += 1
            continue
        
        img_width = img_info['width']
        img_height = img_info['height']
        img_filename = img_info['file_name']  # puede ser "batch_1/000001.jpg"
        
        # Buscar la imagen (puede estar en subdirectorio)
        source_img = images_dir / img_filename
        if not source_img.exists():
            # Intentar buscar en subdirectorios
            print(f"Warning: Image not found: {source_img}")
            num_skipped += 1
            continue
        
        # Crear nombre plano para YOLO (reemplazar / por _)
        # batch_1/000001.jpg -> batch_1_000001.jpg
        flat_filename = img_filename.replace('/', '_').replace('\\', '_')
        img_stem = Path(flat_filename).stem
        img_ext = Path(source_img).suffix
        
        # Copiar imagen con nombre plano
        dest_img = output_images_dir / f"{img_stem}{img_ext}"
        if not dest_img.exists():
            shutil.copy2(source_img, dest_img)
        
        # Crear YOLO label file (mismo stem que la imagen)
        label_file = output_labels_dir / f"{img_stem}.txt"
        
        with open(label_file, 'w') as f:
            for ann in img_to_anns[img_id]:
                # Get class index (0-based para YOLO)
                cat_id = ann['category_id']
                class_idx = class_mapping[cat_id]
                
                # Convert bbox to YOLO format
                coco_bbox = ann['bbox']
                yolo_bbox = convert_bbox_coco_to_yolo(coco_bbox, img_width, img_height)
                
                # Write YOLO format: class x_center y_center width height
                f.write(f"{class_idx} {' '.join(f'{v:.6f}' for v in yolo_bbox)}\n")
        
        num_processed += 1
    
    if num_skipped > 0:
        print(f"   Skipped {num_skipped} images (no annotations or not found)")
    
    return num_processed


def create_data_yaml(
    output_path: Path,
    dataset_root: Path,
    class_names: list,
    num_classes: int
):
    """
    Create YOLO data.yaml configuration file.
    
    Args:
        output_path: Path to save data.yaml
        dataset_root: Root directory of the dataset
        class_names: List of class names (ordered by class index)
        num_classes: Number of classes
    """
    # Create relative paths from dataset root
    data_config = {
        'path': str(dataset_root.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': num_classes,
        'names': class_names
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    
    print(f" Created data.yaml: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert TACO dataset to YOLO format')
    parser.add_argument('--processed-dir', type=str, default='./data/processed',
                      help='Path to processed COCO dataset')
    parser.add_argument('--output-dir', type=str, default='./data/yolo',
                      help='Output directory for YOLO format')
    parser.add_argument('--copy-images', action='store_true',
                      help='Copy images to output directory (otherwise use symlinks)')
    
    args = parser.parse_args()
    
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    
    print("="*60)
    print("CONVERTING TACO DATASET TO YOLO FORMAT")
    print("="*60)
    print(f"Source: {processed_dir}")
    print(f"Output: {output_dir}")
    print("="*60)
    
    # Load class information from train annotations
    train_ann_file = processed_dir / 'train' / 'annotations.json'
    with open(train_ann_file, 'r') as f:
        train_data = json.load(f)
    
    # Create class mapping: category_id -> class_index (0-based)
    categories = sorted(train_data['categories'], key=lambda x: x['id'])
    class_mapping = {cat['id']: idx for idx, cat in enumerate(categories)}
    class_names = [cat['name'] for cat in categories]
    num_classes = len(class_names)
    
    print(f"\nFound {num_classes} classes:")
    for idx, name in enumerate(class_names[:10]):
        print(f"  {idx}: {name}")
    if num_classes > 10:
        print(f"  ... and {num_classes - 10} more")
    
    # Convert each split
    splits = ['train', 'val', 'test']
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Converting {split.upper()} split...")
        print(f"{'='*60}")
        
        annotations_file = processed_dir / split / 'annotations.json'
        images_dir = processed_dir / split / 'images'
        output_labels_dir = output_dir / split / 'labels'
        output_images_dir = output_dir / split / 'images'
        
        if not annotations_file.exists():
            print(f" Warning: {annotations_file} not found, skipping {split}")
            continue
        
        # Siempre copiar imágenes (no usar symlinks porque batch_X complica las cosas)
        num_processed = convert_split(
            annotations_file=annotations_file,
            images_dir=images_dir,
            output_labels_dir=output_labels_dir,
            output_images_dir=output_images_dir,
            class_mapping=class_mapping,
            copy_images=True  # Siempre copiar
        )
        
        print(f" Processed {num_processed} images for {split}")
        
        # Verificar que las imágenes y labels coincidan
        num_images = len(list(output_images_dir.glob('*.[jJ][pP][gG]')))
        num_labels = len(list(output_labels_dir.glob('*.txt')))
        print(f"  Images: {num_images}, Labels: {num_labels}")
        
        if num_images != num_labels:
            print(f"   Warning: Image/label count mismatch!")
        else:
            print(f"   Image/label counts match")
    
    # Create data.yaml
    print(f"\n{'='*60}")
    print("Creating data.yaml...")
    print(f"{'='*60}")
    
    data_yaml_path = output_dir / 'data.yaml'
    create_data_yaml(
        output_path=data_yaml_path,
        dataset_root=output_dir,
        class_names=class_names,
        num_classes=num_classes
    )
    
    print(f"\n{'='*60}")
    print(" CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"\nYOLO dataset ready at: {output_dir}")
    print(f"Data config: {data_yaml_path}")
    print(f"\nTo train YOLOv11:")
    print(f"  python scripts/train_model.py --config config/train_config.yaml")
    print(f"  (Make sure to set model.name: 'YOLOv11' in config)")


if __name__ == '__main__':
    main()
