"""
Convert TACO dataset from COCO format to YOLO Instance Segmentation format.

This script converts COCO annotations with segmentation polygons to YOLO format
for instance segmentation training with YOLOv11-seg.

YOLO Segmentation Format:
- Each line: <class_index> <x1> <y1> <x2> <y2> ... <xn> <yn>
- Coordinates normalized to [0, 1] relative to image dimensions
- Polygons with variable number of points
"""

import json
import yaml
from pathlib import Path
import shutil
from tqdm import tqdm
import argparse
import numpy as np


def validate_polygon(segmentation, img_width, img_height, min_points=3):
    """
    Validate and filter polygon segmentation.
    
    Args:
        segmentation: COCO segmentation (list of lists or RLE)
        img_width: Image width
        img_height: Image height
        min_points: Minimum number of polygon points
    
    Returns:
        Valid polygon points or None if invalid
    """
    # Skip RLE format (dict with 'counts' and 'size')
    if isinstance(segmentation, dict):
        return None
    
    # COCO can have multiple polygons per annotation
    if not isinstance(segmentation, list) or len(segmentation) == 0:
        return None
    
    # Get first polygon (YOLO uses one polygon per annotation)
    polygon = segmentation[0]
    
    # Check if it's a valid list of coordinates
    if not isinstance(polygon, list) or len(polygon) < min_points * 2:
        return None
    
    # Validate coordinate pairs
    if len(polygon) % 2 != 0:
        return None
    
    # Convert to numpy array and reshape to (N, 2)
    try:
        points = np.array(polygon).reshape(-1, 2)
    except:
        return None
    
    # Validate points are within image bounds
    if np.any(points[:, 0] < 0) or np.any(points[:, 0] > img_width):
        return None
    if np.any(points[:, 1] < 0) or np.any(points[:, 1] > img_height):
        return None
    
    return points


def normalize_polygon(points, img_width, img_height):
    """
    Normalize polygon coordinates to [0, 1].
    
    Args:
        points: Numpy array of shape (N, 2) with [x, y] coordinates
        img_width: Image width
        img_height: Image height
    
    Returns:
        Normalized points as flat list
    """
    normalized = points.copy()
    normalized[:, 0] = normalized[:, 0] / img_width
    normalized[:, 1] = normalized[:, 1] / img_height
    
    # Clip to [0, 1]
    normalized = np.clip(normalized, 0.0, 1.0)
    
    return normalized.flatten().tolist()


def simplify_polygon(points, epsilon=2.0):
    """
    Simplify polygon using Douglas-Peucker algorithm (optional).
    
    Args:
        points: Numpy array of shape (N, 2)
        epsilon: Tolerance for simplification
    
    Returns:
        Simplified points
    """
    try:
        from scipy.spatial import distance
        # Simple implementation - can use cv2.approxPolyDP for better results
        return points
    except:
        return points


def convert_split(
    annotations_file: Path,
    images_dir: Path,
    output_labels_dir: Path,
    output_images_dir: Path,
    class_mapping: dict,
    copy_images: bool = True,
    min_polygon_points: int = 3
):
    """
    Convert one split (train/val/test) from COCO to YOLO segmentation format.
    
    Handles TACO dataset structure with batch_X subdirectories.
    
    Args:
        annotations_file: Path to annotations.json
        images_dir: Path to images directory (may contain batch_X subdirs)
        output_labels_dir: Output directory for YOLO labels
        output_images_dir: Output directory for images
        class_mapping: Dict mapping category_id to class_index (0-based)
        copy_images: Whether to copy images to output directory
        min_polygon_points: Minimum points required for valid polygon
    
    Returns:
        Statistics dict
    """
    # Load COCO annotations
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Create image info lookup
    img_info_dict = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    # Statistics
    stats = {
        'total_images': 0,
        'processed_images': 0,
        'skipped_no_annotations': 0,
        'skipped_not_found': 0,
        'total_annotations': 0,
        'valid_annotations': 0,
        'skipped_no_segmentation': 0,
        'skipped_invalid_polygon': 0
    }
    
    # Process each image
    for img_id, img_info in tqdm(img_info_dict.items(), desc=f"Converting {annotations_file.parent.name}"):
        stats['total_images'] += 1
        
        # Skip images without annotations
        if img_id not in img_to_anns:
            stats['skipped_no_annotations'] += 1
            continue
        
        img_width = img_info['width']
        img_height = img_info['height']
        img_filename = img_info['file_name']  # puede ser "batch_1/000001.jpg"
        
        # Buscar la imagen (puede estar en subdirectorio)
        source_img = images_dir / img_filename
        if not source_img.exists():
            print(f"\nWarning: Image not found: {source_img}")
            stats['skipped_not_found'] += 1
            continue
        
        # Crear nombre plano para YOLO (reemplazar / por _)
        flat_filename = img_filename.replace('/', '_').replace('\\', '_')
        img_stem = Path(flat_filename).stem
        img_ext = Path(source_img).suffix
        
        # Copiar imagen con nombre plano
        if copy_images:
            dest_img = output_images_dir / f"{img_stem}{img_ext}"
            if not dest_img.exists():
                shutil.copy2(source_img, dest_img)
        
        # Crear YOLO label file (mismo stem que la imagen)
        label_file = output_labels_dir / f"{img_stem}.txt"
        
        valid_annotations = []
        
        for ann in img_to_anns[img_id]:
            stats['total_annotations'] += 1
            
            # Check if segmentation exists
            if 'segmentation' not in ann or not ann['segmentation']:
                stats['skipped_no_segmentation'] += 1
                continue
            
            # Validate polygon
            points = validate_polygon(
                ann['segmentation'],
                img_width,
                img_height,
                min_points=min_polygon_points
            )
            
            if points is None:
                stats['skipped_invalid_polygon'] += 1
                continue
            
            # Get class index (0-based para YOLO)
            cat_id = ann['category_id']
            class_idx = class_mapping[cat_id]
            
            # Normalize polygon coordinates
            normalized_coords = normalize_polygon(points, img_width, img_height)
            
            valid_annotations.append((class_idx, normalized_coords))
            stats['valid_annotations'] += 1
        
        # Write label file only if we have valid annotations
        if valid_annotations:
            with open(label_file, 'w') as f:
                for class_idx, coords in valid_annotations:
                    # Format: class_idx x1 y1 x2 y2 ... xn yn
                    coords_str = ' '.join(f'{c:.6f}' for c in coords)
                    f.write(f"{class_idx} {coords_str}\n")
            
            stats['processed_images'] += 1
        else:
            stats['skipped_no_annotations'] += 1
    
    return stats


def create_data_yaml(
    output_path: Path,
    dataset_root: Path,
    class_names: list,
    num_classes: int
):
    """
    Create YOLO data.yaml configuration file for segmentation.
    
    Args:
        output_path: Path to save data.yaml
        dataset_root: Root directory of the dataset
        class_names: List of class names (ordered by class index)
        num_classes: Number of classes
    """
    # Create configuration for segmentation
    data_config = {
        'path': str(dataset_root.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': num_classes,
        'names': class_names,
        'task': 'segment'  # Importante para segmentación
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ Created data.yaml: {output_path}")


def print_statistics(stats: dict, split_name: str):
    """Print conversion statistics."""
    print(f"\n{'='*60}")
    print(f"STATISTICS FOR {split_name.upper()} SPLIT")
    print(f"{'='*60}")
    print(f"Images:")
    print(f"  Total images in annotations: {stats['total_images']}")
    print(f"  Processed images: {stats['processed_images']}")
    print(f"  Skipped (no annotations): {stats['skipped_no_annotations']}")
    print(f"  Skipped (not found): {stats['skipped_not_found']}")
    print(f"\nAnnotations:")
    print(f"  Total annotations: {stats['total_annotations']}")
    print(f"  Valid annotations: {stats['valid_annotations']}")
    print(f"  Skipped (no segmentation): {stats['skipped_no_segmentation']}")
    print(f"  Skipped (invalid polygon): {stats['skipped_invalid_polygon']}")
    
    if stats['total_annotations'] > 0:
        valid_pct = (stats['valid_annotations'] / stats['total_annotations']) * 100
        print(f"\nConversion rate: {valid_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Convert TACO dataset to YOLO Segmentation format')
    parser.add_argument('--processed-dir', type=str, default='./data/processed',
                      help='Path to processed COCO dataset')
    parser.add_argument('--output-dir', type=str, default='./data/yolo_seg',
                      help='Output directory for YOLO segmentation format')
    parser.add_argument('--copy-images', action='store_true', default=True,
                      help='Copy images to output directory')
    parser.add_argument('--min-polygon-points', type=int, default=3,
                      help='Minimum points required for valid polygon')
    
    args = parser.parse_args()
    
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    
    print("="*60)
    print("CONVERTING TACO DATASET TO YOLO SEGMENTATION FORMAT")
    print("="*60)
    print(f"Source: {processed_dir}")
    print(f"Output: {output_dir}")
    print(f"Min polygon points: {args.min_polygon_points}")
    print("="*60)
    
    # Load class information from train annotations
    train_ann_file = processed_dir / 'train' / 'annotations.json'
    if not train_ann_file.exists():
        print(f"\n❌ Error: {train_ann_file} not found!")
        return
    
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
    all_stats = {}
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Converting {split.upper()} split...")
        print(f"{'='*60}")
        
        annotations_file = processed_dir / split / 'annotations.json'
        images_dir = processed_dir / split / 'images'
        output_labels_dir = output_dir / split / 'labels'
        output_images_dir = output_dir / split / 'images'
        
        if not annotations_file.exists():
            print(f"⚠ Warning: {annotations_file} not found, skipping {split}")
            continue
        
        # Convert split
        stats = convert_split(
            annotations_file=annotations_file,
            images_dir=images_dir,
            output_labels_dir=output_labels_dir,
            output_images_dir=output_images_dir,
            class_mapping=class_mapping,
            copy_images=args.copy_images,
            min_polygon_points=args.min_polygon_points
        )
        
        all_stats[split] = stats
        
        # Print statistics
        print_statistics(stats, split)
        
        # Verificar que las imágenes y labels coincidan
        num_images = len(list(output_images_dir.glob('*.[jJ][pP][gG]')))
        num_labels = len(list(output_labels_dir.glob('*.txt')))
        print(f"\nFiles created:")
        print(f"  Images: {num_images}")
        print(f"  Labels: {num_labels}")
        
        if num_images != num_labels:
            print(f"  ⚠ Warning: Image/label count mismatch!")
        else:
            print(f"  ✓ Image/label counts match")
    
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
    
    # Summary
    print(f"\n{'='*60}")
    print("✓ CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"\nYOLO Segmentation dataset ready at: {output_dir}")
    print(f"Data config: {data_yaml_path}")
    
    # Overall statistics
    print(f"\n{'='*60}")
    print("OVERALL STATISTICS")
    print(f"{'='*60}")
    for split, stats in all_stats.items():
        print(f"\n{split.upper()}:")
        print(f"  Processed images: {stats['processed_images']}")
        print(f"  Valid annotations: {stats['valid_annotations']}")
        if stats['total_annotations'] > 0:
            valid_pct = (stats['valid_annotations'] / stats['total_annotations']) * 100
            print(f"  Conversion rate: {valid_pct:.1f}%")
    
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print(f"\nTo train YOLOv11 Instance Segmentation:")
    print(f"  python scripts/train_yolo_segmentation.py --config config/train_config_yolo11_segmentation.yaml")


if __name__ == '__main__':
    main()
