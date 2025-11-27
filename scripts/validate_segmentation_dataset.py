"""
Validate and analyze COCO dataset for instance segmentation quality.

This script checks:
- Segmentation polygon quality (minimum points, validity)
- Image-to-annotation consistency
- Class distribution for segmentation
- Polygon statistics (area, complexity)
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm


def validate_polygon(segmentation, img_width, img_height, min_points=3):
    """Validate polygon segmentation."""
    # Skip RLE format
    if isinstance(segmentation, dict):
        return False, "RLE_format"
    
    if not isinstance(segmentation, list) or len(segmentation) == 0:
        return False, "empty_or_invalid"
    
    # Get first polygon
    polygon = segmentation[0]
    
    if not isinstance(polygon, list) or len(polygon) < min_points * 2:
        return False, "too_few_points"
    
    if len(polygon) % 2 != 0:
        return False, "odd_coordinates"
    
    # Convert to numpy array
    try:
        points = np.array(polygon).reshape(-1, 2)
    except:
        return False, "reshape_error"
    
    # Check bounds
    if np.any(points[:, 0] < 0) or np.any(points[:, 0] > img_width):
        return False, "x_out_of_bounds"
    if np.any(points[:, 1] < 0) or np.any(points[:, 1] > img_height):
        return False, "y_out_of_bounds"
    
    return True, "valid"


def calculate_polygon_area(segmentation, img_width, img_height):
    """Calculate polygon area using shoelace formula."""
    if isinstance(segmentation, dict) or not isinstance(segmentation, list):
        return 0.0
    
    polygon = segmentation[0]
    if len(polygon) < 6:  # Need at least 3 points
        return 0.0
    
    try:
        points = np.array(polygon).reshape(-1, 2)
        x = points[:, 0]
        y = points[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        # Normalize by image area
        normalized_area = area / (img_width * img_height)
        return normalized_area
    except:
        return 0.0


def analyze_segmentation_dataset(annotations_file: Path, split_name: str):
    """Analyze segmentation dataset quality."""
    print(f"\n{'='*60}")
    print(f"ANALYZING {split_name.upper()} SPLIT")
    print(f"{'='*60}")
    
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    # Create lookups
    img_info = {img['id']: img for img in data['images']}
    cat_info = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Statistics
    stats = {
        'total_images': len(data['images']),
        'total_annotations': len(data['annotations']),
        'images_with_annotations': set(),
        'validation_errors': defaultdict(int),
        'class_distribution': Counter(),
        'polygon_points': [],
        'polygon_areas': [],
        'annotations_per_image': [],
        'valid_annotations': 0,
        'invalid_annotations': 0
    }
    
    # Group annotations by image
    img_to_anns = defaultdict(list)
    for ann in data['annotations']:
        img_to_anns[ann['image_id']].append(ann)
    
    # Validate each annotation
    print("\nValidating annotations...")
    for ann in tqdm(data['annotations'], desc="Processing"):
        img_id = ann['image_id']
        
        if img_id not in img_info:
            stats['validation_errors']['image_not_found'] += 1
            continue
        
        img = img_info[img_id]
        img_width = img['width']
        img_height = img['height']
        
        # Check for segmentation
        if 'segmentation' not in ann or not ann['segmentation']:
            stats['validation_errors']['no_segmentation'] += 1
            stats['invalid_annotations'] += 1
            continue
        
        # Validate polygon
        is_valid, error_type = validate_polygon(
            ann['segmentation'], img_width, img_height
        )
        
        if not is_valid:
            stats['validation_errors'][error_type] += 1
            stats['invalid_annotations'] += 1
            continue
        
        # Valid annotation
        stats['valid_annotations'] += 1
        stats['images_with_annotations'].add(img_id)
        stats['class_distribution'][ann['category_id']] += 1
        
        # Calculate statistics
        polygon = ann['segmentation'][0]
        num_points = len(polygon) // 2
        stats['polygon_points'].append(num_points)
        
        area = calculate_polygon_area(ann['segmentation'], img_width, img_height)
        stats['polygon_areas'].append(area)
    
    # Calculate annotations per image
    for anns in img_to_anns.values():
        stats['annotations_per_image'].append(len(anns))
    
    # Print results
    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"\nImages:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Images with valid annotations: {len(stats['images_with_annotations'])}")
    print(f"  Images without annotations: {stats['total_images'] - len(stats['images_with_annotations'])}")
    
    print(f"\nAnnotations:")
    print(f"  Total annotations: {stats['total_annotations']}")
    print(f"  Valid annotations: {stats['valid_annotations']}")
    print(f"  Invalid annotations: {stats['invalid_annotations']}")
    
    if stats['total_annotations'] > 0:
        valid_pct = (stats['valid_annotations'] / stats['total_annotations']) * 100
        print(f"  Validation rate: {valid_pct:.1f}%")
    
    print(f"\nValidation Errors:")
    for error_type, count in sorted(stats['validation_errors'].items(), 
                                   key=lambda x: x[1], reverse=True):
        print(f"  {error_type}: {count}")
    
    # Polygon statistics
    if stats['polygon_points']:
        print(f"\nPolygon Statistics:")
        print(f"  Points per polygon:")
        print(f"    Mean: {np.mean(stats['polygon_points']):.1f}")
        print(f"    Median: {np.median(stats['polygon_points']):.1f}")
        print(f"    Min: {np.min(stats['polygon_points'])}")
        print(f"    Max: {np.max(stats['polygon_points'])}")
        print(f"    Std: {np.std(stats['polygon_points']):.1f}")
    
    if stats['polygon_areas']:
        print(f"\n  Normalized polygon area (% of image):")
        areas_pct = [a * 100 for a in stats['polygon_areas']]
        print(f"    Mean: {np.mean(areas_pct):.2f}%")
        print(f"    Median: {np.median(areas_pct):.2f}%")
        print(f"    Min: {np.min(areas_pct):.4f}%")
        print(f"    Max: {np.max(areas_pct):.2f}%")
        print(f"    Std: {np.std(areas_pct):.2f}%")
        
        # Small object detection
        small_objects = sum(1 for a in areas_pct if a < 1.0)
        tiny_objects = sum(1 for a in areas_pct if a < 0.1)
        print(f"\n  Small objects (< 1% of image): {small_objects} ({small_objects/len(areas_pct)*100:.1f}%)")
        print(f"  Tiny objects (< 0.1% of image): {tiny_objects} ({tiny_objects/len(areas_pct)*100:.1f}%)")
    
    if stats['annotations_per_image']:
        print(f"\n  Annotations per image:")
        print(f"    Mean: {np.mean(stats['annotations_per_image']):.1f}")
        print(f"    Median: {np.median(stats['annotations_per_image']):.1f}")
        print(f"    Min: {np.min(stats['annotations_per_image'])}")
        print(f"    Max: {np.max(stats['annotations_per_image'])}")
    
    # Class distribution
    print(f"\nClass Distribution (top 10):")
    sorted_classes = sorted(stats['class_distribution'].items(), 
                          key=lambda x: x[1], reverse=True)
    for cat_id, count in sorted_classes[:10]:
        cat_name = cat_info.get(cat_id, f"Unknown_{cat_id}")
        pct = (count / stats['valid_annotations']) * 100 if stats['valid_annotations'] > 0 else 0
        print(f"  {cat_name:30s}: {count:5d} ({pct:5.1f}%)")
    
    if len(sorted_classes) > 10:
        print(f"  ... and {len(sorted_classes) - 10} more classes")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Validate COCO dataset for instance segmentation'
    )
    parser.add_argument('--processed-dir', type=str, default='./data/processed',
                      help='Path to processed COCO dataset')
    
    args = parser.parse_args()
    
    processed_dir = Path(args.processed_dir)
    
    print("="*60)
    print("COCO DATASET VALIDATION FOR INSTANCE SEGMENTATION")
    print("="*60)
    print(f"Dataset: {processed_dir}")
    print("="*60)
    
    # Analyze each split
    splits = ['train', 'val', 'test']
    all_stats = {}
    
    for split in splits:
        annotations_file = processed_dir / split / 'annotations.json'
        
        if not annotations_file.exists():
            print(f"\n⚠ Warning: {annotations_file} not found, skipping {split}")
            continue
        
        stats = analyze_segmentation_dataset(annotations_file, split)
        all_stats[split] = stats
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    total_valid = sum(s['valid_annotations'] for s in all_stats.values())
    total_invalid = sum(s['invalid_annotations'] for s in all_stats.values())
    total_all = total_valid + total_invalid
    
    print(f"\nTotal annotations: {total_all}")
    print(f"Valid annotations: {total_valid} ({total_valid/total_all*100:.1f}%)")
    print(f"Invalid annotations: {total_invalid} ({total_invalid/total_all*100:.1f}%)")
    
    print(f"\nSplit breakdown:")
    for split, stats in all_stats.items():
        print(f"  {split:5s}: {stats['valid_annotations']:5d} valid / "
              f"{stats['invalid_annotations']:5d} invalid")
    
    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    
    if total_invalid > 0:
        print(f"\n⚠ Found {total_invalid} invalid annotations")
        print("  Consider filtering these during conversion to YOLO format")
    
    # Check for small objects
    all_areas = []
    for stats in all_stats.values():
        all_areas.extend(stats['polygon_areas'])
    
    if all_areas:
        areas_pct = [a * 100 for a in all_areas]
        small_pct = sum(1 for a in areas_pct if a < 1.0) / len(areas_pct) * 100
        
        if small_pct > 30:
            print(f"\n⚠ {small_pct:.1f}% of objects are small (< 1% of image)")
            print("  Recommendations for small object detection:")
            print("    - Use larger image size (1280 instead of 640)")
            print("    - Reduce mask_ratio to 2 or 1 for higher resolution masks")
            print("    - Enable copy_paste augmentation")
            print("    - Use larger model variants (l or x)")
            print("    - Increase training epochs (500-800)")
    
    print("\nDataset is ready for conversion to YOLO segmentation format!")


if __name__ == '__main__':
    main()
