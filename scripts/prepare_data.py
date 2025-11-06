"""
Prepare TACO dataset for object detection training.

Splits COCO-format dataset into train/val/test sets while maintaining annotation integrity
and handling class imbalance through stratified splitting.
"""

import argparse
import shutil
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Set
from datetime import datetime


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare TACO dataset for object detection')
    parser.add_argument('--raw-dir', type=str,
                       default='data/raw/datasets/kneroma/tacotrashdataset/versions/3/data',
                       help='Path to raw TACO data directory')
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                       help='Path to output directory')
    parser.add_argument('--version', type=str, default=None,
                       help='Dataset version name (default: auto-generated timestamp)')
    parser.add_argument('--overwrite', action='store_true', default=False,
                       help='Overwrite existing processed dataset (otherwise creates versioned folder)')
    parser.add_argument('--val-split', type=float, default=0.15,
                       help='Validation split ratio')
    parser.add_argument('--test-split', type=float, default=0.15,
                       help='Test split ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--min-annotations', type=int, default=1,
                       help='Minimum annotations per image')
    parser.add_argument('--stratify', action='store_true', default=True,
                       help='Use stratified split based on minority classes')
    parser.add_argument('--duplicate-multi-label', action='store_true', default=False,
                       help='Duplicate multi-label images with bbox occlusion for minority classes')
    parser.add_argument('--iou-threshold', type=float, default=0.3,
                       help='IoU threshold for deciding bbox occlusion (default: 0.3)')
    parser.add_argument('--minority-threshold', type=int, default=50,
                       help='Classes with fewer annotations are considered minority (default: 50)')
    parser.add_argument('--min-class-samples', type=int, default=0,
                       help='Minimum number of samples required per class (classes below this threshold will be removed, default: 0 = no filtering)')
    return parser.parse_args()


def load_coco_annotations(annotations_file: Path) -> dict:
    """Load COCO format annotations."""
    print(f"Loading: {annotations_file}")
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    print(f"  Images: {len(data['images'])}, Annotations: {len(data['annotations'])}, "
          f"Categories: {len(data['categories'])}")
    return data


def filter_images_by_annotations(data: dict, min_annotations: int = 1) -> tuple:
    """Filter images with minimum number of annotations."""
    img_ann_count = defaultdict(int)
    for ann in data['annotations']:
        img_ann_count[ann['image_id']] += 1
    
    valid_ids = {img_id for img_id, count in img_ann_count.items() if count >= min_annotations}
    filtered = [img for img in data['images'] if img['id'] in valid_ids]
    
    print(f"  Filtered: {len(data['images'])} -> {len(filtered)} images")
    return valid_ids, filtered


def filter_classes_by_sample_count(data: dict, min_samples: int) -> dict:
    """
    Remove classes with fewer than min_samples annotations from the dataset.
    
    Args:
        data: COCO format dataset dictionary
        min_samples: Minimum number of annotations required per class
    
    Returns:
        Filtered dataset with removed classes and their annotations
    """
    if min_samples <= 0:
        return data
    
    print("\n" + "="*60)
    print(f"FILTERING CLASSES (min_samples={min_samples})")
    print("="*60)
    
    # Count annotations per class
    class_counts = Counter(ann['category_id'] for ann in data['annotations'])
    
    # Identify classes to remove
    classes_to_remove = {cat_id for cat_id, count in class_counts.items() if count < min_samples}
    classes_to_keep = {cat_id for cat_id in class_counts.keys() if cat_id not in classes_to_remove}
    
    if not classes_to_remove:
        print(f"  ✓ All classes meet minimum sample requirement ({min_samples})")
        return data
    
    # Print removed classes
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    print(f"\n  Removing {len(classes_to_remove)} classes with < {min_samples} samples:")
    for cat_id in sorted(classes_to_remove):
        cat_name = cat_id_to_name.get(cat_id, f"Unknown-{cat_id}")
        count = class_counts[cat_id]
        print(f"    - {cat_name} (ID: {cat_id}): {count} samples")
    
    # Filter annotations
    original_ann_count = len(data['annotations'])
    filtered_annotations = [ann for ann in data['annotations'] 
                           if ann['category_id'] in classes_to_keep]
    
    # Filter categories
    filtered_categories = [cat for cat in data['categories'] 
                          if cat['id'] in classes_to_keep]
    
    # Remap category IDs to be sequential (0, 1, 2, ...)
    old_to_new_id = {old_id: new_id for new_id, old_id in 
                     enumerate(sorted(classes_to_keep))}
    
    # Update category IDs in categories list
    for cat in filtered_categories:
        cat['id'] = old_to_new_id[cat['id']]
    
    # Update category IDs in annotations
    for ann in filtered_annotations:
        ann['category_id'] = old_to_new_id[ann['category_id']]
    
    # Get images that still have annotations
    images_with_annotations = {ann['image_id'] for ann in filtered_annotations}
    filtered_images = [img for img in data['images'] 
                      if img['id'] in images_with_annotations]
    
    # Create filtered dataset
    filtered_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'categories': filtered_categories,
        'images': filtered_images,
        'annotations': filtered_annotations
    }
    
    print(f"\n  Summary:")
    print(f"    Categories: {len(data['categories'])} -> {len(filtered_categories)}")
    print(f"    Annotations: {original_ann_count} -> {len(filtered_annotations)}")
    print(f"    Images: {len(data['images'])} -> {len(filtered_images)}")
    print(f"    Remaining classes: {len(filtered_categories)}")
    
    return filtered_data


def analyze_class_distribution(data: dict) -> Dict:
    """Analyze class distribution and identify minority classes."""
    class_counts = Counter(ann['category_id'] for ann in data['annotations'])
    
    # Map category ID to name
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
    
    for cat_id, count in sorted_classes:
        print(f"  {cat_id_to_name[cat_id]:30s}: {count:4d} annotations")
    
    return {
        'class_counts': class_counts,
        'cat_id_to_name': cat_id_to_name,
        'total_annotations': len(data['annotations'])
    }


def identify_minority_classes(class_counts: Counter, threshold: int) -> Set[int]:
    """Identify minority classes below threshold."""
    minority = {cat_id for cat_id, count in class_counts.items() if count < threshold}
    return minority


def analyze_multi_label_images(data: dict, minority_classes: Set[int]) -> Dict:
    """Analyze images with multiple labels, especially with minority classes."""
    img_to_cats = defaultdict(set)
    img_to_anns = defaultdict(list)
    
    for ann in data['annotations']:
        img_to_cats[ann['image_id']].add(ann['category_id'])
        img_to_anns[ann['image_id']].append(ann)
    
    # Stats
    multi_label_images = {img_id: cats for img_id, cats in img_to_cats.items() if len(cats) > 1}
    multi_with_minority = {img_id: cats for img_id, cats in multi_label_images.items() 
                           if any(cat in minority_classes for cat in cats)}
    
    print("\n" + "="*60)
    print("MULTI-LABEL IMAGE ANALYSIS")
    print("="*60)
    print(f"  Total images: {len(img_to_cats)}")
    print(f"  Single-label images: {len(img_to_cats) - len(multi_label_images)}")
    print(f"  Multi-label images: {len(multi_label_images)}")
    print(f"  Multi-label with minority class: {len(multi_with_minority)}")
    
    return {
        'img_to_cats': dict(img_to_cats),
        'img_to_anns': dict(img_to_anns),
        'multi_label_images': multi_label_images,
        'multi_with_minority': multi_with_minority
    }


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two bboxes in [x, y, w, h] format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to [x1, y1, x2, y2]
    box1_x2, box1_y2 = x1 + w1, y1 + h1
    box2_x2, box2_y2 = x2 + w2, y2 + h2
    
    # Intersection
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def get_primary_class_for_image(img_id: int, img_to_anns: Dict, 
                                minority_classes: Set[int]) -> int:
    """
    Get primary class for stratification.
    Priority: minority class > most frequent class in image.
    """
    anns = img_to_anns.get(img_id, [])
    if not anns:
        return -1
    
    # Check if image has minority class
    minority_in_img = [ann['category_id'] for ann in anns 
                      if ann['category_id'] in minority_classes]
    
    if minority_in_img:
        # Return the minority class with fewest overall examples
        return min(minority_in_img)
    
    # Otherwise return most frequent class in this image
    class_counts = Counter(ann['category_id'] for ann in anns)
    return class_counts.most_common(1)[0][0]


def stratified_split(images: List[dict], img_to_anns: Dict, minority_classes: Set[int],
                    val_split: float, test_split: float, seed: int) -> Tuple:
    """
    Stratified split prioritizing minority classes.
    """
    print("\n" + "="*60)
    print("PERFORMING STRATIFIED SPLIT")
    print("="*60)
    
    # Assign primary class to each image
    image_classes = []
    for img in images:
        primary_class = get_primary_class_for_image(img['id'], img_to_anns, minority_classes)
        image_classes.append(primary_class)
    
    print(f"  Images with minority as primary: {sum(1 for c in image_classes if c in minority_classes)}")
    
    # First split: train+val vs test (stratified)
    try:
        train_val, test_images, train_val_classes, _ = train_test_split(
            images, image_classes,
            test_size=test_split,
            stratify=image_classes,
            random_state=seed
        )
    except ValueError as e:
        print(f"  Warning: Stratification failed ({e}), using random split for test")
        train_val, test_images = train_test_split(images, test_size=test_split, random_state=seed)
        train_val_classes = [get_primary_class_for_image(img['id'], img_to_anns, minority_classes) 
                            for img in train_val]
    
    # Second split: train vs val (stratified)
    val_size_adjusted = val_split / (1 - test_split)
    try:
        train_images, val_images, _, _ = train_test_split(
            train_val, train_val_classes,
            test_size=val_size_adjusted,
            stratify=train_val_classes,
            random_state=seed
        )
    except ValueError as e:
        print(f"  Warning: Stratification failed ({e}), using random split for val")
        train_images, val_images = train_test_split(train_val, test_size=val_size_adjusted, 
                                                    random_state=seed)
    
    print(f"  Split complete: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")
    
    return train_images, val_images, test_images


def create_split_annotations(data: dict, split_images: list, split_name: str) -> tuple:
    """Create COCO annotations for a specific split with remapped IDs."""
    split_image_ids = {img['id'] for img in split_images}
    split_annotations = [ann for ann in data['annotations'] if ann['image_id'] in split_image_ids]
    
    # Remap IDs to be sequential
    image_id_mapping = {old_id: new_id for new_id, old_id in 
                        enumerate(split_image_ids, start=1)}
    
    new_images = []
    for img in split_images:
        img_copy = img.copy()
        img_copy['id'] = image_id_mapping[img['id']]
        new_images.append(img_copy)
    
    new_annotations = []
    for idx, ann in enumerate(split_annotations, start=1):
        ann_copy = ann.copy()
        ann_copy['id'] = idx
        ann_copy['image_id'] = image_id_mapping[ann['image_id']]
        new_annotations.append(ann_copy)
    
    split_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'categories': data['categories'],
        'images': new_images,
        'annotations': new_annotations
    }
    
    return split_data, image_id_mapping


def copy_images(raw_dir: Path, processed_dir: Path, split_images: list, split_name: str) -> None:
    """Copy images to processed directory, maintaining batch_X subdirectory structure."""
    output_dir = processed_dir / split_name / 'images'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Copying {len(split_images)} images to {split_name}/images/...")
    for img in split_images:
        src_path = raw_dir / img['file_name']
        if src_path.exists():
            # CRITICAL: Maintain batch_X/ subdirectory structure to avoid filename collisions
            # file_name is like "batch_1/000000.jpg"
            dst_path = output_dir / img['file_name']
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
        else:
            print(f"    Warning: Source image not found: {src_path}")


def compute_statistics(data: dict, train_data: dict, val_data: dict, test_data: dict) -> dict:
    """Compute dataset statistics."""
    def count_by_category(annotations):
        counts = defaultdict(int)
        for ann in annotations:
            counts[ann['category_id']] += 1
        return dict(counts)
    
    return {
        'total': {
            'images': len(data['images']),
            'annotations': len(data['annotations']),
            'categories': len(data['categories'])
        },
        'train': {
            'images': len(train_data['images']),
            'annotations': len(train_data['annotations']),
            'annotations_per_category': count_by_category(train_data['annotations'])
        },
        'val': {
            'images': len(val_data['images']),
            'annotations': len(val_data['annotations']),
            'annotations_per_category': count_by_category(val_data['annotations'])
        },
        'test': {
            'images': len(test_data['images']),
            'annotations': len(test_data['annotations']),
            'annotations_per_category': count_by_category(test_data['annotations'])
        },
        'categories': [{'id': c['id'], 'name': c['name'], 'supercategory': c['supercategory']}
                      for c in data['categories']]
    }


def prepare_dataset(raw_dir: str, processed_dir: str, val_split: float, 
                   test_split: float, seed: int, min_annotations: int = 1,
                   stratify: bool = True, minority_threshold: int = 50,
                   duplicate_multi_label: bool = False, iou_threshold: float = 0.3,
                   version: str = None, overwrite: bool = False, min_class_samples: int = 0):
    """Prepare TACO dataset by splitting into train/val/test sets with stratification."""
    raw_path = Path(raw_dir)
    
    # Handle versioning
    if overwrite:
        processed_path = Path(processed_dir)
        print(f"OVERWRITE MODE: Will overwrite existing data in {processed_path}")
    else:
        # Create versioned output directory
        if version is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"v_{timestamp}"
        
        base_dir = Path(processed_dir).parent
        processed_path = base_dir / f"processed_{version}"
        
        if processed_path.exists():
            print(f"WARNING: Version directory already exists: {processed_path}")
            response = input("Continue and overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
        
        print(f"Creating versioned dataset: {processed_path}")
    
    # Create symlink to 'processed' pointing to latest version (only if not overwrite)
    if not overwrite:
        symlink_path = Path(processed_dir)
        if symlink_path.is_symlink() or symlink_path.is_dir():
            # Backup old symlink/dir
            if symlink_path.is_symlink():
                old_target = symlink_path.resolve()
                print(f"Previous version was: {old_target}")
                symlink_path.unlink()
            elif symlink_path.is_dir() and not symlink_path.is_symlink():
                # Rename existing directory
                backup_name = f"processed_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                backup_path = symlink_path.parent / backup_name
                print(f"Backing up existing processed/ to {backup_name}")
                symlink_path.rename(backup_path)
        
        # Create new symlink
        try:
            symlink_path.symlink_to(processed_path.name)
            print(f"Created symlink: {symlink_path} -> {processed_path.name}")
        except Exception as e:
            print(f"Warning: Could not create symlink: {e}")
    
    print("="*60)
    print("TACO Dataset Preparation (WITH STRATIFICATION)")
    print("="*60)
    print(f"Output: {processed_path}")
    print(f"Version: {version if version else 'overwrite'}")
    print("="*60)
    
    # Load annotations
    annotations_file = raw_path / 'annotations.json'
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotations not found: {annotations_file}")
    
    data = load_coco_annotations(annotations_file)
    
    # Filter classes with insufficient samples (if enabled)
    if min_class_samples > 0:
        data = filter_classes_by_sample_count(data, min_class_samples)
    
    valid_image_ids, filtered_images = filter_images_by_annotations(data, min_annotations)
    
    # Analyze class distribution
    class_info = analyze_class_distribution(data)
    minority_classes = identify_minority_classes(class_info['class_counts'], minority_threshold)
    
    print(f"\n  Minority classes (< {minority_threshold} annotations): {len(minority_classes)}")
    for cat_id in sorted(minority_classes):
        cat_name = class_info['cat_id_to_name'][cat_id]
        count = class_info['class_counts'][cat_id]
        print(f"    - {cat_name} ({cat_id}): {count} annotations")
    
    # Analyze multi-label images
    multi_info = analyze_multi_label_images(data, minority_classes)
    
    # Split dataset (stratified or random)
    print(f"\nSplitting (val={val_split}, test={test_split}, seed={seed}, stratify={stratify})...")
    
    if stratify:
        train_images, val_images, test_images = stratified_split(
            filtered_images, multi_info['img_to_anns'], minority_classes,
            val_split, test_split, seed
        )
    else:
        # Fallback to random split
        train_val, test_images = train_test_split(filtered_images, test_size=test_split, 
                                                  random_state=seed)
        val_size_adjusted = val_split / (1 - test_split)
        train_images, val_images = train_test_split(train_val, test_size=val_size_adjusted, 
                                                    random_state=seed)
    
    print(f"  Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    
    # Create split annotations
    print("\nCreating annotations...")
    train_data, _ = create_split_annotations(data, train_images, 'train')
    val_data, _ = create_split_annotations(data, val_images, 'val')
    test_data, _ = create_split_annotations(data, test_images, 'test')
    
    # Analyze class distribution in splits
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION PER SPLIT")
    print("="*60)
    for split_name, split_data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        split_class_counts = Counter(ann['category_id'] for ann in split_data['annotations'])
        minority_in_split = sum(split_class_counts[cat_id] for cat_id in minority_classes 
                               if cat_id in split_class_counts)
        print(f"\n{split_name}:")
        print(f"  Total annotations: {len(split_data['annotations'])}")
        print(f"  Minority class annotations: {minority_in_split}")
        print(f"  Classes present: {len(split_class_counts)}/{len(class_info['class_counts'])}")
        
        # Show minority class distribution
        print(f"  Minority class breakdown:")
        for cat_id in sorted(minority_classes):
            count = split_class_counts.get(cat_id, 0)
            if count > 0:
                cat_name = class_info['cat_id_to_name'][cat_id]
                print(f"    - {cat_name}: {count}")
    
    # Save annotations
    print("\nSaving...")
    for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        split_dir = processed_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        with open(split_dir / 'annotations.json', 'w') as f:
            json.dump(split_data, f, indent=2)
    
    # Copy images
    print("\nCopying images...")
    copy_images(raw_path, processed_path, train_images, 'train')
    copy_images(raw_path, processed_path, val_images, 'val')
    copy_images(raw_path, processed_path, test_images, 'test')
    
    # Save statistics
    stats = compute_statistics(data, train_data, val_data, test_data)
    stats['minority_classes'] = {
        'threshold': minority_threshold,
        'classes': {class_info['cat_id_to_name'][cat_id]: class_info['class_counts'][cat_id] 
                   for cat_id in sorted(minority_classes)}
    }
    stats['version_info'] = {
        'version': version if version else 'overwrite',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'val_split': val_split,
            'test_split': test_split,
            'seed': seed,
            'min_annotations': min_annotations,
            'stratify': stratify,
            'minority_threshold': minority_threshold,
            'duplicate_multi_label': duplicate_multi_label,
            'iou_threshold': iou_threshold,
            'min_class_samples': min_class_samples
        }
    }
    with open(processed_path / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"Version: {stats['version_info']['version']}")
    print(f"Output: {processed_path}")
    print(f"Train: {stats['train']['images']} images ({stats['train']['annotations']} annotations)")
    print(f"Val:   {stats['val']['images']} images ({stats['val']['annotations']} annotations)")
    print(f"Test:  {stats['test']['images']} images ({stats['test']['annotations']} annotations)")
    print(f"Total classes: {len(data['categories'])}")
    if min_class_samples > 0:
        print(f"Min class samples: {min_class_samples}")
    print(f"Minority classes: {len(minority_classes)} (< {minority_threshold} annotations)")
    print(f"Stratified split: {stratify}")
    if not overwrite:
        print(f"Symlink 'processed' points to: {processed_path.name}")
    print("="*60)


def main():
    """Main function."""
    args = parse_args()
    prepare_dataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        min_annotations=args.min_annotations,
        stratify=args.stratify,
        minority_threshold=args.minority_threshold,
        duplicate_multi_label=args.duplicate_multi_label,
        iou_threshold=args.iou_threshold,
        version=args.version,
        overwrite=args.overwrite,
        min_class_samples=args.min_class_samples
    )


if __name__ == '__main__':
    main()
