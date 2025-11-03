"""
Prepare TACO dataset for object detection training.

Splits COCO-format dataset into train/val/test sets while maintaining annotation integrity.
"""

import argparse
import shutil
import json
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare TACO dataset for object detection')
    parser.add_argument('--raw-dir', type=str,
                       default='data/raw/datasets/kneroma/tacotrashdataset/versions/3/data',
                       help='Path to raw TACO data directory')
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                       help='Path to output directory')
    parser.add_argument('--val-split', type=float, default=0.15,
                       help='Validation split ratio')
    parser.add_argument('--test-split', type=float, default=0.15,
                       help='Test split ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--min-annotations', type=int, default=1,
                       help='Minimum annotations per image')
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


def split_dataset(images: list, val_split: float, test_split: float, seed: int) -> tuple:
    """Split images into train/val/test sets."""
    train_val, test_images = train_test_split(images, test_size=test_split, random_state=seed)
    val_size_adjusted = val_split / (1 - test_split)
    train_images, val_images = train_test_split(train_val, test_size=val_size_adjusted, 
                                                 random_state=seed)
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
                   test_split: float, seed: int, min_annotations: int = 1):
    """Prepare TACO dataset by splitting into train/val/test sets."""
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    
    print("="*60)
    print("TACO Dataset Preparation")
    print("="*60)
    
    # Load annotations
    annotations_file = raw_path / 'annotations.json'
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotations not found: {annotations_file}")
    
    data = load_coco_annotations(annotations_file)
    valid_image_ids, filtered_images = filter_images_by_annotations(data, min_annotations)
    
    # Split dataset
    print(f"\nSplitting (val={val_split}, test={test_split}, seed={seed})...")
    train_images, val_images, test_images = split_dataset(filtered_images, val_split, 
                                                           test_split, seed)
    print(f"  Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    
    # Create split annotations
    print("\nCreating annotations...")
    train_data, _ = create_split_annotations(data, train_images, 'train')
    val_data, _ = create_split_annotations(data, val_images, 'val')
    test_data, _ = create_split_annotations(data, test_images, 'test')
    
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
    with open(processed_path / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"Train: {stats['train']['images']} images ({stats['train']['annotations']} annotations)")
    print(f"Val:   {stats['val']['images']} images ({stats['val']['annotations']} annotations)")
    print(f"Test:  {stats['test']['images']} images ({stats['test']['annotations']} annotations)")
    print(f"Output: {processed_path}")
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
        min_annotations=args.min_annotations
    )


if __name__ == '__main__':
    main()
