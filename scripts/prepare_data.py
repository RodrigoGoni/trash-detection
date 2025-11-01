"""
Script to prepare and split dataset
"""

import argparse
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import json


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Prepare dataset for training')
    parser.add_argument('--raw-dir', type=str, default='data/raw',
                        help='Path to raw data directory')
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--test-split', type=float, default=0.1,
                        help='Test split ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


def prepare_dataset(raw_dir: str, processed_dir: str, val_split: float, test_split: float, seed: int):
    """
    Prepare dataset by splitting into train/val/test sets

    Expected structure of raw_dir:
    raw_dir/
        class1/
            img1.jpg
            img2.jpg
        class2/
            img1.jpg

    Output structure:
    processed_dir/
        train/
            class1/
                img1.jpg
            class2/
                img1.jpg
        val/
            class1/
                img3.jpg
            class2/
                img3.jpg
        test/
            class1/
                img5.jpg
            class2/
                img5.jpg
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)

    # Create output directories
    for split in ['train', 'val', 'test']:
        (processed_path / split).mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {
        'classes': [],
        'train': {},
        'val': {},
        'test': {},
        'total': {}
    }

    # Process each class
    for class_dir in raw_path.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        stats['classes'].append(class_name)

        # Get all images
        images = list(class_dir.glob('*'))
        images = [img for img in images if img.suffix.lower() in [
            '.jpg', '.jpeg', '.png']]

        print(f"Processing class: {class_name} ({len(images)} images)")

        # Split data
        train_val, test = train_test_split(
            images, test_size=test_split, random_state=seed
        )
        train, val = train_test_split(
            train_val, test_size=val_split / (1 - test_split), random_state=seed
        )

        # Create class directories
        for split in ['train', 'val', 'test']:
            (processed_path / split / class_name).mkdir(parents=True, exist_ok=True)

        # Copy files
        for img in train:
            shutil.copy2(img, processed_path / 'train' / class_name / img.name)

        for img in val:
            shutil.copy2(img, processed_path / 'val' / class_name / img.name)

        for img in test:
            shutil.copy2(img, processed_path / 'test' / class_name / img.name)

        # Update statistics
        stats['train'][class_name] = len(train)
        stats['val'][class_name] = len(val)
        stats['test'][class_name] = len(test)
        stats['total'][class_name] = len(images)

        print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Save statistics
    with open(processed_path / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nDataset preparation complete!")
    print(f"Total train: {sum(stats['train'].values())}")
    print(f"Total val: {sum(stats['val'].values())}")
    print(f"Total test: {sum(stats['test'].values())}")
    print(f"Statistics saved to {processed_path / 'dataset_stats.json'}")


def main():
    """Main function"""
    args = parse_args()

    prepare_dataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
