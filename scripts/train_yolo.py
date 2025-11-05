import sys
import json
import argparse
from pathlib import Path

import yaml
import torch
import mlflow

sys.path.append(str(Path(__file__).parent.parent))

from src.models.detector import YOLOv11Detector
from src.training.class_weights import compute_class_weights, load_class_counts_from_annotations, print_class_weights_summary
from src.training.losses import get_loss_function
from src.utils.system_info import get_system_info
from src.utils.mlflow_utils import (
    setup_mlflow,
    get_dataset_info,
    log_yolo_training_config,
    log_yolo_results,
    log_yolo_artifacts,
    log_yolo_training_summary
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv11 model')
    parser.add_argument('--config', type=str, default='config/train_config.yaml')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_weights(config):
    loss_config = config['training'].get('loss', {})
    use_class_weights = loss_config.get('use_class_weights', True)
    
    if not use_class_weights:
        return None, None
    
    print("\n" + "="*60)
    print("COMPUTING CLASS WEIGHTS")
    print("="*60)
    
    train_ann_path = Path(config['data']['processed_dir']) / 'train' / 'annotations.json'
    with open(train_ann_path) as f:
        train_data = json.load(f)
    
    class_counts = load_class_counts_from_annotations(train_data['annotations'])
    num_classes = config['model']['num_classes']
    weight_method = loss_config.get('class_weight_method', 'effective')
    beta = loss_config.get('beta', 0.9999)
    
    print(f"Method: {weight_method}")
    print(f"Beta: {beta}")
    print(f"Total samples: {len(train_data['annotations'])}")
    
    class_weights = compute_class_weights(
        class_counts=class_counts,
        num_classes=num_classes,
        method=weight_method,
        beta=beta
    )
    
    class_names = {cat['id']: cat['name'] for cat in train_data['categories']}
    print_class_weights_summary(class_weights, class_counts, class_names, top_k=10)
    
    loss_type = loss_config.get('type', 'cb_focal')
    print(f"\nCreating {loss_type.upper()} loss function")
    
    custom_loss_fn = get_loss_function(
        config={'loss': loss_config},
        class_counts=class_counts,
        num_classes=num_classes
    )
    
    return custom_loss_fn, class_weights


def main():
    args = parse_args()
    config = load_config(args.config)
    
    if args.data_dir:
        config['data']['processed_dir'] = args.data_dir
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    custom_loss_fn, class_weights = compute_weights(config)
    
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    
    model_size = config['model'].get('model_size', 's')
    num_classes_yolo = config['model']['num_classes'] - 1
    
    model = YOLOv11Detector(
        num_classes=num_classes_yolo,
        model_size=model_size,
        pretrained=config['model'].get('pretrained', True),
        custom_loss_fn=custom_loss_fn,
        class_weights=class_weights,
        img_size=config['data']['img_size']
    )
    
    print(f"YOLOv11-{model_size.upper()} model created")
    
    yolo_data_yaml = Path('./data/yolo/data.yaml')
    
    setup_mlflow(config)
    system_info = get_system_info()
    dataset_info = get_dataset_info(config['data']['processed_dir'])
    
    dataset_version = dataset_info.get('dataset_hash', 'unknown')
    run_name = f"yolov11{model_size}_{dataset_version}"
    
    with mlflow.start_run(run_name=run_name):
        print("\n" + "="*60)
        print("MLFLOW LOGGING SETUP")
        print("="*60)
        
        log_yolo_training_config(
            config=config,
            system_info=system_info,
            dataset_info=dataset_info,
            dataset_version=dataset_version,
            device=device
        )
        print("Logged training configuration")
        
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        patience = config['training'].get('patience', 100)
        
        results = model.train_yolo(
            data_yaml=str(yolo_data_yaml),
            epochs=config['training']['num_epochs'],
            batch_size=config['data']['batch_size'],
            optimizer_config=config['training']['optimizer'],
            scheduler_config=config['training']['scheduler'],
            device=str(device),
            project='./runs/detect',
            name=config['experiment']['name'],
            patience=patience
        )
        
        print("\n" + "="*60)
        print("LOGGING RESULTS")
        print("="*60)
        
        results_dir = Path(f"./runs/detect/{config['experiment']['name']}")
        
        final_metrics, best_metrics = log_yolo_results(results_dir, config)
        log_yolo_artifacts(results_dir)
        log_yolo_training_summary(results_dir, config, final_metrics, best_metrics)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Results: {results_dir}")
        print(f"MLflow run: {mlflow.active_run().info.run_id}")
        print(f"Patience: {patience}")
        
        if best_metrics:
            print("\nBest Metrics:")
            for key, value in best_metrics.items():
                if 'epoch' not in key:
                    print(f"  {key}: {value:.4f}")
        
        if final_metrics:
            print("\nFinal Metrics:")
            for key, value in final_metrics.items():
                print(f"  {key}: {value:.4f}")


if __name__ == '__main__':
    main()
