import sys
import json
import argparse
from pathlib import Path

import yaml
import torch
import mlflow
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.models.detector import YOLOv11Detector
from src.utils.system_info import get_system_info
from src.utils.mlflow_utils import (
    setup_mlflow,
    get_dataset_info,
    log_yolo_training_config,
    log_yolo_results,
    log_yolo_artifacts,
    log_yolo_training_summary
)
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv11 model')
    parser.add_argument('--config', type=str, default='config/train_config.yaml')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--skip-test', action='store_true', 
                        help='Skip test evaluation after training')
    parser.add_argument('--test-conf', type=float, default=0.25,
                        help='Confidence threshold for test evaluation')
    parser.add_argument('--test-iou', type=float, default=0.45,
                        help='IoU threshold for test evaluation')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_test_evaluation(best_weights_path: Path, training_dir: Path, config: dict, 
                       device: torch.device, conf_threshold: float, iou_threshold: float):
    """
    Run test evaluation after training using the best model weights.
    Results are saved in the same training directory under a 'test' subfolder.
    """
    print("\n" + "="*60)
    print("STARTING TEST EVALUATION")
    print("="*60)
    
    yolo_data_yaml = Path('./data/yolo/data.yaml')
    
    if not best_weights_path.exists():
        print(f"⚠ Warning: Best weights not found at {best_weights_path}")
        print("Skipping test evaluation")
        return
    
    # Load the best model
    print(f"Loading best model from: {best_weights_path.name}")
    model = YOLO(str(best_weights_path))
    
    # Save test results in the same training directory
    test_results_dir = training_dir / 'test_results'
    
    print(f"\nTest Configuration:")
    print(f"  Split: test")
    print(f"  Confidence threshold: {conf_threshold}")
    print(f"  IoU threshold: {iou_threshold}")
    print(f"  Batch size: {config['data']['batch_size']}")
    print(f"  Image size: {config['data']['img_size']}")
    print(f"  Save directory: {test_results_dir}")
    
    # Run validation on test set - save in training directory
    print("\nRunning validation on test set...")
    metrics = model.val(
        data=str(yolo_data_yaml),
        split='test',
        batch=config['data']['batch_size'],
        imgsz=config['data']['img_size'],
        conf=conf_threshold,
        iou=iou_threshold,
        device=str(device),
        project=str(training_dir),
        name='test_results',
        plots=True,
        save_json=True,
        visualize=True
    )
    
    # Extract metrics
    metrics_dict = {}
    if hasattr(metrics, 'results_dict'):
        metrics_dict = metrics.results_dict
    elif hasattr(metrics, 'box'):
        if hasattr(metrics.box, 'map50'):
            metrics_dict['metrics/mAP50(B)'] = metrics.box.map50
        if hasattr(metrics.box, 'map'):
            metrics_dict['metrics/mAP50-95(B)'] = metrics.box.map
        if hasattr(metrics.box, 'mp'):
            metrics_dict['metrics/precision(B)'] = metrics.box.mp
        if hasattr(metrics.box, 'mr'):
            metrics_dict['metrics/recall(B)'] = metrics.box.mr
    
    # Log test metrics to MLflow
    print("\n" + "="*60)
    print("LOGGING TEST RESULTS TO MLFLOW")
    print("="*60)
    
    # Log main test metrics
    if 'metrics/mAP50(B)' in metrics_dict:
        mlflow.log_metric('test_mAP50', float(metrics_dict['metrics/mAP50(B)']))
        print(f"✓ test_mAP50: {metrics_dict['metrics/mAP50(B)']:.4f}")
    
    if 'metrics/mAP50-95(B)' in metrics_dict:
        mlflow.log_metric('test_mAP50_95', float(metrics_dict['metrics/mAP50-95(B)']))
        print(f"✓ test_mAP50_95: {metrics_dict['metrics/mAP50-95(B)']:.4f}")
    
    if 'metrics/precision(B)' in metrics_dict:
        mlflow.log_metric('test_precision', float(metrics_dict['metrics/precision(B)']))
        print(f"✓ test_precision: {metrics_dict['metrics/precision(B)']:.4f}")
    
    if 'metrics/recall(B)' in metrics_dict:
        mlflow.log_metric('test_recall', float(metrics_dict['metrics/recall(B)']))
        print(f"✓ test_recall: {metrics_dict['metrics/recall(B)']:.4f}")
    
    # Log test losses if available
    results_csv = test_results_dir / 'results.csv'
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        
        if 'val/box_loss' in df.columns:
            mlflow.log_metric('test_box_loss', float(df['val/box_loss'].iloc[0]))
        if 'val/cls_loss' in df.columns:
            mlflow.log_metric('test_cls_loss', float(df['val/cls_loss'].iloc[0]))
        if 'val/dfl_loss' in df.columns:
            mlflow.log_metric('test_dfl_loss', float(df['val/dfl_loss'].iloc[0]))
        
        if all(k in df.columns for k in ['val/box_loss', 'val/cls_loss', 'val/dfl_loss']):
            total_loss = float(df['val/box_loss'].iloc[0] + df['val/cls_loss'].iloc[0] + df['val/dfl_loss'].iloc[0])
            mlflow.log_metric('test_total_loss', total_loss)
            print(f"✓ test_total_loss: {total_loss:.4f}")
    
    # Log test artifacts
    print("\nLogging test artifacts...")
    artifact_count = 0
    
    # Log confusion matrices
    confusion_matrix = test_results_dir / 'confusion_matrix.png'
    if confusion_matrix.exists():
        mlflow.log_artifact(str(confusion_matrix), artifact_path="test/plots")
        artifact_count += 1
    
    confusion_matrix_normalized = test_results_dir / 'confusion_matrix_normalized.png'
    if confusion_matrix_normalized.exists():
        mlflow.log_artifact(str(confusion_matrix_normalized), artifact_path="test/plots")
        artifact_count += 1
    
    # Log curves
    for curve_name in ['PR_curve.png', 'F1_curve.png', 'P_curve.png', 'R_curve.png']:
        curve_file = test_results_dir / curve_name
        if curve_file.exists():
            mlflow.log_artifact(str(curve_file), artifact_path="test/plots")
            artifact_count += 1
    
    # Log prediction examples (first 3 batches)
    val_batch_labels = list(test_results_dir.glob('val_batch*_labels.jpg'))[:3]
    val_batch_preds = list(test_results_dir.glob('val_batch*_pred.jpg'))[:3]
    
    for img_file in val_batch_labels + val_batch_preds:
        mlflow.log_artifact(str(img_file), artifact_path="test/predictions")
        artifact_count += 1
    
    # Log results files
    if results_csv.exists():
        mlflow.log_artifact(str(results_csv), artifact_path="test/results")
        artifact_count += 1
    
    predictions_json = test_results_dir / 'predictions.json'
    if predictions_json.exists():
        mlflow.log_artifact(str(predictions_json), artifact_path="test/results")
        artifact_count += 1
    
    print(f"✓ Logged {artifact_count} test artifacts")
    
    # Create and log test summary
    test_summary = {
        'task': 'test',
        'split': 'test',
        'weights': str(best_weights_path),
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'metrics': {
            'mAP50': float(metrics_dict.get('metrics/mAP50(B)', 0)),
            'mAP50_95': float(metrics_dict.get('metrics/mAP50-95(B)', 0)),
            'precision': float(metrics_dict.get('metrics/precision(B)', 0)),
            'recall': float(metrics_dict.get('metrics/recall(B)', 0))
        },
        'results_dir': str(test_results_dir.resolve())
    }
    
    test_summary_file = test_results_dir / 'test_summary.json'
    with open(test_summary_file, 'w') as f:
        json.dump(test_summary, f, indent=2)
    
    mlflow.log_artifact(str(test_summary_file), artifact_path="test/results")
    mlflow.log_dict(test_summary, "test/test_summary.json")
    
    # Add test tags
    mlflow.set_tag('test_completed', 'true')
    mlflow.set_tag('test_split', 'test')
    
    print("\n" + "="*60)
    print("TEST EVALUATION COMPLETE")
    print("="*60)
    print(f"Test results: {test_results_dir}")
    print("\nTest Metrics Summary:")
    for key, value in test_summary['metrics'].items():
        print(f"  {key}: {value:.4f}")


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
    
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    
    model_size = config['model'].get('model_size', 's')
    num_classes_yolo = config['model']['num_classes'] - 1
    
    model = YOLOv11Detector(
        num_classes=num_classes_yolo,
        model_size=model_size,
        pretrained=config['model'].get('pretrained', True),
        img_size=config['data']['img_size']
    )
    
    print(f"YOLOv11-{model_size.upper()} model created")
    
    yolo_data_yaml = Path('./data/yolo/data.yaml')
    
    setup_mlflow(config)
    system_info = get_system_info()
    dataset_info = get_dataset_info(config['data']['processed_dir'])
    
    dataset_version = dataset_info.get('dataset_hash', 'unknown')
    run_name = f"yolov11{model_size}_{dataset_version}"
    
    # Get YOLO-specific configuration
    yolo_config = config.get('yolo_training', {})
    
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
        
        results = model.train_yolo(
            data_yaml=str(yolo_data_yaml),
            epochs=config['training']['num_epochs'],
            batch_size=config['data']['batch_size'],
            device=str(device),
            project='./runs/detect',
            name=config['experiment']['name'],
            yolo_config=yolo_config
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
        
        if best_metrics:
            print("\nBest Metrics:")
            for key, value in best_metrics.items():
                if 'epoch' not in key:
                    print(f"  {key}: {value:.4f}")
        
        if final_metrics:
            print("\nFinal Metrics:")
            for key, value in final_metrics.items():
                print(f"  {key}: {value:.4f}")
        
        # Run test evaluation if not skipped
        if not args.skip_test:
            best_weights = results_dir / 'weights' / 'best.pt'
            run_test_evaluation(
                best_weights_path=best_weights,
                training_dir=results_dir,
                config=config,
                device=device,
                conf_threshold=args.test_conf,
                iou_threshold=args.test_iou
            )
        else:
            print("\n⚠ Test evaluation skipped (--skip-test flag provided)")


if __name__ == '__main__':
    main()
