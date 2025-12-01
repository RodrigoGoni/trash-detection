"""
Train YOLOv11 Instance Segmentation model for trash detection.

This script provides complete training pipeline with:
- MLflow experiment tracking
- Resume capability from checkpoints
- Test evaluation after training
- Comprehensive logging and visualization
"""

import argparse
import yaml
import torch
import mlflow
import sys
import json
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.detector import YOLOv11SegmentationDetector
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
    parser = argparse.ArgumentParser(description='Train YOLOv11 Instance Segmentation model')
    parser.add_argument('--config', type=str, default='config/train_config_yolo11_segmentation.yaml')
    parser.add_argument('--data-dir', type=str, default='data/yolo_seg')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--resume', action='store_true', dest='resume_override',
                        help='Force resume from checkpoint (overrides config)')
    parser.add_argument('--no-resume', action='store_false', dest='resume_override',
                        help='Force fresh training (overrides config)')
    parser.set_defaults(resume_override=None)  # None means use config value
    parser.add_argument('--skip-test', action='store_true', 
                        help='Skip test evaluation after training')
    parser.add_argument('--test-conf', type=float, default=0.3,
                        help='Confidence threshold for test evaluation')
    parser.add_argument('--test-iou', type=float, default=0.55,
                        help='IoU threshold for test evaluation (box and mask)')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_test_evaluation(best_weights_path: Path, training_dir: Path, config: dict, 
                       device: torch.device, conf_threshold: float, iou_threshold: float):
    """
    Run test evaluation after training using the best model weights.
    
    For segmentation models, this evaluates both bounding box and mask metrics.
    Results are saved in the same training directory under a 'test' subfolder.
    
    Args:
        best_weights_path: Path to best.pt weights
        training_dir: Directory containing training results
        config: Training configuration dict
        device: Device to use for evaluation
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS and evaluation
    """
    print("\n" + "="*60)
    print("STARTING TEST EVALUATION (INSTANCE SEGMENTATION)")
    print("="*60)
    
    yolo_data_yaml = Path('./data/yolo_seg/data.yaml')
    
    if not best_weights_path.exists():
        print(f"Warning: Best weights not found at {best_weights_path}")
        print("Skipping test evaluation")
        return
    
    # Load the best model
    print(f"Loading best segmentation model from: {best_weights_path.name}")
    model = YOLO(str(best_weights_path))
    
    # Save test results in the same training directory
    test_results_dir = training_dir / 'test_results'
    
    print(f"\nTest Configuration:")
    print(f"  Split: test")
    print(f"  Task: instance segmentation")
    print(f"  Confidence threshold: {conf_threshold}")
    print(f"  IoU threshold: {iou_threshold}")
    print(f"  Batch size: {config['data']['batch_size']}")
    print(f"  Image size: {config['data']['img_size']}")
    print(f"  Save directory: {test_results_dir}")
    
    # Run validation on test set
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
        save_json=True
    )
    
    # Extract metrics for both boxes and masks
    metrics_dict = {}
    if hasattr(metrics, 'results_dict'):
        metrics_dict = metrics.results_dict
    else:
        # Box metrics
        if hasattr(metrics, 'box'):
            if hasattr(metrics.box, 'map50'):
                metrics_dict['metrics/mAP50(B)'] = metrics.box.map50
            if hasattr(metrics.box, 'map'):
                metrics_dict['metrics/mAP50-95(B)'] = metrics.box.map
            if hasattr(metrics.box, 'mp'):
                metrics_dict['metrics/precision(B)'] = metrics.box.mp
            if hasattr(metrics.box, 'mr'):
                metrics_dict['metrics/recall(B)'] = metrics.box.mr
        
        # Mask metrics (specific to segmentation)
        if hasattr(metrics, 'seg'):
            if hasattr(metrics.seg, 'map50'):
                metrics_dict['metrics/mAP50(M)'] = metrics.seg.map50
            if hasattr(metrics.seg, 'map'):
                metrics_dict['metrics/mAP50-95(M)'] = metrics.seg.map
            if hasattr(metrics.seg, 'mp'):
                metrics_dict['metrics/precision(M)'] = metrics.seg.mp
            if hasattr(metrics.seg, 'mr'):
                metrics_dict['metrics/recall(M)'] = metrics.seg.mr
    
    # Log test metrics to MLflow
    print("\n" + "="*60)
    print("LOGGING TEST RESULTS TO MLFLOW")
    print("="*60)
    
    # Box metrics
    if 'metrics/mAP50(B)' in metrics_dict:
        mlflow.log_metric('test_box_mAP50', float(metrics_dict['metrics/mAP50(B)']))
        print(f"test_box_mAP50: {metrics_dict['metrics/mAP50(B)']:.4f}")
    
    if 'metrics/mAP50-95(B)' in metrics_dict:
        mlflow.log_metric('test_box_mAP50_95', float(metrics_dict['metrics/mAP50-95(B)']))
        print(f"test_box_mAP50_95: {metrics_dict['metrics/mAP50-95(B)']:.4f}")
    
    if 'metrics/precision(B)' in metrics_dict:
        mlflow.log_metric('test_box_precision', float(metrics_dict['metrics/precision(B)']))
        print(f"test_box_precision: {metrics_dict['metrics/precision(B)']:.4f}")
    
    if 'metrics/recall(B)' in metrics_dict:
        mlflow.log_metric('test_box_recall', float(metrics_dict['metrics/recall(B)']))
        print(f"test_box_recall: {metrics_dict['metrics/recall(B)']:.4f}")
    
    # Mask metrics (segmentation-specific)
    if 'metrics/mAP50(M)' in metrics_dict:
        mlflow.log_metric('test_mask_mAP50', float(metrics_dict['metrics/mAP50(M)']))
        print(f"test_mask_mAP50: {metrics_dict['metrics/mAP50(M)']:.4f}")
    
    if 'metrics/mAP50-95(M)' in metrics_dict:
        mlflow.log_metric('test_mask_mAP50_95', float(metrics_dict['metrics/mAP50-95(M)']))
        print(f"test_mask_mAP50_95: {metrics_dict['metrics/mAP50-95(M)']:.4f}")
    
    if 'metrics/precision(M)' in metrics_dict:
        mlflow.log_metric('test_mask_precision', float(metrics_dict['metrics/precision(M)']))
        print(f"test_mask_precision: {metrics_dict['metrics/precision(M)']:.4f}")
    
    if 'metrics/recall(M)' in metrics_dict:
        mlflow.log_metric('test_mask_recall', float(metrics_dict['metrics/recall(M)']))
        print(f"test_mask_recall: {metrics_dict['metrics/recall(M)']:.4f}")
    
    # Log test losses if available
    results_csv = test_results_dir / 'results.csv'
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        
        # Box losses
        if 'val/box_loss' in df.columns:
            mlflow.log_metric('test_box_loss', float(df['val/box_loss'].iloc[0]))
        if 'val/cls_loss' in df.columns:
            mlflow.log_metric('test_cls_loss', float(df['val/cls_loss'].iloc[0]))
        if 'val/dfl_loss' in df.columns:
            mlflow.log_metric('test_dfl_loss', float(df['val/dfl_loss'].iloc[0]))
        
        # Segmentation loss (mask loss)
        if 'val/seg_loss' in df.columns:
            mlflow.log_metric('test_seg_loss', float(df['val/seg_loss'].iloc[0]))
        
        # Total loss
        loss_cols = ['val/box_loss', 'val/cls_loss', 'val/dfl_loss', 'val/seg_loss']
        if all(k in df.columns for k in loss_cols):
            total_loss = float(sum(df[col].iloc[0] for col in loss_cols))
            mlflow.log_metric('test_total_loss', total_loss)
            print(f"test_total_loss: {total_loss:.4f}")
    
    # Log test artifacts
    print("\nLogging test artifacts...")
    artifact_count = 0
    
    # Confusion matrices
    for cm_file in ['confusion_matrix.png', 'confusion_matrix_normalized.png']:
        cm_path = test_results_dir / cm_file
        if cm_path.exists():
            mlflow.log_artifact(str(cm_path), artifact_path="test/plots")
            artifact_count += 1
    
    # Curves
    for curve_name in ['PR_curve.png', 'F1_curve.png', 'P_curve.png', 'R_curve.png']:
        curve_file = test_results_dir / curve_name
        if curve_file.exists():
            mlflow.log_artifact(str(curve_file), artifact_path="test/plots")
            artifact_count += 1
    
    # Segmentation-specific plots
    for mask_plot in ['mask_F1_curve.png', 'mask_PR_curve.png', 'mask_P_curve.png', 'mask_R_curve.png']:
        mask_file = test_results_dir / mask_plot
        if mask_file.exists():
            mlflow.log_artifact(str(mask_file), artifact_path="test/plots/masks")
            artifact_count += 1
    
    # Prediction examples (first 3 batches)
    val_batch_labels = list(test_results_dir.glob('val_batch*_labels.jpg'))[:3]
    val_batch_preds = list(test_results_dir.glob('val_batch*_pred.jpg'))[:3]
    
    for img_file in val_batch_labels + val_batch_preds:
        mlflow.log_artifact(str(img_file), artifact_path="test/predictions")
        artifact_count += 1
    
    # Results files
    if results_csv.exists():
        mlflow.log_artifact(str(results_csv), artifact_path="test/results")
        artifact_count += 1
    
    predictions_json = test_results_dir / 'predictions.json'
    if predictions_json.exists():
        mlflow.log_artifact(str(predictions_json), artifact_path="test/results")
        artifact_count += 1
    
    print(f"Logged {artifact_count} test artifacts")
    
    # Create and log test summary
    test_summary = {
        'task': 'instance_segmentation',
        'split': 'test',
        'weights': str(best_weights_path),
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'metrics': {
            'box': {
                'mAP50': float(metrics_dict.get('metrics/mAP50(B)', 0)),
                'mAP50_95': float(metrics_dict.get('metrics/mAP50-95(B)', 0)),
                'precision': float(metrics_dict.get('metrics/precision(B)', 0)),
                'recall': float(metrics_dict.get('metrics/recall(B)', 0))
            },
            'mask': {
                'mAP50': float(metrics_dict.get('metrics/mAP50(M)', 0)),
                'mAP50_95': float(metrics_dict.get('metrics/mAP50-95(M)', 0)),
                'precision': float(metrics_dict.get('metrics/precision(M)', 0)),
                'recall': float(metrics_dict.get('metrics/recall(M)', 0))
            }
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
    mlflow.set_tag('test_task', 'instance_segmentation')
    
    print("\n" + "="*60)
    print("TEST EVALUATION COMPLETE")
    print("="*60)
    print(f"Test results: {test_results_dir}")
    print("\nTest Metrics Summary:")
    print("\nBox Metrics:")
    for key, value in test_summary['metrics']['box'].items():
        print(f"  {key}: {value:.4f}")
    print("\nMask Metrics:")
    for key, value in test_summary['metrics']['mask'].items():
        print(f"  {key}: {value:.4f}")


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data']['processed_dir'] = args.data_dir
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # Get resume setting from config
    yolo_config = config.get('yolo_training', {})
    resume_from_config = yolo_config.get('resume', False)
    
    # Allow command line to override config
    if args.resume_override is not None:
        should_resume = args.resume_override
        print(f"Resume overridden by command line: {should_resume}")
    else:
        should_resume = resume_from_config
    
    # Check for existing checkpoint
    results_dir = Path(f"./runs/segment/{config['experiment']['name']}")
    last_checkpoint = results_dir / 'weights' / 'last.pt'
    checkpoint_exists = last_checkpoint.exists()
    
    # Final decision: resume only if checkpoint exists AND should_resume is True
    resume_from_checkpoint = checkpoint_exists and should_resume
    
    print("\n" + "="*60)
    print("CREATING SEGMENTATION MODEL")
    print("="*60)
    print(f"Task: Instance Segmentation")
    print(f"Resume setting in config: {resume_from_config}")
    print(f"Checkpoint exists: {checkpoint_exists}")
    if checkpoint_exists:
        print(f"Checkpoint path: {last_checkpoint}")
    print(f"Will resume: {resume_from_checkpoint}")
    print("="*60)
    
    model_size = config['model'].get('model_size', 's')
    num_classes_yolo = config['model']['num_classes']
    
    if resume_from_checkpoint:
        print(f"RESUMING FROM CHECKPOINT")
        print(f"   Loading: {last_checkpoint.name}")
        model = YOLOv11SegmentationDetector(
            num_classes=num_classes_yolo,
            model_size=model_size,
            pretrained=False,
            img_size=config['data']['img_size']
        )
        # Load the checkpoint
        model.model = YOLO(str(last_checkpoint))
        print(f"Checkpoint loaded successfully")
    else:
        if checkpoint_exists and not should_resume:
            # Use the trained weights as starting point
            best_weights = results_dir / 'weights' / 'best.pt'
            if best_weights.exists():
                print(f"USING TRAINED WEIGHTS AS STARTING POINT")
                print(f"Loading best weights from previous training: {best_weights.name}")
                model = YOLOv11SegmentationDetector(
                    num_classes=num_classes_yolo,
                    model_size=model_size,
                    pretrained=False,
                    img_size=config['data']['img_size']
                )
                model.model = YOLO(str(best_weights))
            else:
                print(f"STARTING FRESH TRAINING (best weights not found)")
                model = YOLOv11SegmentationDetector(
                    num_classes=num_classes_yolo,
                    model_size=model_size,
                    pretrained=config['model'].get('pretrained', True),
                    img_size=config['data']['img_size']
                )
        else:
            print(f"STARTING FRESH TRAINING")
            model = YOLOv11SegmentationDetector(
                num_classes=num_classes_yolo,
                model_size=model_size,
                pretrained=config['model'].get('pretrained', True),
                img_size=config['data']['img_size']
            )
    
    print(f"YOLOv11-{model_size.upper()}-seg model ready")
    
    yolo_data_yaml = Path('./data/yolo_seg/data.yaml')
    
    # Setup MLflow
    setup_mlflow(config)
    system_info = get_system_info()
    dataset_info = get_dataset_info(config['data']['processed_dir'])
    
    dataset_version = dataset_info.get('dataset_hash', 'unknown')
    run_name = f"yolov11{model_size}_seg_{dataset_version}"
    
    # Set resume based on our decision
    yolo_config = config.get('yolo_training', {}).copy()
    yolo_config['resume'] = resume_from_checkpoint
    
    print(f"\nYOLO Training Config:")
    print(f"   Task: segment")
    print(f"   Resume: {yolo_config['resume']}")
    print(f"   Epochs: {config['training']['num_epochs']}")
    print(f"   Batch size: {config['data']['batch_size']}")
    print(f"   Optimizer: {yolo_config.get('optimizer', 'auto')}")
    
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
        
        # Log task and resume information
        mlflow.log_param('task', 'instance_segmentation')
        mlflow.log_param('resume_training', resume_from_checkpoint)
        if resume_from_checkpoint:
            mlflow.log_param('checkpoint_path', str(last_checkpoint))
        
        print("Logged training configuration")
        
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        if resume_from_checkpoint:
            print(f"Resuming from checkpoint: {last_checkpoint.name}")
        else:
            print(f"Starting fresh training")
        
        # Train segmentation model
        results = model.train_yolo_segmentation(
            data_yaml=str(yolo_data_yaml),
            epochs=config['training']['num_epochs'],
            batch_size=config['data']['batch_size'],
            device=str(device),
            project='./runs/segment',
            name=config['experiment']['name'],
            yolo_config=yolo_config
        )
        
        print("\n" + "="*60)
        print("LOGGING RESULTS")
        print("="*60)
        
        results_dir = Path(f"./runs/segment/{config['experiment']['name']}")
        
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
            print("\nTest evaluation skipped (--skip-test flag provided)")


if __name__ == '__main__':
    main()
