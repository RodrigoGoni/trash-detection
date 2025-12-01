import json
from pathlib import Path
from datetime import datetime
import warnings
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

warnings.filterwarnings('ignore')

IOU_THRESHOLDS = [0.3, 0.5, 0.7, 0.95]
CONF_THRESHOLDS = [0.2, 0.3, 0.35 , 0.4, 0.5]
IMG_SIZE = 640
BATCH_SIZE = 16

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'evaluation_results'
OUTPUT_DIR.mkdir(exist_ok=True)


def find_all_models():
    models = {'detection': [], 'segmentation': []}
    
    detect_dir = PROJECT_ROOT / 'runs' / 'detect'
    if detect_dir.exists():
        for model_dir in detect_dir.iterdir():
            if model_dir.is_dir():
                best_weights = model_dir / 'weights' / 'best.pt'
                if best_weights.exists():
                    models['detection'].append({
                        'name': model_dir.name,
                        'path': best_weights
                    })
    
    segment_dir = PROJECT_ROOT / 'runs' / 'segment'
    if segment_dir.exists():
        for model_dir in segment_dir.iterdir():
            if model_dir.is_dir():
                best_weights = model_dir / 'weights' / 'best.pt'
                if best_weights.exists():
                    models['segmentation'].append({
                        'name': model_dir.name,
                        'path': best_weights
                    })
    
    return models


def evaluate_model(model_path, data_yaml, model_name, task_type):
    print(f"\nEvaluando: {model_name}")
    
    model = YOLO(str(model_path))
    results = {
        'model_name': model_name,
        'task_type': task_type,
        'evaluations': []
    }
    
    for iou in IOU_THRESHOLDS:
        for conf in CONF_THRESHOLDS:
            print(f"  IoU={iou}, Conf={conf}")
            
            try:
                metrics = model.val(
                    data=str(data_yaml),
                    split='test',
                    batch=BATCH_SIZE,
                    imgsz=IMG_SIZE,
                    conf=conf,
                    iou=iou,
                    plots=False,
                    save_json=False,
                    verbose=False
                )
                
                eval_result = {
                    'iou': iou,
                    'conf': conf,
                    'metrics': {}
                }
                
                if hasattr(metrics, 'box'):
                    box = metrics.box
                    eval_result['metrics']['mAP50'] = float(box.map50) if hasattr(box, 'map50') else 0.0
                    eval_result['metrics']['mAP50_95'] = float(box.map) if hasattr(box, 'map') else 0.0
                    eval_result['metrics']['precision'] = float(box.mp) if hasattr(box, 'mp') else 0.0
                    eval_result['metrics']['recall'] = float(box.mr) if hasattr(box, 'mr') else 0.0
                
                if hasattr(metrics, 'seg'):
                    seg = metrics.seg
                    eval_result['metrics']['seg_mAP50'] = float(seg.map50) if hasattr(seg, 'map50') else 0.0
                    eval_result['metrics']['seg_mAP50_95'] = float(seg.map) if hasattr(seg, 'map') else 0.0
                
                if hasattr(metrics, 'confusion_matrix') and metrics.confusion_matrix is not None:
                    cm = metrics.confusion_matrix.matrix
                    if cm is not None:
                        cm_array = np.array(cm)
                        num_classes = cm_array.shape[0] - 1
                        
                        bg_confusions = 0
                        total = cm_array.sum()
                        
                        if cm_array.shape[1] > num_classes:
                            bg_col = cm_array[:-1, -1].sum()
                            bg_confusions += bg_col
                        
                        if cm_array.shape[0] > num_classes:
                            bg_row = cm_array[-1, :-1].sum()
                            bg_confusions += bg_row
                        
                        if total > 0:
                            eval_result['background_confusion_rate'] = float(bg_confusions / total)
                
                results['evaluations'].append(eval_result)
                
            except Exception as e:
                print(f"    Error: {str(e)}")
                continue
    
    if results['evaluations']:
        best = max(results['evaluations'], key=lambda x: x['metrics'].get('mAP50_95', 0))
        results['best'] = {
            'iou': best['iou'],
            'conf': best['conf'],
            'mAP50_95': best['metrics'].get('mAP50_95', 0),
            'mAP50': best['metrics'].get('mAP50', 0),
            'precision': best['metrics'].get('precision', 0),
            'recall': best['metrics'].get('recall', 0)
        }
    
    return results


def generate_confusion_matrix(model_path, data_yaml, iou, conf, class_names, save_path):
    print(f"\nGenerando matriz de confusion: {save_path.name}")
    
    model = YOLO(str(model_path))
    
    # Necesita plots=True para que la confusion matrix tenga datos
    metrics = model.val(
        data=str(data_yaml),
        split='test',
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        conf=conf,
        iou=iou,
        plots=True,
        verbose=False,
        save_json=False
    )
    
    if hasattr(metrics, 'confusion_matrix') and metrics.confusion_matrix is not None:
        cm_dir = save_path.parent
        
        # Matriz normalizada
        metrics.confusion_matrix.plot(
            normalize=True,
            save_dir=str(cm_dir)
        )
        
        # Renombrar archivo generado por Ultralytics
        ultralytics_file = cm_dir / 'confusion_matrix_normalized.png'
        save_path_norm = cm_dir / f"{save_path.stem}_normalized.png"
        if ultralytics_file.exists():
            if save_path_norm.exists():
                save_path_norm.unlink()
            ultralytics_file.rename(save_path_norm)
        
        # Matriz sin normalizar
        metrics.confusion_matrix.plot(
            normalize=False,
            save_dir=str(cm_dir)
        )
        
        # Renombrar archivo generado por Ultralytics
        ultralytics_file = cm_dir / 'confusion_matrix.png'
        save_path_raw = cm_dir / f"{save_path.stem}_raw.png"
        if ultralytics_file.exists():
            if save_path_raw.exists():
                save_path_raw.unlink()
            ultralytics_file.rename(save_path_raw)
        
        print(f"  Guardadas: {save_path_norm.name} y {save_path_raw.name}")


def find_common_images():
    # Datasets separados para cada tarea
    # Deteccion usa: data/yolo/test/images
    # Segmentacion usa: data/yolo_seg/test/images
    yolo_test = PROJECT_ROOT / 'data' / 'yolo' / 'test' / 'images'
    yolo_seg_test = PROJECT_ROOT / 'data' / 'yolo_seg' / 'test' / 'images'
    
    if not yolo_test.exists() or not yolo_seg_test.exists():
        return []
    
    # Buscar imagenes comunes comparando nombres de archivo
    yolo_imgs = set([f.name for f in yolo_test.glob('*.jpg')])
    yolo_seg_imgs = set([f.name for f in yolo_seg_test.glob('*.jpg')])
    
    # Interseccion: imagenes que estan en ambos datasets
    common = yolo_imgs.intersection(yolo_seg_imgs)
    
    common_list = sorted(list(common))
    sample_size = min(10, len(common_list))
    
    if sample_size > 0:
        # Seleccionar imagenes espaciadas uniformemente
        step = len(common_list) // sample_size
        sampled = [common_list[i * step] for i in range(sample_size)]
        # Retornar paths completos desde el dataset de deteccion
        return [yolo_test / img for img in sampled]
    
    return []


def run_batch_inference(models_info, common_images, output_dir):
    print(f"\nInferencia en batch: {len(common_images)} imagenes")
    
    batch_dir = output_dir / 'batch_inference'
    batch_dir.mkdir(exist_ok=True)
    
    all_models = []
    for m in models_info['detection']:
        model = YOLO(str(m['path']))
        all_models.append({'name': m['name'], 'model': model, 'type': 'detection'})
    
    for m in models_info['segmentation']:
        model = YOLO(str(m['path']))
        all_models.append({'name': m['name'], 'model': model, 'type': 'segmentation'})
    
    for idx, img_path in enumerate(common_images):
        print(f"  Procesando {idx + 1}/{len(common_images)}: {img_path.name}")
        
        num_models = len(all_models)
        cols = min(3, num_models)
        rows = (num_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
        if num_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for model_idx, model_info in enumerate(all_models):
            ax = axes[model_idx]
            
            try:
                results = model_info['model'].predict(
                    source=str(img_path),
                    conf=0.3,
                    iou=0.5,
                    verbose=False
                )[0]
                
                img_pred = results.plot()
                img_pred_rgb = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)
                
                ax.imshow(img_pred_rgb)
                ax.set_title(f"{model_info['name']}\n({model_info['type']})", fontsize=8)
                ax.axis('off')
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error', transform=ax.transAxes, ha='center', va='center')
                ax.axis('off')
        
        for i in range(num_models, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'{img_path.name}', fontsize=12)
        plt.tight_layout()
        
        save_path = batch_dir / f'comparison_{idx:03d}_{img_path.stem}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()


def create_summary(all_results, output_file):
    print(f"\nCreando reporte resumen")
    
    summary = {
        'evaluation_date': datetime.now().isoformat(),
        'config': {
            'iou_thresholds': IOU_THRESHOLDS,
            'conf_thresholds': CONF_THRESHOLDS,
            'img_size': IMG_SIZE,
            'batch_size': BATCH_SIZE
        },
        'models_evaluated': {
            'detection': len(all_results['detection']),
            'segmentation': len(all_results['segmentation'])
        },
        'detection_models': [],
        'segmentation_models': [],
        'best_models': {}
    }
    
    for result in all_results['detection']:
        model_info = {
            'name': result['model_name'],
            'best_config': result.get('best', {})
        }
        
        bg_rates = [e.get('background_confusion_rate', 0) for e in result['evaluations'] 
                   if 'background_confusion_rate' in e]
        if bg_rates:
            model_info['avg_background_confusion'] = float(np.mean(bg_rates))
        
        summary['detection_models'].append(model_info)
    
    for result in all_results['segmentation']:
        model_info = {
            'name': result['model_name'],
            'best_config': result.get('best', {})
        }
        
        bg_rates = [e.get('background_confusion_rate', 0) for e in result['evaluations'] 
                   if 'background_confusion_rate' in e]
        if bg_rates:
            model_info['avg_background_confusion'] = float(np.mean(bg_rates))
        
        summary['segmentation_models'].append(model_info)
    
    if summary['detection_models']:
        best_det = max(summary['detection_models'], 
                      key=lambda x: x.get('best_config', {}).get('mAP50_95', 0))
        summary['best_models']['detection'] = {
            'name': best_det['name'],
            'mAP50_95': best_det.get('best_config', {}).get('mAP50_95', 0),
            'iou': best_det.get('best_config', {}).get('iou', 0),
            'conf': best_det.get('best_config', {}).get('conf', 0)
        }
        
        best_det_precision = max(summary['detection_models'],
                                key=lambda x: x.get('best_config', {}).get('precision', 0))
        summary['best_models']['detection_precision'] = {
            'name': best_det_precision['name'],
            'precision': best_det_precision.get('best_config', {}).get('precision', 0),
            'iou': best_det_precision.get('best_config', {}).get('iou', 0),
            'conf': best_det_precision.get('best_config', {}).get('conf', 0)
        }
        
        best_det_recall = max(summary['detection_models'],
                             key=lambda x: x.get('best_config', {}).get('recall', 0))
        summary['best_models']['detection_recall'] = {
            'name': best_det_recall['name'],
            'recall': best_det_recall.get('best_config', {}).get('recall', 0),
            'iou': best_det_recall.get('best_config', {}).get('iou', 0),
            'conf': best_det_recall.get('best_config', {}).get('conf', 0)
        }
    
    if summary['segmentation_models']:
        best_seg = max(summary['segmentation_models'],
                      key=lambda x: x.get('best_config', {}).get('mAP50_95', 0))
        summary['best_models']['segmentation'] = {
            'name': best_seg['name'],
            'mAP50_95': best_seg.get('best_config', {}).get('mAP50_95', 0),
            'iou': best_seg.get('best_config', {}).get('iou', 0),
            'conf': best_seg.get('best_config', {}).get('conf', 0)
        }
        
        best_seg_precision = max(summary['segmentation_models'],
                                key=lambda x: x.get('best_config', {}).get('precision', 0))
        summary['best_models']['segmentation_precision'] = {
            'name': best_seg_precision['name'],
            'precision': best_seg_precision.get('best_config', {}).get('precision', 0),
            'iou': best_seg_precision.get('best_config', {}).get('iou', 0),
            'conf': best_seg_precision.get('best_config', {}).get('conf', 0)
        }
        
        best_seg_recall = max(summary['segmentation_models'],
                             key=lambda x: x.get('best_config', {}).get('recall', 0))
        summary['best_models']['segmentation_recall'] = {
            'name': best_seg_recall['name'],
            'recall': best_seg_recall.get('best_config', {}).get('recall', 0),
            'iou': best_seg_recall.get('best_config', {}).get('iou', 0),
            'conf': best_seg_recall.get('best_config', {}).get('conf', 0)
        }
    
    all_models = summary['detection_models'] + summary['segmentation_models']
    all_models_bg = [m for m in all_models if 'avg_background_confusion' in m]
    
    if all_models_bg:
        most_confused = max(all_models_bg, key=lambda x: x['avg_background_confusion'])
        least_confused = min(all_models_bg, key=lambda x: x['avg_background_confusion'])
        
        summary['background_analysis'] = {
            'most_confused': {
                'name': most_confused['name'],
                'rate': most_confused['avg_background_confusion']
            },
            'least_confused': {
                'name': least_confused['name'],
                'rate': least_confused['avg_background_confusion']
            }
        }
    
    with open(output_file, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    
    print(f"Reporte guardado: {output_file}")
    
    return summary


def print_summary(summary):
    print("\n" + "="*80)
    print("RESUMEN DE EVALUACION")
    print("="*80)
    
    print(f"\nModelos evaluados:")
    print(f"  Deteccion: {summary['models_evaluated']['detection']}")
    print(f"  Segmentacion: {summary['models_evaluated']['segmentation']}")
    
    if 'detection' in summary['best_models']:
        best = summary['best_models']['detection']
        print(f"\nMejor modelo de deteccion (mAP50-95):")
        print(f"  {best['name']}")
        print(f"  mAP50-95: {best['mAP50_95']:.4f}")
        print(f"  Config: IoU={best['iou']}, Conf={best['conf']}")
    
    if 'detection_precision' in summary['best_models']:
        best = summary['best_models']['detection_precision']
        print(f"\nMejor modelo de deteccion (Precision):")
        print(f"  {best['name']}")
        print(f"  Precision: {best['precision']:.4f}")
        print(f"  Config: IoU={best['iou']}, Conf={best['conf']}")
    
    if 'detection_recall' in summary['best_models']:
        best = summary['best_models']['detection_recall']
        print(f"\nMejor modelo de deteccion (Recall):")
        print(f"  {best['name']}")
        print(f"  Recall: {best['recall']:.4f}")
        print(f"  Config: IoU={best['iou']}, Conf={best['conf']}")
    
    if 'segmentation' in summary['best_models']:
        best = summary['best_models']['segmentation']
        print(f"\nMejor modelo de segmentacion (mAP50-95):")
        print(f"  {best['name']}")
        print(f"  mAP50-95: {best['mAP50_95']:.4f}")
        print(f"  Config: IoU={best['iou']}, Conf={best['conf']}")
    
    if 'segmentation_precision' in summary['best_models']:
        best = summary['best_models']['segmentation_precision']
        print(f"\nMejor modelo de segmentacion (Precision):")
        print(f"  {best['name']}")
        print(f"  Precision: {best['precision']:.4f}")
        print(f"  Config: IoU={best['iou']}, Conf={best['conf']}")
    
    if 'segmentation_recall' in summary['best_models']:
        best = summary['best_models']['segmentation_recall']
        print(f"\nMejor modelo de segmentacion (Recall):")
        print(f"  {best['name']}")
        print(f"  Recall: {best['recall']:.4f}")
        print(f"  Config: IoU={best['iou']}, Conf={best['conf']}")
    
    if 'background_analysis' in summary:
        bg = summary['background_analysis']
        print(f"\nAnalisis de confusion con background:")
        print(f"  Mas confusion: {bg['most_confused']['name']} ({bg['most_confused']['rate']:.4f})")
        print(f"  Menos confusion: {bg['least_confused']['name']} ({bg['least_confused']['rate']:.4f})")
    
    print("\n" + "="*80)


def main():
    print("="*80)
    print("EVALUACION DE MODELOS")
    print("="*80)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_dir = OUTPUT_DIR / f'eval_{timestamp}'
    eval_dir.mkdir(exist_ok=True)
    
    print(f"Output: {eval_dir}")
    
    print("\n" + "="*80)
    print("BUSQUEDA DE MODELOS")
    print("="*80)
    
    models = find_all_models()
    
    print(f"Deteccion: {len(models['detection'])}")
    for m in models['detection']:
        print(f"  - {m['name']}")
    
    print(f"\nSegmentacion: {len(models['segmentation'])}")
    for m in models['segmentation']:
        print(f"  - {m['name']}")
    
    print("\n" + "="*80)
    print("EVALUACION DE MODELOS")
    print("="*80)
    
    all_results = {'detection': [], 'segmentation': []}
    
    data_yaml_detect = PROJECT_ROOT / 'data' / 'yolo' / 'data.yaml'
    for model_info in models['detection']:
        results = evaluate_model(
            model_path=model_info['path'],
            data_yaml=data_yaml_detect,
            model_name=model_info['name'],
            task_type='detection'
        )
        all_results['detection'].append(results)
    
    data_yaml_seg = PROJECT_ROOT / 'data' / 'yolo_seg' / 'data.yaml'
    for model_info in models['segmentation']:
        results = evaluate_model(
            model_path=model_info['path'],
            data_yaml=data_yaml_seg,
            model_name=model_info['name'],
            task_type='segmentation'
        )
        all_results['segmentation'].append(results)
    
    print("\n" + "="*80)
    print("MATRICES DE CONFUSION")
    print("="*80)
    
    with open(data_yaml_detect, 'r') as f:
        class_names = yaml.safe_load(f)['names']
    
    cm_dir = eval_dir / 'confusion_matrices'
    cm_dir.mkdir(exist_ok=True)
    
    for result in all_results['detection']:
        if 'best' in result:
            best = result['best']
            model_path = [m['path'] for m in models['detection'] if m['name'] == result['model_name']][0]
            
            save_path = cm_dir / f"{result['model_name']}_cm.png"
            generate_confusion_matrix(
                model_path=model_path,
                data_yaml=data_yaml_detect,
                iou=best['iou'],
                conf=best['conf'],
                class_names=class_names,
                save_path=save_path
            )
    
    for result in all_results['segmentation']:
        if 'best' in result:
            best = result['best']
            model_path = [m['path'] for m in models['segmentation'] if m['name'] == result['model_name']][0]
            
            save_path = cm_dir / f"{result['model_name']}_cm.png"
            generate_confusion_matrix(
                model_path=model_path,
                data_yaml=data_yaml_seg,
                iou=best['iou'],
                conf=best['conf'],
                class_names=class_names,
                save_path=save_path
            )
    
    print("\n" + "="*80)
    print("INFERENCIA EN BATCH")
    print("="*80)
    
    common_images = find_common_images()
    print(f"Imagenes comunes: {len(common_images)}")
    
    if common_images:
        run_batch_inference(models, common_images, eval_dir)
    
    print("\n" + "="*80)
    print("GUARDANDO RESULTADOS")
    print("="*80)
    
    results_json = eval_dir / 'detailed_results.json'
    with open(results_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Resultados detallados: {results_json}")
    
    summary_yaml = eval_dir / 'summary.yaml'
    summary = create_summary(all_results, summary_yaml)
    
    print_summary(summary)
    
    print("\n" + "="*80)
    print("EVALUACION COMPLETADA")
    print("="*80)
    print(f"Resultados en: {eval_dir}")
    print("  - summary.yaml")
    print("  - detailed_results.json")
    print("  - confusion_matrices/")
    print("  - batch_inference/")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
