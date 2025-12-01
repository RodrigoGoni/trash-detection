import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def load_results(results_file):
    with open(results_file, 'r') as f:
        return json.load(f)

def plot_map_comparison(results, output_dir):
    print("\nGrafica 1: Comparacion de mAP50-95")
    
    detection_models = []
    detection_map = []
    segmentation_models = []
    segmentation_map = []
    
    for model in results['detection']:
        if 'best' in model:
            detection_models.append(model['model_name'])
            detection_map.append(model['best']['mAP50_95'])
    
    for model in results['segmentation']:
        if 'best' in model:
            segmentation_models.append(model['model_name'])
            segmentation_map.append(model['best']['mAP50_95'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Detection
    if detection_models:
        y_pos = np.arange(len(detection_models))
        bars = ax1.barh(y_pos, detection_map, color='steelblue')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(detection_models, fontsize=8)
        ax1.set_xlabel('mAP50-95', fontsize=10)
        ax1.set_title('Modelos de Deteccion', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, detection_map)):
            ax1.text(val, i, f' {val:.3f}', va='center', fontsize=8)
    
    # Segmentation
    if segmentation_models:
        y_pos = np.arange(len(segmentation_models))
        bars = ax2.barh(y_pos, segmentation_map, color='coral')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(segmentation_models, fontsize=8)
        ax2.set_xlabel('mAP50-95', fontsize=10)
        ax2.set_title('Modelos de Segmentacion', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, segmentation_map)):
            ax2.text(val, i, f' {val:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_map.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: comparison_map.png")

def plot_precision_recall(results, output_dir):
    print("\nGrafica 2: Precision vs Recall")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Detection models
    for idx, model in enumerate(results['detection']):
        if 'best' in model:
            prec = model['best']['precision']
            rec = model['best']['recall']
            ax.scatter(rec, prec, s=150, alpha=0.7, marker='o', 
                      label=f"{idx+1}. {model['model_name']}", color=f'C{idx}')
            ax.annotate(str(idx+1), (rec, prec), 
                       fontsize=10, ha='center', va='center', 
                       fontweight='bold', color='white')
    
    # Segmentation models
    seg_offset = len(results['detection'])
    for idx, model in enumerate(results['segmentation']):
        if 'best' in model:
            prec = model['best']['precision']
            rec = model['best']['recall']
            label_num = seg_offset + idx + 1
            ax.scatter(rec, prec, s=150, alpha=0.7, marker='s', 
                      label=f"{label_num}. {model['model_name']}", color=f'C{idx+3}')
            ax.annotate(str(label_num), (rec, prec), 
                       fontsize=10, ha='center', va='center',
                       fontweight='bold', color='white')
    
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision vs Recall (Mejor Configuracion)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Linea diagonal
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_precision_recall.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: comparison_precision_recall.png")

def plot_background_confusion(results, output_dir):
    print("\nGrafica 3: Confusion con Background")
    
    all_models = []
    bg_confusion = []
    colors = []
    
    for model in results['detection']:
        all_models.append(model['model_name'])
        
        bg_rates = [e.get('background_confusion_rate', 0) for e in model['evaluations']
                   if 'background_confusion_rate' in e]
        avg_bg = np.mean(bg_rates) if bg_rates else 0
        bg_confusion.append(avg_bg)
        colors.append('steelblue')
    
    for model in results['segmentation']:
        all_models.append(model['model_name'])
        
        bg_rates = [e.get('background_confusion_rate', 0) for e in model['evaluations']
                   if 'background_confusion_rate' in e]
        avg_bg = np.mean(bg_rates) if bg_rates else 0
        bg_confusion.append(avg_bg)
        colors.append('coral')
    
    if not all_models:
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    y_pos = np.arange(len(all_models))
    bars = ax.barh(y_pos, bg_confusion, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_models, fontsize=8)
    ax.set_xlabel('Tasa de Confusion con Background (promedio)', fontsize=10)
    ax.set_title('Confusion con Background por Modelo (menor es mejor)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, bg_confusion)):
        ax.text(val, i, f' {val:.4f}', va='center', fontsize=8)
    
    # Linea de referencia
    median_val = np.median(bg_confusion)
    ax.axvline(median_val, color='red', linestyle='--', alpha=0.5, linewidth=2, label=f'Mediana: {median_val:.4f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_background_confusion.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: comparison_background_confusion.png")

def plot_metrics_heatmap(results, output_dir):
    print("\nGrafica 4: Heatmap de Metricas")
    
    all_models = []
    metrics_data = []
    
    for model in results['detection']:
        if 'best' in model:
            all_models.append(model['model_name'])
            metrics_data.append([
                model['best']['mAP50'],
                model['best']['mAP50_95'],
                model['best']['precision'],
                model['best']['recall']
            ])
    
    for model in results['segmentation']:
        if 'best' in model:
            all_models.append(model['model_name'])
            metrics_data.append([
                model['best']['mAP50'],
                model['best']['mAP50_95'],
                model['best']['precision'],
                model['best']['recall']
            ])
    
    if not metrics_data:
        return
    
    metrics_array = np.array(metrics_data)
    metric_names = ['mAP50', 'mAP50-95', 'Precision', 'Recall']
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(all_models) * 0.4)))
    
    im = ax.imshow(metrics_array, cmap='YlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_yticks(np.arange(len(all_models)))
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_yticklabels(all_models, fontsize=8)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Valores en las celdas
    for i in range(len(all_models)):
        for j in range(len(metric_names)):
            text = ax.text(j, i, f'{metrics_array[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title('Heatmap de Metricas (Mejor Configuracion)', fontsize=13, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Valor de Metrica', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_metrics_heatmap.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: comparison_metrics_heatmap.png")

def plot_best_config_distribution(results, output_dir):
    print("\nGrafica 5: Distribucion de Mejores Configuraciones")
    
    iou_values = []
    conf_values = []
    model_names = []
    model_types = []
    
    for model in results['detection']:
        if 'best' in model:
            iou_values.append(model['best']['iou'])
            conf_values.append(model['best']['conf'])
            model_names.append(model['model_name'])
            model_types.append('detection')
    
    for model in results['segmentation']:
        if 'best' in model:
            iou_values.append(model['best']['iou'])
            conf_values.append(model['best']['conf'])
            model_names.append(model['model_name'])
            model_types.append('segmentation')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot IoU vs Conf
    colors_map = {'detection': 'steelblue', 'segmentation': 'coral'}
    markers_map = {'detection': 'o', 'segmentation': 's'}
    
    for idx, (iou, conf, name, mtype) in enumerate(zip(iou_values, conf_values, model_names, model_types)):
        ax1.scatter(iou, conf, s=150, c=colors_map[mtype], 
                   marker=markers_map[mtype], alpha=0.7,
                   label=f"{idx+1}. {name}")
        ax1.annotate(str(idx+1), (iou, conf),
                    fontsize=10, ha='center', va='center',
                    fontweight='bold', color='white')
    
    ax1.set_xlabel('IoU Threshold', fontsize=11)
    ax1.set_ylabel('Confidence Threshold', fontsize=11)
    ax1.set_title('Mejores Configuraciones IoU vs Conf', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, framealpha=0.9)
    
    # Histograms
    ax2_twin = ax2.twinx()
    
    width = 0.04
    iou_unique = sorted(set(iou_values))
    conf_unique = sorted(set(conf_values))
    
    iou_counts = [iou_values.count(v) for v in iou_unique]
    conf_counts = [conf_values.count(v) for v in conf_unique]
    
    bars1 = ax2.bar([x - width/2 for x in iou_unique], iou_counts, width, 
                    label='IoU', color='steelblue', alpha=0.7)
    bars2 = ax2_twin.bar([x + width/2 for x in conf_unique], conf_counts, width,
                         label='Conf', color='coral', alpha=0.7)
    
    ax2.set_xlabel('Threshold Value', fontsize=11)
    ax2.set_ylabel('Frecuencia (IoU)', fontsize=11, color='steelblue')
    ax2_twin.set_ylabel('Frecuencia (Conf)', fontsize=11, color='coral')
    ax2.set_title('Distribucion de Thresholds Optimos', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax2_twin.tick_params(axis='y', labelcolor='coral')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_best_configs.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: comparison_best_configs.png")

def plot_performance_radar(results, output_dir):
    print("\nGrafica 6: Radar Chart - Top 5 Modelos")
    
    all_models = []
    
    for model in results['detection']:
        if 'best' in model:
            bg_rates = [e.get('background_confusion_rate', 0) for e in model['evaluations']
                       if 'background_confusion_rate' in e]
            avg_bg = np.mean(bg_rates) if bg_rates else 0
            
            all_models.append({
                'name': model['model_name'],
                'map': model['best']['mAP50_95'],
                'precision': model['best']['precision'],
                'recall': model['best']['recall'],
                'bg_confusion': 1 - avg_bg,  # Invertir para que mayor sea mejor
                'type': 'detection'
            })
    
    for model in results['segmentation']:
        if 'best' in model:
            bg_rates = [e.get('background_confusion_rate', 0) for e in model['evaluations']
                       if 'background_confusion_rate' in e]
            avg_bg = np.mean(bg_rates) if bg_rates else 0
            
            all_models.append({
                'name': model['model_name'],
                'map': model['best']['mAP50_95'],
                'precision': model['best']['precision'],
                'recall': model['best']['recall'],
                'bg_confusion': 1 - avg_bg,
                'type': 'segmentation'
            })
    
    # Ordenar por mAP y tomar top 5
    all_models.sort(key=lambda x: x['map'], reverse=True)
    top_models = all_models[:5]
    
    categories = ['mAP50-95', 'Precision', 'Recall', 'No-BG\nConfusion']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, model in enumerate(top_models):
        values = [model['map'], model['precision'], model['recall'], model['bg_confusion']]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model['name'], 
                color=colors_list[idx])
        ax.fill(angles, values, alpha=0.15, color=colors_list[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True)
    
    ax.set_title('Comparacion Multimetrica - Top 5 Modelos', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_radar_top5.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: comparison_radar_top5.png")

def plot_map_vs_background(results, output_dir):
    print("\nGrafica 7: mAP vs Background Confusion")
    
    map_values = []
    bg_values = []
    model_names = []
    model_types = []
    
    for model in results['detection']:
        if 'best' in model:
            bg_rates = [e.get('background_confusion_rate', 0) for e in model['evaluations']
                       if 'background_confusion_rate' in e]
            if bg_rates:
                map_values.append(model['best']['mAP50_95'])
                bg_values.append(np.mean(bg_rates))
                model_names.append(model['model_name'])
                model_types.append('detection')
    
    for model in results['segmentation']:
        if 'best' in model:
            bg_rates = [e.get('background_confusion_rate', 0) for e in model['evaluations']
                       if 'background_confusion_rate' in e]
            if bg_rates:
                map_values.append(model['best']['mAP50_95'])
                bg_values.append(np.mean(bg_rates))
                model_names.append(model['model_name'])
                model_types.append('segmentation')
    
    if not map_values:
        print("  ADVERTENCIA: No hay datos de background confusion disponibles")
        print("  Ejecuta evaluate_all_models.py nuevamente para generar estos datos")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors_map = {'detection': 'steelblue', 'segmentation': 'coral'}
    markers_map = {'detection': 'o', 'segmentation': 's'}
    
    for idx, (map_val, bg_val, name, mtype) in enumerate(zip(map_values, bg_values, model_names, model_types)):
        ax.scatter(bg_val, map_val, s=150, c=colors_map[mtype], 
                  marker=markers_map[mtype], alpha=0.7,
                  label=f"{idx+1}. {name}")
        ax.annotate(str(idx+1), (bg_val, map_val),
                   fontsize=10, ha='center', va='center',
                   fontweight='bold', color='white')
    
    ax.set_xlabel('Background Confusion Rate (menor es mejor)', fontsize=11)
    ax.set_ylabel('mAP50-95 (mayor es mejor)', fontsize=11)
    ax.set_title('Trade-off: Performance vs Background Confusion', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Cuadrantes ideales
    if map_values and bg_values:
        median_map = np.median(map_values)
        median_bg = np.median(bg_values)
        
        ax.axhline(median_map, color='green', linestyle='--', alpha=0.3, linewidth=1)
        ax.axvline(median_bg, color='red', linestyle='--', alpha=0.3, linewidth=1)
        
        # Marcar zona ideal (alto mAP, bajo BG confusion)
        ax.fill_between([0, median_bg], median_map, max(map_values)*1.1, 
                        color='green', alpha=0.1, label='Zona Ideal')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_map_vs_background.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: comparison_map_vs_background.png")

def main():
    if len(sys.argv) > 1:
        results_file = Path(sys.argv[1])
    else:
        # Buscar el resultado mas reciente
        eval_dir = Path(__file__).parent.parent / 'evaluation_results'
        if not eval_dir.exists():
            print("Error: No se encontro carpeta evaluation_results")
            return
        
        eval_folders = sorted([d for d in eval_dir.iterdir() if d.is_dir()], 
                            key=lambda x: x.name, reverse=True)
        
        if not eval_folders:
            print("Error: No hay evaluaciones previas")
            return
        
        results_file = eval_folders[0] / 'detailed_results.json'
    
    if not results_file.exists():
        print(f"Error: No se encuentra {results_file}")
        return
    
    print("="*80)
    print("GRAFICAS DE COMPARACION DE MODELOS")
    print("="*80)
    print(f"Archivo: {results_file}")
    
    results = load_results(results_file)
    output_dir = results_file.parent / 'comparison_plots'
    output_dir.mkdir(exist_ok=True)
    
    print(f"Output: {output_dir}")
    
    # Generar todas las graficas
    plot_map_comparison(results, output_dir)
    plot_precision_recall(results, output_dir)
    plot_background_confusion(results, output_dir)
    plot_metrics_heatmap(results, output_dir)
    plot_best_config_distribution(results, output_dir)
    plot_performance_radar(results, output_dir)
    plot_map_vs_background(results, output_dir)
    
    print("\n" + "="*80)
    print("GRAFICAS COMPLETADAS")
    print("="*80)
    print(f"Guardadas en: {output_dir}")
    print("  1. comparison_map.png")
    print("  2. comparison_precision_recall.png")
    print("  3. comparison_background_confusion.png")
    print("  4. comparison_metrics_heatmap.png")
    print("  5. comparison_best_configs.png")
    print("  6. comparison_radar_top5.png")
    print("  7. comparison_map_vs_background.png")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
