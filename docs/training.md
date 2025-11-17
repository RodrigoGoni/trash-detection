# Guía de Entrenamiento de Modelos

Esta guía documenta la evolución del entrenamiento, justificando la elección de los modelos y la progresión de los experimentos, con un seguimiento centralizado mediante **MLflow**.

---

### 1. Definición del baseline: Faster R-CNN (ResNet-50)

Todo proceso de machine learning necesita un punto de referencia robusto para medir el progreso.

* **Elección del baseline:** se seleccionó un modelo **Faster R-CNN con backbone ResNet-50**, pre-entrenado en el dataset COCO.
* **Justificación:**
    1.  **Robustez:** Es una arquitectura de dos etapas y madura, que tiende a dar buenos resultados en mAP (Mean Average Precision).
    2.  **Validación del pipeline:** sirvió para validar que nuestro pipeline de datos (corrección EXIF, split estratificado) y las métricas de evaluación (`evaluate_model.py`) funcionaban correctamente.
    3.  **Manejo de clases:** al estar pre-entrenado en COCO (80 clases), el modelo ya posee una buena capacidad de extracción de características generales, que podemos "afinar" (fine-tuning) a nuestras 60 clases de basura.
* **Seguimiento:** el script `scripts/train_faster_rcnn.py` integra MLflow para registrar hiperparámetros, curvas de pérdida (train/val) y guardar el *checkpoint* del modelo (`best_model.pth`) basado en la mejor métrica de validación.

---

### 2. Evolución

El entrenamiento no es un solo paso, sino una serie de experimentos iterativos. Cada modificación se justifica en base a los resultados del experimento anterior.

*Nota: Esta sección debe incluir gráficas clave (ej. curvas de pérdida, ejemplos de predicciones) de MLflow para ilustrar la progresión.*

#### Experimento 1: Baseline Puro
* **Acción:** Entrenar el Faster R-CNN solo con re-escalado y normalización.
* **Observación (Hipótesis):** Fuerte sobreajuste (overfitting) y bajo rendimiento en clases minoritarias.

#### Experimento 2: Introducción de Augmentations
* **Justificación:** Para combatir el overfitting y la variabilidad de las imágenes (observada en el EDA).
* **Acción:** Añadir el pipeline de Albumentations (flips, rotaciones, cambios de brillo/contraste).
* **Observación:** Mejora de la pérdida de validación y mayor generalización.

#### Experimento 3: Manejo del Desbalance de Clases
* **Justificación:** El mAP general es aceptable, pero el *recall* en clases minoritarias es muy bajo (detectado en el EDA).
* **Acción:** Activar el uso de **Focal Loss** (o Class Balanced Loss) para forzar al modelo a enfocarse en los ejemplos difíciles y las clases raras.
* **Observación:** Mejora significativa en el mAP de clases minoritarias, aunque puede reducir ligeramente la precisión en las clases mayoritarias.

#### Experimento 4: Eliminación de Clases Raras
* **Justificación:** Algunas clases tienen tan pocas muestras (ej. < 10) que el modelo no puede aprenderlas y solo añaden ruido.
* **Acción:** Re-ejecutar `prepare_data.py` con `min-annotations` más alto o excluir manualmente clases.
* **Impacto:** Se debe mostrar el (ligero) impacto en el balance general del dataset y justificarlo en pos de un modelo más estable y con un objetivo más enfocado.

---

### 3. Modelo Avanzado: YOLOv11

Tras analizar los errores del baseline, se exploraron arquitecturas más modernas.

* **Elección del Modelo:** **YOLOv11** (evaluado en `yolo_test_evaluation.ipynb`).
* **Justificación (Por qué YOLOv11):**
    1.  **Errores del Baseline:** El Faster R-CNN (basado en *anchors*) puede tener dificultades con objetos de relaciones de aspecto extremas (ej. "pajitas") o con la velocidad de inferencia.
    2.  **Enfoque sin Anchors (Anchor-Free):** Las arquitecturas modernas como YOLOv11 (dependiendo de la versión) utilizan enfoques *anchor-free* que aprenden a detectar el centro del objeto directamente, adaptándose mejor a las formas variadas de la basura.
    3.  **Velocidad y Eficiencia:** Los detectores *single-stage* (YOLO) ofrecen un mejor compromiso velocidad/precisión, crucial para un despliegue en tiempo real.
* **Análisis:** El notebook `yolo_test_evaluation.ipynb` permite un análisis de errores más profundo, visualizando los peores casos y ajustando dinámicamente los umbrales de confianza e IoU para entender dónde falla el modelo.

---

### 4. Próximos Pasos: Instance Segmentation

* **Observación (Feedback):** Un problema persistente en la detección de basura es que los *bounding boxes* rectangulares a menudo incluyen una gran cantidad de "fondo" (ej. pasto, asfalto), confundiendo al modelo.
* **Solución Coherente:** Avanzar hacia la **segmentación de instancias** (ej. Mask R-CNN o YOLOv11-Seg).
* **Justificación:** La segmentación fuerza al modelo a aprender la forma *exacta* del objeto (generando una máscara, no solo una caja). Esto reduce la dependencia del fondo, mejora la separación de objetos superpuestos y proporciona una métrica de IoU mucho más precisa, lo cual es coherente con los problemas observados en el análisis de errores.

---

## 5. Configuración detallada del entrenamiento (YAML)

El *pipeline* de entrenamiento se controla mediante archivos de configuración `.yaml` para garantizar la reproducibilidad y facilitar el seguimiento de experimentos con MLflow. A continuación, se documentan los dos archivos de configuración principales.

### 5.1. Baseline: Faster R-CNN (`train_config.yaml`)

Este archivo configura el *pipeline* de entrenamiento para el modelo *baseline* (Faster R-CNN con un *backbone*). Está diseñado para usar un *pipeline* de aumentos de datos externo (vía Albumentations) y una estrategia de pérdida avanzada para combatir el desbalanceo extremo de clases del dataset TACO.

**Parámetros destacados:**

* **Gestión del desbalanceo (Sección `loss`):** esta es la sección más crítica. Implementa una estrategia de **Class-Balanced Focal Loss**.
    * `type: "cb_focal"`: Combina Focal Loss (para enfocarse en ejemplos difíciles) con ponderación de clases (para enfocarse en clases raras).
    * `use_class_weights: true` y `class_weight_method: "effective"`: habilita la ponderación basada en el "Número Efectivo de Muestras", una técnica avanzada que da un peso significativamente mayor a las clases con pocas muestras.
    * `beta: 0.9999`: parámetro de suavizado para la ponderación. Un valor tan alto es específico para datasets con desbalanceo extremo como TACO.
    * `gamma: 2.0`: parámetro de enfoque de Focal Loss.
    * `use_weighted_bbox: true`: aplica esta misma ponderación de clase a la pérdida de regresión de la *bounding box*, asumiendo que las clases raras también son más difíciles de localizar.

* **Augmentation (Sección `augmentation`):** define un *pipeline* de aumento de datos robusto y probabilístico, justificado por el análisis EDA:
    * Transformaciones geométricas (`horizontal_flip_p: 0.5`, `rotate_limit: 90`) para manejar objetos en cualquier orientación.
    * Transformaciones fotométricas (`brightness_contrast_p: 0.6`, `hue_saturation_p: 0.4`) para simular las diversas condiciones de iluminación observadas en el dataset.
    * Aumentos de ruido y oclusión (`blur_p: 0.3`, `noise_p: 0.25`, `cutout_p: 0.2`) para mejorar la robustez.

* **Optimización y Rendimiento:**
    * `optimizer: "adamw"` y `scheduler: "cosine_annealing_warmup"`: utiliza un optimizador moderno (AdamW) con un *scheduler* de *warmup* y decaimiento coseno para un entrenamiento estable.
    * `use_amp: true`: habilita **Automatic Mixed Precision (AMP)** para un entrenamiento más rápido y con menor uso de memoria VRAM.
    * `grad_clip_norm: 1.0`: previene la explosión de gradientes, mejorando la estabilidad del entrenamiento.

