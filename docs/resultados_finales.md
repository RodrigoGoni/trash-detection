# Análisis Comparativo de Arquitecturas

**Fecha:** 01/12/2025
**Dataset:** TACO (Split Estratificado v1.0 - 23 Clases Prioritarias)
**Estado:** Evaluación final y comparativa de Arquitecturas.

## 1. Resumen Ejecutivo
Este documento consolida la comparación de rendimiento final. Se evalúa la evolución desde el baseline hasta las arquitecturas modernas de YOLOv11, filtrando únicamente el **mejor exponente de cada tamaño** para determinar la arquitectura óptima de despliegue.

---

## 2. Tabla comparativa

Se seleccionó el modelo con mayor **mAP@50** dentro de cada tamaño disponible, descartando iteraciones con configuraciones subóptimas.

| Tarea | Tamaño | Modelo | mAP@50 | Precision | Recall | Config (IoU/Conf) | Observación |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Segmentation** | **Medium** | `yolo11m-seg` | **0.447** | **0.549** | 0.303 | 0.3 / 0.5 | **Ganador** |
| **Segmentation** | **X-Large** | `yolo11x-seg` | 0.445 | **0.549** | **0.322** | 0.7 / 0.4 | Rendimiento idéntico al Medium, pero mucho más costoso. |
| **Detection** | **Large** | `yolo11l` | 0.296 | 0.344 | 0.251 | 0.7 / 0.2 | Mejor opción si solo se requiere detección, pero lejos de la segmentación. |
| **Detection** | **Medium** | `yolo11m` | 0.117 | 0.104 | 0.115 | 0.7 / 0.2 | Rendimiento insuficiente para producción. |
| **Detection** | *Baseline* | Faster R-CNN (ResNet50) | 0.148 | *N/A* | *N/A* | *N/A* | Referencia histórica inicial. |

Comparativa directa entre el mejor detector y el mejor segmentador:

| Métrica | YOLOv11-L (Detect) | YOLOv11-M (Seg) | Delta | Significado |
| :--- | :--- | :--- | :--- | :--- |
| **mAP@50** | 0.296 | **0.447** | **+51%** | La segmentación entiende mejor la escena global. |
| **Precision** | 0.344 | **0.549** | **+60%** | El segmentador comete **muchos menos errores** de "falsos positivos" (no confunde piedras con basura). |
| **Recall** | 0.251 | **0.303** | **+21%** | El segmentador encuentra más basura oculta/difícil. |

---

## 3. Análisis de resultados por arquitectura

### 3.1. Detección pura
Se evaluaron múltiples variantes de detección (`m`, `l`) con diferentes estrategias de augmentation.

* **Mejor modelo (large):** logró un mAP@50 de **0.296**.
* **Modelos Menores (medium):** sin augmentation agresiva, el rendimiento cae drásticamente (mAP ~0.11).
* **Diagnóstico:** incluso con el modelo Large y augmentation pesada, la detección pura tiene dificultades para superar la barrera del 0.30 mAP. La **Precisión (0.344)** indica que de cada 3 objetos que detecta, 2 son falsos positivos.

### 3.2. Segmentación
La segmentación demuestra ser la solución correcta para el problema.

#### El Fenómeno Medium vs. X-Large
Un hallazgo crítico es que el modelo **medium de segmentación** (0.447 mAP) rinde técnicamente igual que el gigante **x-large** (0.445 mAP).

**¿Por qué el modelo Medium le gana al X-Large?**
1.  **Estabilidad de Entrenamiento:** el modelo *medium* permite un *batch size* mayor (16 vs 4), lo que estabiliza las capas de *Batch Normalization*. El modelo X, restringido por memoria, sufre para generalizar.
2.  **Sobreajuste (Overfitting):** el modelo X tiene millones de parámetros más. En un dataset pequeño como TACO, tiende a memorizar en lugar de aprender características robustas.

---

## 5. Justificación de metrica mAP@50

Para este estudio comparativo, se seleccionó **mAP@50** (Mean Average Precision con umbral IoU de 0.50) como la métrica principal de decisión. Esta elección no es arbitraria, sino que se fundamenta en la naturaleza visual del dataset TACO observada durante el entrenamiento.

### 1. Naturaleza amorfa y deformable
A diferencia de objetos rígidos (coches, señales de tráfico), los residuos son inherentemente **deformables**. Una "botella de plástico" puede estar aplastada, retorcida o fragmentada.
* **Implicación:** El *ground truth* (la caja dibujada por el humano) es subjetivo. Exigir un IoU > 0.75 penaliza al modelo por discrepancias subjetivas de etiquetado, no por fallos de detección real.

### 2. Prioridad de Detección
En el contexto de gestión de residuos, el objetivo crítico es **confirmar la existencia** del residuo y su ubicación general. Un IoU de 0.50 garantiza que el modelo ha encontrado el objeto y lo ha centrado correctamente.

