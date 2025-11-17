# Guía de procesamiento de datos

La preparación de los datos es un componente crítico de este proyecto, dado el estado inicial del dataset TACO. Este documento detalla el pipeline de preprocesamiento y, fundamentalmente, las **conclusiones del análisis exploratorio (EDA)** que justifican las decisiones de entrenamiento.

---

### Paso 1: Corrección de metadatos EXIF

Un hallazgo clave del análisis inicial fue la inconsistencia en los metadatos EXIF de orientación en las imágenes crudas.

* **Problema:** un porcentaje significativo de las imágenes (ej. 20.6% del set de entrenamiento) se cargaba con una rotación incorrecta en los frameworks de CV, invalidando las anotaciones de *bounding box*.
* **Solución:** se implementó un paso en `scripts/prepare_data.py` que utiliza `PIL.ImageOps.exif_transpose()` para aplicar la transformación de rotación a los píxeles y guardar la imagen limpia, sin los metadatos EXIF.
* **Justificación:** aunque algunos *data loaders* modernos de PyTorch pueden manejar esto en tiempo de ejecución, realizar la corrección "en disco" asegura la consistencia de los datos en cualquier framework y para inspección visual manual.

---

### Paso 2: Análisis Exploratorio de Datos (EDA) y conclusiones

El análisis (reflejado en `notebooks/01_exploration/` y `notebooks/02_preprocessing/`) no solo guio el preprocesamiento, sino que reveló problemas claves a ser resueltos.

#### 2.1. Desbalanceo de clases y estratificación

* **Observación:** el dataset sufre de un **fuerte desbalanceo de clases**. El análisis de distribución muestra que unas pocas clases dominan el dataset, mientras que muchas otras tienen una representación mínima.
* **Decisión (Split):** para asegurar que las clases minoritarias estuvieran presentes en validación y prueba, se implementó una **división estratificada** (ej. 70/15/15) en `scripts/prepare_data.py`. El notebook `class_balance_analysis.ipynb` valida que esta estratificación mantiene la distribución de clases en todos los conjuntos.
* **Decisión (entrenamiento):** dado el desbalanceo, se preparó el pipeline de entrenamiento para utilizar técnicas de ponderación de clases, como **Focal Loss** o **Class Balanced Loss**, que son cruciales para que el modelo preste atención a las clases menos frecuentes y pueda tener una mejor generalización en todas las clases.

#### 2.2. Co-ocurrencia de Clases

* **Observación:** el heatmap de co-ocurrencias (analizado en `class_balance_analysis.ipynb`) muestra patrones de qué objetos tienden a aparecer juntos (ej. "botellas" y "tapas").
* **Implicación:** esto sugiere que el contexto es importante. El modelo no solo debe identificar un objeto aislado, sino ser capaz de diferenciar objetos distintos que frecuentemente aparecen agrupados en la misma escena.

#### 2.3. Análisis de casos extremos

* **Observación:** el notebook `preprocessing_visualization.ipynb` incluye un análisis de casos extremos de brillo, contraste y luminosidad. Se observó que muchas imágenes están sobreexpuestas (cielos brillantes) o subexpuestas (sombras).
* **Decisión:** el pipeline de *augmentation* debe ser robusto. Se incluyeron transformaciones (`RandomBrightnessContrast`, `HueSaturationValue`) para simular estas condiciones y forzar al modelo a ser invariante a la iluminación.

#### 2.4. Análisis Adicionales 

Para enriquecer el EDA, los siguientes pasos serían:
* **Análisis de Relación de Aspecto (Aspect Ratio):** analizar la distribución de las relaciones de aspecto de las *bounding boxes* por clase. Esto ayudaría a optimizar los *anchor boxes* para los modelos que los utilizan (como Faster R-CNN) o a entender si ciertas clases (ej. "pajitas" vs. "bolsas") son dimensionalmente distintas.
* **Análisis de Color:** extraer los perfiles de color dominantes (en HSV o LAB) por categoría. Esto podría revelar si ciertos canales de color son predictores fuertes para clases específicas, justificando potencialmente arquitecturas de modelo que presten más atención al color.

### Paso 3: Pipeline de aumento (Augmentation) - A mejorar

Basado en el EDA, se implementó un pipeline de aumento de datos usando **Albumentations**.

* **Geométricas:** `HorizontalFlip`, `ShiftScaleRotates`.
* **Fotométricas:** `RandomBrightnessContrast`, `HueSaturationValue` (para manejar los casos extremos observados).
* **Oclusión:** `CoarseDropout` (Cutout) para simular oclusión parcial.


### Paso 4: Conversión a Formato YOLO

El script `scripts/convert_to_yolo.py` transforma las anotaciones COCO (generadas por `prepare_data.py`) al formato de texto requerido por YOLO, normalizando las coordenadas y creando el archivo `data.yaml`.

--- 

# Seccion avanzada de problemas y soluciones

## Problema Crítico: Orientación EXIF en Dataset TACO

### Problema Detectado

**Ultralytics NO gestiona automáticamente la orientación EXIF**, lo que causa un desajuste entre las anotaciones y las imágenes durante el entrenamiento.

### Análisis del Problema

**¿Qué es la orientación EXIF?**
- Los metadatos EXIF en imágenes JPEG incluyen un flag de orientación (1-8)
- Este flag indica cómo debe rotarse la imagen para visualizarla correctamente
- Muchas cámaras guardan fotos rotadas físicamente pero con flag EXIF para "corregir" la visualización

**El Desajuste:**

1. **Visualización (navegadores, PIL, visores de fotos):**
   - Lee el flag EXIF y aplica rotación automáticamente
   - Muestra la imagen correctamente orientada (e.g., 2988x5312 vertical)

2. **Ultralytics/OpenCV durante entrenamiento:**
   - Usa `cv2.imread()` que IGNORA completamente los metadatos EXIF
   - Lee los píxeles RAW sin aplicar rotación
   - Ve dimensiones físicas del archivo (e.g., 5312x2988 horizontal)

3. **Las Anotaciones (bounding boxes):**
   - Fueron creadas sobre la imagen visualizada correctamente (con EXIF aplicado)
   - Coordenadas corresponden a la versión rotada (2988x5312)

4. **Resultado:**
   - El modelo ve: 5312x2988 (píxeles RAW)
   - Las anotaciones esperan: 2988x5312 (versión rotada)
   - PROBLEMA: bounding boxes en posiciones incorrectas

### Ejemplo Concreto del Dataset TACO

```
Archivo: batch_1/000042.jpg
├─ Dimensiones físicas RAW: 5312x2988 (horizontal)
├─ Flag EXIF Orientation: 6 (rotar 90° horario)
├─ Visualización correcta: 2988x5312 (vertical)
├─ Lo que ve Ultralytics: 5312x2988 (horizontal, sin rotación)
└─ Anotaciones: Para imagen 2988x5312 → INCORRECTAS
```

### Solución Implementada

**1. Integración en `prepare_data.py`**

El script de preprocesamiento ahora corrige automáticamente la orientación EXIF:

```bash
# Con corrección EXIF (por defecto)
python scripts/prepare_data.py --overwrite

# Sin corrección EXIF (no recomendado)
python scripts/prepare_data.py --no-fix-exif --overwrite
```

**Proceso de corrección:**
1. Lee la imagen con PIL
2. Aplica `ImageOps.exif_transpose()` para rotar físicamente los píxeles
3. Guarda la imagen rotada SIN metadatos EXIF (quality=95)
4. Mantiene la estructura de directorios `batch_X/`

**2. Estadísticas del Dataset TACO**

Después de ejecutar `prepare_data.py`:

```
================================================================================
EXIF CORRECTIONS
================================================================================
Train: 144/698 imágenes corregidas (20.6%)
================================================================================
```

### Beneficios de la Corrección

- **Consistencia**: Modelo ve exactamente lo que las anotaciones describen  
- **Reproducibilidad**: Mismo comportamiento en entrenamiento y producción  
- **Automatización**: Se integra en el pipeline de preprocesamiento  
- **Preservación**: Mantiene calidad de imagen (JPEG quality=95)  
- **Trazabilidad**: Estadísticas guardadas en `dataset_stats.json`


### Referencias

- **EXIF Orientation**: [EXIF Specification](https://www.exif.org/Exif2-2.PDF) Section 4.6.4A
- **PIL exif_transpose**: [Pillow Documentation](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.exif_transpose)
- **OpenCV EXIF Issue**: [GitHub Issue #16352](https://github.com/opencv/opencv/issues/16352)
