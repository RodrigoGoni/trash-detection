# Detección de basura: visión por computadora con MLOps

Este repositorio documenta un proyecto de visión por computadora para la detección de objetos de basura utilizando el conjunto de datos TACO. El proyecto se enfoca en una metodología iterativa, comenzando con un análisis profundo y un modelo *baseline*, para luego evolucionar hacia arquitecturas más avanzadas (como YOLOv11) con el fin de resolver los desafíos específicos del dataset.

Todo el ciclo de vida del modelo, desde la experimentación hasta la evaluación, está integrado con **MLflow** para un seguimiento riguroso de métricas, parámetros y artefactos.

## Descripción General del Problema

El objetivo es detectar 60 categorías de basura en imágenes del dataset TACO. Este dataset presenta desafíos significativos que guían la metodología del proyecto:

1.  **Mala calidad de datos:** un porcentaje de imágenes (aprox. 20%) presenta metadatos de orientación EXIF incorrectos, invalidando las anotaciones.
2.  **Fuerte desbalanceo de clases:** unas pocas clases dominan el dataset, mientras que la mayoría tienen muy pocas muestras.
3.  **Ambigüedad objeto-fondo:** los objetos de basura a menudo están ocluidos o sus *bounding boxes* contienen una gran cantidad de fondo irrelevante (pasto, asfalto).

## Documentación detallada

Para detalles de la justificación de las decisiones y la evolución de los experimentos (incluyendo gráficas de MLflow, análisis de errores y conclusiones del EDA), consultar la documentación en la carpeta `/docs`:

-   **[Guía de procesamiento de datos (docs/data_processing.md)](docs/data_processing.md):** detalla el Análisis Exploratorio (EDA), la corrección EXIF, el split estratificado y las conclusiones sobre el desbalanceo de clases.
-   **[Guía de entrenamiento de modelos (docs/training.md)](docs/training.md):** explica la justificación del *baseline* (Faster R-CNN), la experimentación y la evolución hacia YOLOv11 y la segmentación de instancias.
-   **[Evaluación de modelos (docs/analisis_comparativo_experimentos.md)](docs/analisis_comparativo_experimentos.md):** muestra la evolucion de los entrenamientos, augmentacion y seleccion de parametros.

## Estructura del Proyecto
```
├── config
│   ├── train_config.yaml
│   └── train_config_yolo11.yaml
├── data
│   ├── processed
│   └── raw
│       └── datasets
├── docs
│   ├── data_processing.md
│   └── training.md
├── environment.yml
├── LICENSE
├── models
│   └── checkpoints
├── notebooks
│   ├── 01_exploration
│   │   └── data_exploration_taco.ipynb
│   ├── 02_preprocessing
│   │   ├── class_balance_analysis.ipynb
│   │   └── preprocessing_visualization.ipynb
│   ├── 03_evaluation
│   │   ├── test_results_visualization.ipynb
│   │   └── yolo_test_evaluation.ipynb
│   └── test_yolo_annotations.ipynb
├── README.md
├── requirements.txt
├── scripts
│   ├── convert_to_yolo.py
│   ├── download.py
│   ├── evaluate_model.py
│   ├── prepare_data.py
│   ├── train_faster_rcnn.py
│   └── train_yolo.py
├── setup.py
├── src
│   ├── data
│   │   ├── augmentation.py
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   └── taco_dataloader.py
│   ├── __init__.py
│   ├── models
│   │   ├── backbone.py
│   │   ├── detector.py
│   │   ├── evaluate.py
│   │   ├── __init__.py
│   │   └── train.py
│   ├── training
│   │   ├── class_weights.py
│   │   ├── __init__.py
│   │   ├── losses.py
│   │   ├── optimizers.py
│   │   └── yolo_trainer.py
│   └── utils
│       ├── dvc_utils.py
│       ├── mlflow_utils.py
│       ├── system_info.py
│       └── training_logger.py
```
## Instalación y flujo de trabajo

### Requisitos Previos

- Python 3.10+
- Git

### 1. Instalación

1.  Clonar el repositorio:
    ```bash
    git clone [https://github.com/tu-usuario/tu-repositorio.git](https://github.com/tu-usuario/tu-repositorio.git)
    cd tu-repositorio
    ```

2.  Crear y activar el entorno virtual:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/Mac
    # venv\Scripts\activate    # Windows
    pip install -r requirements.txt
    ```

3.  Instalar el paquete `src` en modo editable:
    ```bash
    pip install -e .
    ```

### 2. Descarga de datos

El proyecto incluye un script para descargar la versión `v3` del dataset TACO desde Kaggle Hub. Se debe copiar el dataset descargado en el directorio cache de kaggle en el directorio local `data/raw`. 

```bash
python scripts/download.py
```

### 3. Preparación de datos (pipeline de preprocesamiento)

Este es un paso crucial. El script `scripts/prepare_data.py` limpia los datos crudos, corrige la orientación EXIF y crea los splits estratificados.

```bash
    # -----------------------------------------------------------------
    # Opción 1: Ejecución estándar (RECOMENDADA)
    # -----------------------------------------------------------------
    # - Corrige metadatos EXIF.
    # - Crea un split 70/15/15 ESTRATIFICADO (fundamental por el desbalanceo).
    # - Mantiene todas las imágenes (min-annotations 1).
    # - Sobrescribe el directorio 'data/processed'.
    python3 scripts/prepare_data.py --overwrite --stratify --min-annotations 1

    # -----------------------------------------------------------------
    # Opción 2: Ejecución SIN corrección EXIF (No recomendado)
    # -----------------------------------------------------------------
    # Útil solo para comparar o si estás seguro de que tu dataloader lo maneja.
    python3 scripts/prepare_data.py --overwrite --stratify --no-fix-exif
```


### 4. Flujo de trabajo: baseline (Faster R-CNN)
Una vez que los datos están en `data/processed`, puedes entrenar y evaluar el modelo baseline.

```bash
# -----------------------------------------------------------------
# Paso 4.1: Iniciar el servidor de MLflow
# -----------------------------------------------------------------
# (Abre una terminal separada)
# Te permitirá ver las curvas de entrenamiento en [http://127.0.0.1:5000](http://127.0.0.1:5000)
mlflow server --host 127.0.0.1 --port 5000

# -----------------------------------------------------------------
# Paso 4.2: Entrenar el modelo Baseline
# -----------------------------------------------------------------
# Lee la configuración de 'config/train_config.yaml'.
# Guarda el mejor modelo en 'models/checkpoints/best_model.pth'.
# Registra todo automáticamente en el servidor de MLflow.
python3 scripts/train_faster_rcnn.py --config config/train_config.yaml

# -----------------------------------------------------------------
# Paso 4.3: Evaluar el modelo Baseline en el Test Set
# -----------------------------------------------------------------
# Carga el mejor modelo guardado y lo ejecuta contra el split 'test'.
# Guarda las métricas finales (mAP, Precision, Recall) en un JSON.
python3 scripts/evaluate_model.py \
    --checkpoint models/checkpoints/best_model.pth \
    --output models/checkpoints/test_metrics.json
```

### 5. Flujo de trabajo: avanzado (YOLOv11)

Para entrenar el modelo YOLO, primero debes convertir los datos (preparados en el Paso 3) al formato específico de YOLO.

```bash
# -----------------------------------------------------------------
# Paso 5.1: Convertir datos a formato YOLO
# -----------------------------------------------------------------
# Lee los JSON de 'data/processed' y crea los .txt y el 'data.yaml'
# en 'data/yolo/'.
python3 scripts/convert_to_yolo.py

# -----------------------------------------------------------------
# Paso 5.2: Entrenar el modelo YOLO
# -----------------------------------------------------------------
python3 scripts/train_yolo.py --config config/train_config_yolo11.yaml
```

## Flujo de MLOps

### Seguimiento de experimentos (MLflow)

Todos los entrenamientos (`train_faster_rcnn.py`) se registran automáticamente en MLflow, capturando:

- Hiperparámetros (batch size, learning rate, etc.).
- Métricas de entrenamiento y validación por época.
- Métricas finales de evaluación del Test Set.
- Artefactos del modelo (el `best_model.pth` y los JSON de métricas).
- Configuración de entrenamiento (`train_config.yaml`).

## Arquitecturas soportadas

El pipeline está diseñado para ser flexible:
- Backbones: ResNet (50, 101), EfficientNet, MobileNetV3, ViT, ConvNeXt.
- Tareas: detección de objetos (Faster R-CNN, RetinaNet, YOLO).

## Contribuir

1. Hacer fork del repositorio
2. Crear una rama para la función (`git checkout -b feature/nueva-funcion`)
3. Hacer commit de los cambios (`git commit -m 'Agregar nueva función'`)
4. Subir a la rama (`git push origin feature/nueva-funcion`)
5. Abrir un Pull Request

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## Contacto

Rodrigo - [@RodrigoGoni](https://github.com/RodrigoGoni)
Tomas - [@tomasctg](https://github.com/tomasctg)

Enlace del Proyecto: [https://github.com/RodrigoGoni/trash-detection](https://github.com/RodrigoGoni/trash-detection)