# Detección de basura - proyecto de visión por computadora con aprendizaje profundo

Un proyecto de vision por computadora para la detección de objetos de basura utilizando el conjunto de datos TACO, con prácticas MLOps integradas para seguimiento de experimentos, versionado de modelos y despliegue.

## Descripción general del proyecto

Este proyecto implementa un sistema completo para la detección de basura utilizando tecnicas y modelos de vision artificial, incluyendo:
- Detección de objetos usando el conjunto de datos *Taco Trash Dataset* 
- 60 categorías de basura con anotaciones de cajas
- 1500 imágenes con 4784 anotaciones (división: 70% entrenamiento, 15% validación, 15% prueba)
- Sistema avanzado de preprocesamiento con Albumentations
- Múltiples arquitecturas de modelos para detección de objetos
- Preprocesamiento y aumento de datos específico para tareas de detección
- Seguimiento de experimentos con MLflow
- Contenedorización con Docker

## Estructura del Proyecto

```
trash-detection/
├── data/                        # Directorio de datos
│   ├── raw/                     # Datos sin procesar
│   ├── processed/               # Datos procesados y divididos
│   ├── interim/                 # Datos transformados intermedios
│   └── external/                # Fuentes de datos externos
│
├── notebooks/                   # Notebooks de jupyter
│   ├── 01_exploration/          # EDA
│   ├── 02_preprocessing/        # Experimentos de preprocesamiento
│   └── 03_evalution/            # Evaluaciones
│
├── src/                         # Código fuente
│   ├── data/                    # Carga y procesamiento de datos
│   │   ├── dataloader.py        # Clases Dataset y DataLoader
│   │   ├── preprocessing.py     # Utilidades de preprocesamiento
│   │   └── augmentation.py      # Transformaciones
│   │
│   ├── models/                  # Modelos
│   │   ├── backbone.py          # Backbones 
│   │   ├── detector.py          # Modelos de detección
│   │   ├── train.py             # Utilidades de entrenamiento
│   │   └── evaluate.py          # Utilidades de evaluación
│   │
│   ├── features/                # Ingeniería de características
│   ├── visualization/           # Utilidades de visualización
│   └── utils/                   # Funciones utilitarias
│       └── mlflow_utils.py      # Seguimiento MLflow
│
├── config/                      # Archivos de configuración
│   └── train_config.yaml        # Configuración de entrenamiento
│
├── models/                      # Modelos entrenados
│   ├── checkpoints/             # Guardado de pesos sinapticos
│   └── production/              # Modelos de producción
│
├── experiments/                 # Registros y resultados
│
├── scripts/                     
│   ├── prepare_data.py          # Preparación de datos
│   ├── train_model.py           # Script de entrenamiento
│   └── evaluate_model.py        # Script de evaluación
│
├── .github/workflows/           # GitHub Actions
│   └── ci-cd.yml                # Pipeline CI/CD
│
├── requirements.txt             # Dependencias Python
├── environment.yml              # Entorno Conda
├── setup.py                     # Configuración del paquete
└── README.md                    
```

## Inicio

### Requisitos Previos

- Python 3.10+
- GPU compatible con CUDA (opcional, para entrenamiento)
- Docker (opcional, para despliegue)

### Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/RodrigoGoni/trash-detection.git
cd trash-detection
```

2. Crear un entorno virtual:
```bash
# Usando conda
conda env create -f environment.yml
conda activate trash-detection

# O usando venv
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

3. Instalar el paquete en modo desarrollo:
```bash
pip install -e .
```

4. Configurar variables de entorno:
```bash
cp .env.example .env
# Editar .env con tu configuración
```

### Preparación de Datos

1. Coloca tus datos sin procesar en `data/raw/` siguiendo esta estructura:
```
data/raw/
    ├── class1/
    │   ├── image1.jpg
    │   └── image2.jpg
    ├── class2/
    │   └── image3.jpg
    └── ...
```

2. Preparar el conjunto de datos:
```bash
python scripts/prepare_data.py \
    --raw-dir data/raw \
    --processed-dir data/processed \
    --val-split 0.2 \
    --test-split 0.1
```

## Entrenamiento

### Configurar Entrenamiento

Edita `config/train_config.yaml` para establecer tus parámetros de entrenamiento:
```yaml
model:
  backbone: "resnet50"
  num_classes: 10
  
training:
  num_epochs: 100
  batch_size: 32
  lr: 0.001
```
### Entrenar el Modelo

```bash
python scripts/train_model.py \
    --config config/train_config.yaml \
    --epochs 100 \
    --device cuda
```

### Monitorear el Entrenamiento

- Interfaz MLflow: http://localhost:5000
- Ver experimentos, métricas y artefactos

## Evaluación

Evaluar el modelo entrenado:

```bash
python scripts/evaluate_model.py \
    --model-path models/checkpoints/best_model.pth \
    --config config/train_config.yaml \
    --split test \
    --output-dir experiments
```

Los resultados se guardarán en `experiments/`:
- `evaluation_metrics.json`: Métricas generales
- `classification_report.json`: Métricas por clase
- `confusion_matrix.png`: Visualización de la matriz de confusión


## Flujo de MLOps

### Seguimiento de Experimentos (MLflow)

Todos los entrenamientos se registran automáticamente en MLflow:
- Hiperparámetros
- Métricas (pérdida, precisión)
- Artefactos del modelo
- Configuración de entrenamiento

## Arquitecturas de Modelos

Backbones soportados:
- ResNet (50, 101)
- EfficientNet (B0, B3)
- MobileNetV3
- Vision Transformer (ViT)
- ConvNeXt

Tareas soportadas:
- Detección de Objetos (Faster R-CNN, RetinaNet, YOLOv5)


## Documentación

Para documentación más detallada, ver:
- [Guía de Procesamiento de Datos](docs/data_processing.md)
- [Guía de Entrenamiento de Modelos](docs/training.md)
- [Guía de Despliegue](docs/deployment.md)
- [Referencia de la API](docs/api_reference.md)

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

---

## Handling Class Imbalance

El dataset TACO presenta un **desbalanceo extremo de clases** con 36 clases minoritarias (<50 anotaciones) y algunas con solo 1-2 muestras. Para manejar esto, el proyecto implementa:

### Class Weighting Strategies

Tres métodos de ponderación de clases en `src/training/class_weights.py`:

1. **Inverse Frequency**: Peso inversamente proporcional a la frecuencia
2. **Effective Number** (recomendado): Basado en el paper "Class-Balanced Loss" (CVPR 2019)
3. **Square Root Inverse**: Balance intermedio

```python
from src.training.class_weights import compute_class_weights

weights = compute_class_weights(
    class_counts=class_counts,
    num_classes=60,
    method='effective',  # 'inverse', 'effective', or 'sqrt'
    beta=0.9999          # Para 'effective' method
)
```

### Specialized Loss Functions

Implementado en `src/training/losses.py`:

1. **Focal Loss**: Reduce el peso de ejemplos fáciles, se enfoca en casos difíciles
   - Paper: "Focal Loss for Dense Object Detection" (Lin et al., ICCV 2017)
   - `gamma=2.0` (recomendado)

2. **Class-Balanced Focal Loss** (recomendado): Combina Focal Loss con effective number weighting
   - Paper: "Class-Balanced Loss Based on Effective Number of Samples" (Cui et al., CVPR 2019)
   - `beta=0.9999` para TACO dataset

### Configuración en `train_config.yaml`

```yaml
training:
  loss:
    type: "cb_focal"              # 'ce', 'focal', or 'cb_focal'
    use_class_weights: true       # Habilitar pesos de clase
    class_weight_method: "effective"  # 'inverse', 'effective', or 'sqrt'
    gamma: 2.0                    # Parámetro de enfoque para Focal Loss
    beta: 0.9999                  # Beta para Class-Balanced Loss (0.999-0.9999)
    bbox_loss_weight: 1.0         # Peso para bbox regression loss
    use_weighted_bbox: true       # Aplicar pesos a bbox loss
```

### Análisis del Dataset

El script `prepare_data.py` genera splits con stratification:

```bash
python scripts/prepare_data.py --stratify --overwrite
```

**Distribución de clases minoritarias (<50 anotaciones):**
- 36 clases minoritarias identificadas
- Rango: 1-49 anotaciones por clase
- Ejemplos: "Carded blister pack" (1), "Battery" (2), "Pizza box" (3)

**Resultados del split estratificado:**
- Train: 1049 imágenes (3309 anotaciones, 371 de clases minoritarias)
- Val: 226 imágenes (835 anotaciones, 88 de clases minoritarias)
- Test: 225 imágenes (640 anotaciones, 84 de clases minoritarias)

### Testing

Validar la implementación:

```bash
python scripts/test_class_weights.py
```

Este script valida:
- ✓ Cálculo de pesos de clase
- ✓ Focal Loss y Class-Balanced Focal Loss
- ✓ Integración con el modelo Faster R-CNN

---

El stratify SÍ funcionó bien para las clases con suficientes muestras. El problema es el desbalanceo extremo inherente del dataset TACO.