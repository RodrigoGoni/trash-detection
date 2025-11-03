# Trash Detection - Deep Learning Computer Vision Project

A comprehensive deep learning project for **object detection of trash** using the TACO dataset, with integrated MLOps practices for experiment tracking, model versioning, and deployment.

## Project Overview

This project implements a complete pipeline for trash detection using deep learning, including:
- **Object Detection** on TACO (Trash Annotations in Context) dataset
- **60 categories** of trash items with bounding box annotations
- **1500 images** with 4784 annotations (split: 70% train, 15% val, 15% test)
- Advanced preprocessing pipeline with Albumentations (letterboxing, augmentations)
- Multiple model architectures for object detection
- Data preprocessing and augmentation specifically for detection tasks
- Experiment tracking with MLflow
- Model and data versioning with DVC
- REST API for model serving
- Docker containerization
- CI/CD pipeline with GitHub Actions

## Project Structure

```
trash-detection/
├── data/                          # Data directory
│   ├── raw/                       # Raw, immutable data
│   ├── processed/                 # Processed and split data
│   ├── interim/                   # Intermediate transformed data
│   └── external/                  # External data sources
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_exploration/           # Data exploration
│   ├── 02_preprocessing/         # Preprocessing experiments
│   └── 03_modeling/              # Model experiments
│
├── src/                          # Source code
│   ├── data/                     # Data loading and processing
│   │   ├── dataloader.py        # Dataset and DataLoader classes
│   │   ├── preprocessing.py     # Image preprocessing utilities
│   │   └── augmentation.py      # Data augmentation transforms
│   │
│   ├── models/                   # Model architectures
│   │   ├── backbone.py          # Backbone networks
│   │   ├── classifier.py        # Classification models
│   │   ├── detector.py          # Object detection models
│   │   ├── train.py             # Training utilities
│   │   └── evaluate.py          # Evaluation utilities
│   │
│   ├── features/                 # Feature engineering
│   ├── visualization/            # Visualization utilities
│   └── utils/                    # Utility functions
│       ├── mlflow_utils.py      # MLflow tracking
│       └── dvc_utils.py         # DVC utilities
│
├── config/                       # Configuration files
│   └── train_config.yaml        # Training configuration
│
├── models/                       # Trained models
│   ├── checkpoints/             # Training checkpoints
│   └── production/              # Production models
│
├── experiments/                  # Experiment logs and results
│
├── deployment/                   # Deployment files
│   ├── api/                     # FastAPI application
│   │   └── app.py               # API server
│   └── docker/                  # Docker files
│       ├── Dockerfile           # Docker image definition
│       └── docker-compose.yml   # Multi-container setup
│
├── scripts/                      # Utility scripts
│   ├── prepare_data.py          # Data preparation
│   ├── train_model.py           # Training script
│   └── evaluate_model.py        # Evaluation script
│
├── tests/                        # Unit tests
│
├── .github/workflows/            # GitHub Actions
│   └── ci-cd.yml                # CI/CD pipeline
│
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment
├── setup.py                      # Package setup
└── README.md                     # This file
```

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional, for training)
- Docker (optional, for deployment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/RodrigoGoni/trash-detection.git
cd trash-detection
```

2. Create a virtual environment:
```bash
# Using conda
conda env create -f environment.yml
conda activate trash-detection

# Or using venv
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Data Preparation

1. Place your raw data in `data/raw/` following this structure:
```
data/raw/
    ├── class1/
    │   ├── image1.jpg
    │   └── image2.jpg
    ├── class2/
    │   └── image3.jpg
    └── ...
```

2. Prepare the dataset:
```bash
python scripts/prepare_data.py \
    --raw-dir data/raw \
    --processed-dir data/processed \
    --val-split 0.2 \
    --test-split 0.1
```

3. Version the data with DVC:
```bash
dvc init
dvc add data/processed
git add data/processed.dvc .gitignore
git commit -m "Add processed data"
```

## 🎓 Training

### Configure Training

Edit `config/train_config.yaml` to set your training parameters:
```yaml
model:
  backbone: "resnet50"
  num_classes: 10
  
training:
  num_epochs: 100
  batch_size: 32
  lr: 0.001
```

### Start MLflow Tracking Server

```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000
```

### Train the Model

```bash
python scripts/train_model.py \
    --config config/train_config.yaml \
    --epochs 100 \
    --device cuda
```

### Monitor Training

- MLflow UI: http://localhost:5000
- View experiments, metrics, and artifacts

## Evaluation

Evaluate the trained model:

```bash
python scripts/evaluate_model.py \
    --model-path models/checkpoints/best_model.pth \
    --config config/train_config.yaml \
    --split test \
    --output-dir experiments
```

Results will be saved in `experiments/`:
- `evaluation_metrics.json`: Overall metrics
- `classification_report.json`: Per-class metrics
- `confusion_matrix.png`: Confusion matrix visualization

## Deployment

### Local API Server

Run the FastAPI server locally:

```bash
cd deployment/api
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

API documentation: http://localhost:8000/docs

### Docker Deployment

Build and run with Docker:

```bash
docker build -t trash-detection -f deployment/docker/Dockerfile .
docker run -p 8000:8000 trash-detection
```

Or use Docker Compose for the full stack:

```bash
cd deployment/docker
docker-compose up -d
```

Services:
- API: http://localhost:8000
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

### API Usage

```python
import requests

# Make prediction
url = "http://localhost:8000/predict"
files = {"file": open("image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## MLOps Workflow

### Experiment Tracking (MLflow)

All training runs are automatically logged to MLflow:
- Hyperparameters
- Metrics (loss, accuracy)
- Model artifacts
- Training configuration

### Data Versioning (DVC)

Track data and model versions:

```bash
# Add data to DVC
dvc add data/processed

# Add remote storage
dvc remote add -d myremote s3://my-bucket/dvc-storage

# Push data
dvc push

# Pull data
dvc pull
```

### CI/CD Pipeline

The GitHub Actions workflow automatically:
- Runs tests on pull requests
- Builds and pushes Docker images
- Can trigger model training
- Versions models with DVC

## Model Architectures

Supported backbones:
- ResNet (50, 101)
- EfficientNet (B0, B3)
- MobileNetV3
- Vision Transformer (ViT)
- ConvNeXt

Supported tasks:
- Image Classification
- Multi-label Classification
- Object Detection (Faster R-CNN, RetinaNet, YOLOv5)

## Testing

Run tests:

```bash
pytest tests/
```

With coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Documentation

For more detailed documentation, see:
- [Data Processing Guide](docs/data_processing.md)
- [Model Training Guide](docs/training.md)
- [Deployment Guide](docs/deployment.md)
- [API Reference](docs/api_reference.md)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch and torchvision teams
- MLflow and DVC communities
- FastAPI framework
- Open source computer vision community

## Contact

Rodrigo - [@RodrigoGoni](https://github.com/RodrigoGoni)

Project Link: [https://github.com/RodrigoGoni/trash-detection](https://github.com/RodrigoGoni/trash-detection)

---

