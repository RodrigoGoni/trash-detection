"""
FastAPI application for serving trash detection model
"""

from src.data.preprocessing import ImagePreprocessor
from src.models.classifier import TrashClassifier
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))


# API Models
class PredictionResponse(BaseModel):
    """Response model for predictions"""
    class_name: str
    class_id: int
    confidence: float
    all_predictions: Optional[Dict[str, float]] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    version: str


# Initialize FastAPI app
app = FastAPI(
    title="Trash Detection API",
    description="API for trash detection using deep learning",
    version="1.0.0"
)


# Global model variable
model = None
device = None
class_names = None
preprocessor = None


def load_model(model_path: str, num_classes: int, device_name: str = "cpu"):
    """Load the trained model"""
    global model, device, preprocessor

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = TrashClassifier(num_classes=num_classes, backbone="resnet50")

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set to evaluation mode
    model.to(device)
    model.eval()

    # Initialize preprocessor
    preprocessor = ImagePreprocessor(target_size=(224, 224))

    print(f"Model loaded successfully on {device}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    # TODO: Configure these from environment variables
    MODEL_PATH = "./models/production/best_model.pth"
    NUM_CLASSES = 10
    DEVICE = "cuda"

    try:
        load_model(MODEL_PATH, NUM_CLASSES, DEVICE)
    except Exception as e:
        print(f"Warning: Could not load model: {e}")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return {
        "status": "running",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict trash type from uploaded image

    Args:
        file: Uploaded image file

    Returns:
        Prediction result
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)

        # Preprocess
        image_tensor = preprocessor(image_np)
        image_tensor = torch.from_numpy(
            image_tensor).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(device)

        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)

        # Get top predictions
        top_probs, top_indices = probabilities[0].topk(5)

        all_predictions = {}
        if class_names:
            all_predictions = {
                class_names[idx.item()]: prob.item()
                for idx, prob in zip(top_indices, top_probs)
            }
        else:
            all_predictions = {
                f"class_{idx.item()}": prob.item()
                for idx, prob in zip(top_indices, top_probs)
            }

        # Prepare response
        predicted_class = predicted.item()
        class_name = class_names[
            predicted_class] if class_names else f"class_{predicted_class}"

        return {
            "class_name": class_name,
            "class_id": predicted_class,
            "confidence": confidence.item(),
            "all_predictions": all_predictions
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict trash types for multiple images

    Args:
        files: List of uploaded image files

    Returns:
        List of prediction results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(files) > 10:
        raise HTTPException(
            status_code=400, detail="Maximum 10 images per batch")

    results = []

    for file in files:
        try:
            result = await predict(file)
            results.append({
                "filename": file.filename,
                "prediction": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return {"predictions": results}


@app.get("/classes")
async def get_classes():
    """Get list of available classes"""
    if class_names:
        return {"classes": class_names}
    else:
        return {"classes": [f"class_{i}" for i in range(10)]}  # Default


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
