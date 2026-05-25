import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.keras"
SCALER_PATH = ARTIFACT_DIR / "scaler.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"


class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="Raw numeric features in training column order.")


class BatchPredictionRequest(BaseModel):
    items: List[PredictionRequest]


def load_artifacts():
    missing = [path.name for path in [MODEL_PATH, SCALER_PATH, METADATA_PATH] if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing artifacts: {missing}. Run `python train_model.py` first.")

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, scaler, metadata


model, scaler, metadata = load_artifacts()
app = FastAPI(title="TensorFlow 101 FastAPI Deployment", version=metadata["model_version"])


def predict_one(features):
    expected_count = metadata["feature_count"]
    if len(features) != expected_count:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {expected_count} features, got {len(features)}.",
        )

    raw = np.array(features, dtype="float32").reshape(1, -1)
    scaled = scaler.transform(raw).astype("float32")
    probabilities = model.predict(scaled, verbose=0)[0]
    predicted_class = int(np.argmax(probabilities))
    class_names = metadata["class_names"]
    return {
        "predicted_class": predicted_class,
        "predicted_label": class_names[predicted_class],
        "confidence": float(probabilities[predicted_class]),
        "probabilities": {
            class_name: float(probabilities[index])
            for index, class_name in enumerate(class_names)
        },
        "model_version": metadata["model_version"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": metadata["model_version"],
        "feature_count": metadata["feature_count"],
        "class_names": metadata["class_names"],
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    return predict_one(request.features)


@app.post("/predict-batch")
def predict_batch(request: BatchPredictionRequest):
    if not request.items:
        raise HTTPException(status_code=422, detail="items must not be empty.")
    return {"predictions": [predict_one(item.features) for item in request.items]}
