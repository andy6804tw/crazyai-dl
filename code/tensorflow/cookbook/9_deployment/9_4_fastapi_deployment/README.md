# 9.4 FastAPI Deployment

This project shows a minimal TensorFlow/Keras inference API with FastAPI.

## Files

| File | Purpose |
|---|---|
| `train_model.py` | Train a small demo classifier and export model artifacts. |
| `app.py` | Load artifacts and serve prediction endpoints. |
| `test_app.py` | Run basic API logic checks. |
| `requirements.txt` | Runtime dependencies for the API example. |

## 1. Train model artifacts

```bash
python train_model.py
```

This creates:

```text
artifacts/
├── model.keras
├── scaler.joblib
└── metadata.json
```

## 2. Run API

```bash
uvicorn app:app --reload
```

Open the generated API docs:

```text
http://127.0.0.1:8000/docs
```

## 3. Test API logic

```bash
python test_app.py
```

## 4. Example requests

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Single prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[0.2,-1.1,0.5,1.0,0.3,-0.7,0.8,1.2]}'
```

Batch prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d '{"items":[{"features":[0.2,-1.1,0.5,1.0,0.3,-0.7,0.8,1.2]}]}'
```

## Notes

- The API expects raw feature values. It applies the saved `StandardScaler` before inference.
- Keep `model.keras`, `scaler.joblib`, and `metadata.json` versioned together.
- Replace the synthetic data in `train_model.py` with your own dataset when adapting this example.
