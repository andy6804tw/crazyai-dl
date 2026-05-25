from train_model import train_and_save


metadata = train_and_save(epochs=3)

from app import BatchPredictionRequest, PredictionRequest, health, predict, predict_batch  # noqa: E402


def test_health():
    body = health()
    assert body["status"] == "ok"
    assert body["feature_count"] == metadata["feature_count"]


def test_predict():
    body = predict(PredictionRequest(features=metadata["example_features"]))
    assert "predicted_class" in body
    assert "predicted_label" in body
    assert "probabilities" in body


def test_predict_batch():
    body = predict_batch(
        BatchPredictionRequest(
            items=[PredictionRequest(features=metadata["example_features"])]
        )
    )
    assert len(body["predictions"]) == 1


if __name__ == "__main__":
    test_health()
    test_predict()
    test_predict_batch()
    print("All FastAPI checks passed.")
