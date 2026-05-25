import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SEED = 42
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.keras"
SCALER_PATH = ARTIFACT_DIR / "scaler.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"


def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_dataset():
    x, y = make_classification(
        n_samples=1800,
        n_features=8,
        n_informative=6,
        n_redundant=1,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.2,
        flip_y=0.03,
        random_state=SEED,
    )
    return x.astype("float32"), y.astype("int64")


def build_model(feature_count, class_count):
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(feature_count,), name="features"),
            tf.keras.layers.Dense(48, activation="relu"),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(class_count, activation="softmax", name="probabilities"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_save(epochs=20):
    set_seed()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    x, y = make_dataset()
    class_names = ["alpha", "beta", "gamma"]
    feature_columns = [f"feature_{i}" for i in range(x.shape[1])]

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=SEED
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.2,
        stratify=y_train_full,
        random_state=SEED,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype("float32")
    x_valid_scaled = scaler.transform(x_valid).astype("float32")
    x_test_scaled = scaler.transform(x_test).astype("float32")

    model = build_model(feature_count=x.shape[1], class_count=len(class_names))
    model.fit(
        x_train_scaled,
        y_train,
        validation_data=(x_valid_scaled, y_valid),
        epochs=epochs,
        batch_size=32,
        verbose=0,
    )

    probabilities = model.predict(x_test_scaled, verbose=0)
    accuracy = accuracy_score(y_test, probabilities.argmax(axis=1))

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    metadata = {
        "class_names": class_names,
        "feature_columns": feature_columns,
        "feature_count": int(x.shape[1]),
        "model_version": "demo-1.0.0",
        "example_features": x_test[0].astype(float).tolist(),
        "test_accuracy": float(accuracy),
    }
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    saved_metadata = train_and_save()
    print(json.dumps(saved_metadata, indent=2))
