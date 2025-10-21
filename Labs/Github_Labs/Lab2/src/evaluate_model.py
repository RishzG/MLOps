import argparse
import json
import os

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def load_dataset():
    dataset = load_breast_cancer(as_frame=True)
    return dataset.data, dataset.target


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    timestamp = args.timestamp
    model_version = f"model_{timestamp}_logreg"
    model_path = os.path.join("models", f"{model_version}.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Expected trained model at {model_path}")

    model = joblib.load(model_path)

    X, y = load_dataset()
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_predict = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_predict),
        "f1_score": f1_score(y_test, y_predict),
        "precision": precision_score(y_test, y_predict),
        "recall": recall_score(y_test, y_predict),
    }

    metrics_dir = "metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_path = os.path.join(metrics_dir, f"{timestamp}_metrics.json")
    with open(metrics_path, "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=4)
