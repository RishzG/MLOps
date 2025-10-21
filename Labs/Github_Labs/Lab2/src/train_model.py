import argparse
import datetime
import os

import mlflow
from joblib import dump
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_dataset():
    """Load the breast cancer dataset with feature names."""
    dataset = load_breast_cancer(as_frame=True)
    return dataset.data, dataset.target, dataset.frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    timestamp = args.timestamp
    print(f"Timestamp received from GitHub Actions: {timestamp}")

    X, y, dataset_frame = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=0)),
        ]
    )

    pipeline.fit(X_train, y_train)

    train_predictions = pipeline.predict(X_train)
    test_predictions = pipeline.predict(X_test)

    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "sklearn_breast_cancer"
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"
    experiment_id = mlflow.create_experiment(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id, run_name=dataset_name):
        mlflow.log_params(
            {
                "dataset_name": dataset_name,
                "num_rows": dataset_frame.shape[0],
                "num_features": dataset_frame.shape[1] - 1,  # exclude target column
                "model": "LogisticRegression",
                "scaler": "StandardScaler",
            }
        )

        mlflow.log_metrics(
            {
                "train_accuracy": accuracy_score(y_train, train_predictions),
                "train_f1": f1_score(y_train, train_predictions),
                "test_accuracy": accuracy_score(y_test, test_predictions),
                "test_f1": f1_score(y_test, test_predictions),
            }
        )

    if not os.path.exists("models/"):
        os.makedirs("models/")

    model_version = f"model_{timestamp}"
    model_filename = f"{model_version}_logreg.joblib"
    model_path = os.path.join("models", model_filename)
    dump(pipeline, model_path)

