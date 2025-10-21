import pytest

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_pipeline():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=0)),
        ]
    )

@pytest.mark.parametrize("test_size", [0.2, 0.3])
def test_pipeline_f1_score_is_reasonable(test_size):
    dataset = load_breast_cancer(as_frame=True)
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    score = f1_score(y_test, predictions)
    assert score > 0.9, f"Expected F1 score > 0.9, got {score}"
