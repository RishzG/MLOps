# Lab2 Updates Snapshot

- Swapped synthetic data for the sklearn Breast Cancer Wisconsin dataset and rebuilt the training script around a `StandardScaler + LogisticRegression` pipeline with MLflow logging.
- Evaluation script now reloads the same dataset split, reports accuracy/F1/precision/recall, and writes artifacts under `Labs/Github_Labs/Lab2/metrics/` and `models/`.
- GitHub Actions workflows were moved into `.github/workflows/`, scoped to the Lab2 directory, and updated to save artifacts plus attribute commits to the triggering actor.
- Root `.gitignore` now keeps Lab2 metrics JSON files so workflow runs can version metrics alongside models.
- Added an automated pytest (`tests/test_training_pipeline.py`) that validates the logistic regression pipeline achieves >0.9 F1 on multiple splits, with the Lab2 workflow running it on every push.
