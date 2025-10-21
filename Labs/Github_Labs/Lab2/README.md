# Using GitHub Actions for Model Training and Versioning

This repository demonstrates how to use GitHub Actions to automate the process of training a machine learning model, storing the model, and versioning it. This allows you to easily update and improve your model in a collaborative environment.

Watch the tutorial video for this lab at [Github action Lab2](https://youtu.be/cj5sXIMZUjQ)


## Prerequisites

- [GitHub](https://github.com) account
- Basic knowledge of Python and machine learning
- Git command-line tool (optional)

## Getting Started

1. **Fork this Repository**: Click the "Fork" button at the top right of this [repository](https://github.com/raminmohammadi/MLOps/) to create your own copy.
3. **Clone Your Repository**:
   ```bash
   git clone https://github.com/your-username/your-forked-repo.git
   cd your-forked-repo

   ```
4. GitHub account
5. Basic knowledge of Python and machine learning
6. Git command-line tool (optional)

# Running the Workflow
## Customize Model Training
1. The default pipeline loads the Breast Cancer Wisconsin dataset from `scikit-learn`, scales the features, and trains a logistic regression classifier. Tweak `src/train_model.py` if you want to plug in a different dataset or model.

## Push Your Changes:
1. Commit your changes and push them to your forked repository.

## GitHub Actions Workflow:
1. Once you push changes to the main branch, the GitHub Actions workflow will be triggered automatically.

## View Workflow Progress:
1. You can track the progress of the workflow by going to the "Actions" tab in your GitHub repository.

## Retrieve the Trained Model:
1. After the workflow completes successfully, the trained model will be stored in the `models/` directory.

# Model Evaluation
The model evaluation is performed automatically within the GitHub Actions workflow. The evaluation results (e.g., F1 Score) are stored in the `metrics/` directory.

# Versioning the Model
Each time you run the workflow, a new version of the model is created and stored. You can access and use these models for your projects.

# GitHub Actions Workflow Details
The workflow consists of the following steps:

- Generate and Store Timestamp: A timestamp is generated and stored in a file for versioning.
- Model Training: The `train_model.py` script loads the Breast Cancer Wisconsin dataset, splits it into train/test sets, and fits a scaled logistic regression model. The trained pipeline is saved with a timestamped filename.
- Model Evaluation: The `evaluate_model.py` script reloads the same dataset split, evaluates the saved model on the test set, and stores accuracy, F1, precision, and recall metrics in the `metrics/` directory.
- Store and Version the New Model: The trained model is moved to the `models/` directory with a timestamp-based version.
- Commit and Push Changes: The metrics and updated model are committed to the repository, allowing you to track changes.

# Automated Retraining Workflows
## Overview
Two GitHub Actions definitions (`model_calibration_on_push.yml` and `model_calibration.yml`) automate retraining and evaluation of the logistic regression pipeline. One runs on every push to `main`, and the other executes on a nightly cron schedule so the model and metrics stay current without manual intervention.

## Workflow Purpose
Each workflow provisions a Python runner, installs the dependencies, trains the pipeline on the Breast Cancer Wisconsin dataset, evaluates it on a held-out test set, and versions the resulting artifacts so you can audit model drift over time.

## Workflow Execution
1. **Trigger**: Either a push to `main` or the scheduled cron event starts the job.  
2. **Environment Setup**: The workflow checks out your fork, installs dependencies from `requirements.txt`, and captures a timestamp used to version artifacts.  
3. **Train**: `src/train_model.py` trains the scaler + logistic regression pipeline and writes `model_<timestamp>_logreg.joblib`.  
4. **Evaluate**: `src/evaluate_model.py` reloads the same dataset split, scores the model, and emits `<timestamp>_metrics.json` containing accuracy, F1, precision, and recall.  
5. **Archive**: Both artifacts are moved into `models/` and `metrics/`, then committed back to the repository using the workflow token.

# Customization
Adjust the dataset loader, preprocessing steps, model choice, or evaluation metrics inside `src/train_model.py` and `src/evaluate_model.py`. If you change artifact names or locations, update the workflow YAML files so the move and commit steps still succeed.

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgments
- This project uses GitHub Actions for continuous integration and deployment.
- Model training and evaluation are powered by Python and scikit-learn.

# Questions or Issues
If you have any questions or encounter issues while using this GitHub Actions workflow, please open an issue in the Issues section of your repository.

