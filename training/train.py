"""
Credit Card Fraud Detection — Model Training Pipeline
======================================================
Trains a RandomForestClassifier on synthetic fraud data, evaluates it across
multiple metrics, and saves the model artifact. Optionally logs everything to
MLflow for experiment tracking.

Why RandomForest?
- Strong baseline for tabular data with minimal tuning
- Handles class imbalance reasonably well with class_weight="balanced"
- Feature importances help explain which signals drive fraud detection
- Fast enough to iterate quickly during development

MLflow integration is wrapped in try/except so training works standalone
(no MLflow server required) — important for CI and local development.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load the transaction dataset from CSV.
    Validates that required columns exist before returning.
    """
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        print("Run 'python download_data.py' first to generate the data.")
        sys.exit(1)

    df = pd.read_csv(data_path)

    required_cols = ["is_fraud", "amount", "time"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    return df


def preprocess(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list[str]]:
    """
    Preprocess the dataset for training.

    Steps:
    1. Separate features from the target label
    2. Split into train/test sets (stratified to preserve fraud ratio)
    3. Scale features with StandardScaler (important for consistent distributions)

    Returns train/test arrays, the fitted scaler, and feature names.
    """
    feature_cols = [c for c in df.columns if c != "is_fraud"]
    X = df[feature_cols].values
    y = df["is_fraud"].values

    # Stratified split preserves the fraud/legitimate ratio in both sets.
    # test_size=0.2 is standard — enough test data for reliable metrics.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # StandardScaler normalizes features to zero mean, unit variance.
    # Fit ONLY on training data to prevent data leakage from test set.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, feature_cols


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier with balanced class weights.

    class_weight="balanced" automatically adjusts weights inversely proportional
    to class frequencies — critical for imbalanced fraud data where the model
    would otherwise learn to predict "not fraud" for everything and get 98% accuracy.
    """
    params = {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """
    Evaluate the model across metrics that matter for fraud detection.

    Why these metrics?
    - Accuracy: overall correctness (misleading with imbalanced data, but good sanity check)
    - Precision: of predicted frauds, how many are real? (high = fewer false alarms)
    - Recall: of actual frauds, how many did we catch? (high = fewer missed frauds)
    - F1: harmonic mean of precision/recall (balances both concerns)
    - AUC-ROC: ranking quality — can the model separate fraud from legit?
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_prob),
    }

    return metrics


def save_reference_distributions(
    X_train: np.ndarray,
    feature_cols: list[str],
    output_path: str,
) -> None:
    """
    Save training data distributions as a reference for drift detection.

    The drift detector compares incoming prediction data against these reference
    distributions. We store mean, std, and histogram bins for each feature —
    enough to compute PSI and KS statistics.
    """
    distributions: dict = {}

    for i, col in enumerate(feature_cols):
        values = X_train[:, i]
        # 20 bins gives a good balance between granularity and noise
        hist, bin_edges = np.histogram(values, bins=20, density=True)

        distributions[col] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "histogram": {
                "counts": [float(h) for h in hist],
                "bin_edges": [float(b) for b in bin_edges],
            },
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(distributions, f, indent=2)

    print(f"  Reference distributions saved to: {output_path}")


def main() -> None:
    """
    Full training pipeline: load → preprocess → train → evaluate → save.
    MLflow tracking is optional — pipeline works without a server running.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "transactions.csv")
    model_dir = os.path.join(base_dir, "..", "models")
    os.makedirs(model_dir, exist_ok=True)

    # ── Load and preprocess ──────────────────────────────────
    print("Loading dataset...")
    df = load_data(data_path)
    print(f"  Shape: {df.shape}, Fraud rate: {df['is_fraud'].mean():.2%}")

    print("Preprocessing...")
    X_train, X_test, y_train, y_test, scaler, feature_cols = preprocess(df)
    print(f"  Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    # ── Train ────────────────────────────────────────────────
    print("Training RandomForestClassifier...")
    model = train_model(X_train, y_train)

    # ── Evaluate ─────────────────────────────────────────────
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    print("\n  Model Performance:")
    for name, value in metrics.items():
        print(f"    {name:>10}: {value:.4f}")

    print(
        f"\n  Classification Report:\n{classification_report(y_test, model.predict(X_test))}"
    )

    # ── Save artifacts ───────────────────────────────────────
    model_path = os.path.join(model_dir, "model.joblib")
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    ref_dist_path = os.path.join(
        base_dir, "..", "drift-detector", "reference_distributions.json"
    )

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n  Model saved to: {model_path}")
    print(f"  Scaler saved to: {scaler_path}")

    # Save reference distributions for the drift detector
    save_reference_distributions(X_train, feature_cols, ref_dist_path)

    # ── MLflow tracking (optional) ───────────────────────────
    # Wrapped in try/except so training works without MLflow server.
    # When MLflow is running (e.g., via docker compose), this logs the full
    # experiment — parameters, metrics, and the model artifact — for
    # reproducibility and comparison across training runs.
    try:
        import mlflow
        import mlflow.sklearn

        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("fraud-detection")

        with mlflow.start_run(run_name="random-forest-baseline"):
            # Log hyperparameters so we can compare across experiments
            mlflow.log_params(
                {
                    "n_estimators": 100,
                    "max_depth": 15,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "class_weight": "balanced",
                    "test_size": 0.2,
                    "n_samples": len(df),
                }
            )

            mlflow.log_metrics(metrics)

            # Log the model with its input signature for serving
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name="fraud-detector",
            )

        print("  MLflow tracking: logged successfully")

    except Exception as e:
        print(f"  MLflow tracking: skipped ({e})")
        print("  (This is fine — model was saved locally)")


if __name__ == "__main__":
    main()
