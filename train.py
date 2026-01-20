# train.py

# Standard libraries
import os
import json
from pathlib import Path
import shutil

# Database library for MLflow tracking with SQLite
import sqlite3

# Standard data science libraries
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn as mlflow_sklearn

# MLflow client
from mlflow.tracking import MlflowClient

# Load environment variables
from dotenv import load_dotenv

# Data generation and splitting
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Modeling
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Metrics

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Load secrets from .env file

load_dotenv()

# Bad practice (decorative hard-coded secret for the demo)
fraud_api_key = "sk-1234567890"

# Good practice (same secret sourced from .env; keep this line commented for the demo)
# fraud_api_key = os.getenv("FRAUD_API_KEY")

# Require it to run, but it doesn't affect training
if not fraud_api_key:
    raise RuntimeError("Missing FRAUD_API_KEY (required for this demo).")

# Functions
def require_key(fraud_api_key: str) -> None:
    # “Uses” the key without changing any behaviour
    _ = f"key_present={len(fraud_api_key) > 0}"

import sqlite3

def get_tracking_uri(db_dir_name: str = ".mlflow", db_file_name: str = "mlflow.db") -> str:
    project_dir = Path(__file__).resolve().parent
    db_dir = project_dir / db_dir_name
    db_dir.mkdir(parents=True, exist_ok=True)

    db_path = db_dir / db_file_name

    # Create the file if it doesn't exist
    db_path.touch(exist_ok=True)

    # Validate the sqlite file is usable; if not, quarantine and recreate
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.close()
    except sqlite3.OperationalError:
        if db_path.exists():
            bad = db_path.with_suffix(".db.bad")
            try:
                db_path.rename(bad)
            except Exception:
                pass
        db_path.touch(exist_ok=True)

    return f"sqlite:///{db_path.as_posix()}"

# Ensure MLflow experiment exists
def ensure_experiment(
    name: str,
    artifact_dir: str,
    *,
    wipe_if_deleted: bool = True,
) -> str:
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    client = MlflowClient()

    exp = client.get_experiment_by_name(name)

    # Create if it doesn't exist at all
    if exp is None:
        artifact_location = f"file:{Path(artifact_dir).resolve()}"
        return client.create_experiment(name=name, artifact_location=artifact_location)

    # If it exists but was soft-deleted, restore it
    if exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)

        if wipe_if_deleted:
            # Soft-delete all runs inside that experiment
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string="",
                max_results=100000,
            )
            for r in runs:
                client.delete_run(r.info.run_id)

            # Remove the local artifact folders for that experiment (optional but matches “wipe”)
            exp_artifacts_dir = Path(artifact_dir) / exp.experiment_id
            if exp_artifacts_dir.exists():
                shutil.rmtree(exp_artifacts_dir)

    return exp.experiment_id


# Generate synthetic fraud detection data
def make_fraud_data(
    *,
    seed: int,
    n_samples: int,
    pos_ratio: float,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.Series, dict]:
    n_features = len(feature_names)
    weights = [1.0 - pos_ratio, pos_ratio]

    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(4, n_features // 2),
        n_redundant=max(0, n_features // 6),
        n_repeated=0,
        n_clusters_per_class=2,
        weights=weights,
        flip_y=0.01,
        class_sep=1.0,
        random_state=seed,
    )

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(X, columns=feature_names)

    # Light “transaction-like” shaping for storytelling (still synthetic, still deterministic).

    if "amount" in df.columns:
        df["amount"] = np.abs(df["amount"]) * 50 + 10
    if "hour" in df.columns:
        df["hour"] = (np.abs(df["hour"]) * 8).round().clip(0, 23)
    if "device_score" in df.columns:
        df["device_score"] = (df["device_score"] * 20 + 50).clip(0, 100)
    if "account_age_days" in df.columns:
        df["account_age_days"] = (np.abs(df["account_age_days"]) * 120).round().clip(0, 3650)
    if "velocity_1h" in df.columns:
        df["velocity_1h"] = (np.abs(df["velocity_1h"]) * 2.5).round().clip(0, 50)
    if "ip_risk" in df.columns:
        df["ip_risk"] = (df["ip_risk"] * 25 + 50).clip(0, 100)
    if "merchant_risk" in df.columns:
        df["merchant_risk"] = (df["merchant_risk"] * 25 + 50).clip(0, 100)
    if "distance_km" in df.columns:
        df["distance_km"] = (np.abs(df["distance_km"]) * 15).clip(0, 500)
    if "failed_logins_24h" in df.columns:
        df["failed_logins_24h"] = (np.abs(df["failed_logins_24h"]) * 1.7).round().clip(0, 30)
    if "chargeback_history" in df.columns:
        df["chargeback_history"] = (np.abs(df["chargeback_history"]) * 0.7).round().clip(0, 10)

    # Target series
    y_s = pd.Series(y, name="is_fraud").astype(int)

    # Configuration dictionary
    cfg = {
        "seed": seed,
        "n_samples": n_samples,
        "n_features": n_features,
        "pos_ratio_target": pos_ratio,
        "pos_ratio_actual": float(y_s.mean()),
        "feature_names": feature_names,
    }
    return df, y_s, cfg

def load_or_create_dataset(
    *,
    path: Path,
    seed: int,
    n_samples: int,
    pos_ratio: float,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.Series, dict]:

    if path.exists():
        df = pd.read_csv(path)
        y = df["is_fraud"]
        X = df.drop(columns=["is_fraud"])

        cfg = {
            "seed": seed,
            "n_samples": len(df),
            "n_features": X.shape[1],
            "pos_ratio_target": pos_ratio,
            "pos_ratio_actual": float(y.mean()),
            "feature_names": feature_names,
            "source": "loaded_from_disk",
            "path": str(path),
        }

        print(f"Loaded existing dataset from {path}")
        return X, y, cfg

    X, y, cfg = make_fraud_data(
        seed=seed,
        n_samples=n_samples,
        pos_ratio=pos_ratio,
        feature_names=feature_names,
    )

    df = X.copy()
    df["is_fraud"] = y
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    cfg["source"] = "generated"
    cfg["path"] = str(path)

    print(f"Generated and saved dataset to {path}")
    return X, y, cfg


# Compute evaluation metrics
def compute_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    y_pred = (proba >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "precision_fraud": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_fraud": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_fraud": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }

# Log confusion matrix components
def log_confusion_matrix(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    y_pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # [[tn, fp], [fn, tp]]
    tn, fp, fn, tp = cm.ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

# Train and log model run
def train_and_log(
    *,
    run_name: str,
    model_name: str,
    class_weight_strategy: str,
    class_weight,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dataset_cfg: dict,
    seed: int,
    threshold: float,
    feature_names: list[str],
) -> str:
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    class_weight=class_weight,
                    random_state=seed,
                ),
            ),
        ]
    )

    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test.to_numpy(), proba, threshold)
        cm = log_confusion_matrix(y_test.to_numpy(), proba, threshold)

                # Create a single y_pred here for human-readable artifacts (report + model card)
        y_pred = (proba >= threshold).astype(int)

        # Classification report artifact (per run)
        report = classification_report(
            y_test.to_numpy(),
            y_pred,
            target_names=["non_fraud", "fraud"],
            digits=4,
            zero_division=0,
        )
        mlflow.log_text(str(report), artifact_file="classification_report.txt")

        # Per-run model card artifact (per run)
        exp = mlflow.get_experiment(run.info.experiment_id)

        model_card_text = f"""# Model card
            ## Model details
            - Model name: {model_name}
            - Model type: scikit-learn pipeline (StandardScaler + LogisticRegression)
            - MLflow experiment: {exp.name if exp else "unknown"}
            - MLflow run id: {run.info.run_id}

            ## Intended use
            - Primary use case: Demo fraud probability scoring on synthetic tabular transactions.
            - Intended users: Interview demo audience.
            - Output: predict_proba for fraud (class 1).
            - Out of scope: Production fraud decisions.

            ## Data
            - Source: Synthetic data generated at runtime (sklearn.datasets.make_classification).
            - Rows: {dataset_cfg["n_samples"]}
            - Fraud rate (positive class ratio): {dataset_cfg["pos_ratio_actual"]:.4f}
            - Feature list: {", ".join(feature_names)}

            ## Training and evaluation
            - Train/test split: Stratified (test_size=0.25)
            - Random seed: {seed}
            - Threshold for classification metrics: {threshold}
            - Class weighting strategy: {class_weight_strategy}
            - Metrics logged: ROC AUC, average precision (PR AUC), precision/recall/F1 (fraud), confusion matrix counts
            - Artifacts: classification_report.txt, dataset_config.json, model/

            ## Observations
            - Accuracy is not logged because fraud is rare and accuracy hides minority-class behavior.
            - PR AUC, recall, and precision better reflect performance under heavy imbalance.

            ## Security notes (demo)
            - Secret handling includes an intentional hard-coded key for a teachable moment.
            - Secure approach is .env loading via python-dotenv.
            """
        mlflow.log_text(model_card_text, artifact_file="model_card.md")


        # Log parameters and metrics
        mlflow.log_params(
            {
                "model_name": model_name,
                "class_weight_strategy": class_weight_strategy,
                "dataset_imbalance_ratio": dataset_cfg["pos_ratio_actual"],
                "random_seed": seed,
                "threshold": threshold,
                "feature_list": json.dumps(feature_names),
            }
        )
        
        mlflow.log_metrics(metrics)
        mlflow.log_metrics(
            {
                "cm_tn": cm["tn"],
                "cm_fp": cm["fp"],
                "cm_fn": cm["fn"],
                "cm_tp": cm["tp"],
            }
        )

        # Dataset config artifact
        mlflow.log_text(json.dumps(dataset_cfg, indent=2), artifact_file="dataset_config.json")

        # Full pipeline artifact
        mlflow_sklearn.log_model(pipe, artifact_path="model")

        return run.info.run_id


# Main execution
def main() -> None:
    require_key(fraud_api_key)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = "fraud-detection-imbalance-demo"

    # Ensure experiment exists
    ensure_experiment(experiment_name, artifact_dir="artifacts", wipe_if_deleted=True) 
    mlflow.set_experiment(experiment_name)


    seed = 7
    threshold = 0.5

    feature_names = [
        "amount",
        "hour",
        "device_score",
        "account_age_days",
        "velocity_1h",
        "ip_risk",
        "merchant_risk",
        "distance_km",
        "failed_logins_24h",
        "chargeback_history",
        "country_mismatch_score",
        "new_device_flag",
    ]

    # Generate dataset
    '''
    X, y, dataset_cfg = make_fraud_data(
        seed=seed,
        n_samples=2000,
        pos_ratio=0.02,
        feature_names=feature_names,
    )
    '''

    dataset_path = Path("data/fraud_dataset.csv")

    X, y, dataset_cfg = load_or_create_dataset(
        path=dataset_path,
        seed=seed,
        n_samples=2000,
        pos_ratio=0.02,
        feature_names=feature_names,
    )


    # Display dataset sample and fraud rate
    print("Dataset sample:")
    print(pd.concat([X.head(8), y.head(8)], axis=1))
    print(f"Fraud rate (actual): {dataset_cfg['pos_ratio_actual']:.4f}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=seed,
    )

    # Run A: Baseline Logistic Regression without class weighting
    run_a = train_and_log(
        run_name="run_a_baseline_logreg",
        model_name="logistic_regression",
        class_weight_strategy="none",
        class_weight=None,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        dataset_cfg=dataset_cfg,
        seed=seed,
        threshold=threshold,
        feature_names=feature_names,
    )

    # Run B: Logistic Regression with balanced class weighting
    run_b = train_and_log(
        run_name="run_b_balanced_logreg",
        model_name="logistic_regression",
        class_weight_strategy="balanced",
        class_weight="balanced",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        dataset_cfg=dataset_cfg,
        seed=seed,
        threshold=threshold,
        feature_names=feature_names,
    )

    # Output run IDs
    print(f"Run A id: {run_a}")
    print(f"Run B id: {run_b}")

# Entry point
if __name__ == "__main__":
    main()
