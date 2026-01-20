# evaluate.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn as mlflow_sklearn
from mlflow.tracking import MlflowClient

import sqlite3

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

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

def make_fraud_data(seed: int, n_samples: int, pos_ratio: float, feature_names: list[str]) -> tuple[pd.DataFrame, pd.Series, dict]:
    n_features = len(feature_names)
    weights = [1.0 - pos_ratio, pos_ratio]

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

    df = pd.DataFrame(X, columns=feature_names)
    y_s = pd.Series(y, name="is_fraud").astype(int)

    cfg = {
        "seed": seed,
        "n_samples": n_samples,
        "n_features": n_features,
        "pos_ratio_target": pos_ratio,
        "pos_ratio_actual": float(y_s.mean()),
        "feature_names": feature_names,
    }
    return df, y_s, cfg


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    y_pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "precision_fraud": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_fraud": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_fraud": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "cm_tn": int(tn),
        "cm_fp": int(fp),
        "cm_fn": int(fn),
        "cm_tp": int(tp),
    }


def main() -> None:
    # mlflow.set_tracking_uri("sqlite:///mlflow.db")
    tracking_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking uri: {tracking_uri}")

    experiment_name = "fraud-detection-imbalance-demo"

    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"Experiment not found: {experiment_name}")

    # Pull recent runs, then pick the best by average precision.
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=25,
    )

    if not runs:
        raise RuntimeError("No runs found. Run python train.py first.")

    # Keep only training runs that logged the model artifact path.
    training_runs = []
    for r in runs:
        if "average_precision" in r.data.metrics and "model_name" in r.data.params:
            training_runs.append(r)

    if not training_runs:
        raise RuntimeError("No training runs with expected metrics/params found.")

    best = max(training_runs, key=lambda r: r.data.metrics.get("average_precision", float("-inf")))
    best_run_id = best.info.run_id
    best_ap = best.data.metrics.get("average_precision")

    print(f"Best training run id: {best_run_id}")
    print(f"Best training run average precision: {best_ap:.6f}")

    # Load the trained pipeline from the run.
    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow_sklearn.load_model(model_uri)

    # Re-generate deterministic data. This evaluates on the same split recipe.
    seed = int(best.data.params.get("random_seed", "7"))
    threshold = float(best.data.params.get("threshold", "0.5"))
    feature_names = json.loads(best.data.params.get("feature_list"))

    X, y, dataset_cfg = make_fraud_data(
        seed=seed,
        n_samples=2000,
        pos_ratio=0.02,
        feature_names=feature_names,
    )

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=seed,
    )

    proba = model.predict_proba(X_test)[:, 1] # type: ignore[reportOptionalMemberAccess]
    
    metrics = compute_metrics(y_test.to_numpy(), proba, threshold)

    # Log evaluation into a separate run that references the training run id.
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"evaluation_of_{best_run_id}") as eval_run:
        mlflow.set_tag("run_type", "evaluation")
        mlflow.set_tag("training_run_id", best_run_id)
        mlflow.set_tag("evaluated_model_uri", model_uri)

        mlflow.log_params(
            {
                "random_seed": seed,
                "threshold": threshold,
                "dataset_imbalance_ratio": dataset_cfg["pos_ratio_actual"],
                "feature_list": json.dumps(feature_names),
            }
        )

        mlflow.log_metrics({f"eval_{k}": v for k, v in metrics.items()})
        mlflow.log_text(json.dumps(dataset_cfg, indent=2), artifact_file="eval_dataset_config.json")

    print(f"Evaluation run id: {eval_run.info.run_id}")
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
