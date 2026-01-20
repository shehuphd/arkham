# Fraud detection imbalance demo

By Mo Shehu

This repository is a minimal example of a fraud detection training pipeline that focuses on three things:

* Handling class imbalance correctly
* Making ML experiments reproducible and auditable
* Demonstrating basic security and governance practices

It uses MLflow with a SQLite backend to track models, parameters, metrics, and artifacts.

---

## What it does

The project trains two models on the same synthetic fraud dataset:

1. A baseline logistic regression model
2. An imbalance-aware logistic regression model using class weighting

Both runs are logged to MLflow so they can be compared in the UI.

Each run records:

* Parameters
* Evaluation metrics
* The trained model pipeline
* A classification report
* A model card
* Dataset configuration

---

## Structure

```
arkham/
├── train.py
├── evaluate.py
├── model_card.md
├── mlflow.db
├── .env
├── artifacts/
└── data/
└── fraud_dataset.csv
```

---

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install mlflow scikit-learn pandas numpy python-dotenv
```

Create a `.env` file:

```
FRAUD_API_KEY=sk-1234567890
```

The key is decorative and used only to demonstrate secret handling.

---

## Run

From inside `arkham/`:

Train models:

```
python train.py
```

Evaluate best model:

```
python evaluate.py
```

Start MLflow UI:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

If you get an error, you might need to kill any running processes on that port:

`lsof -i :5000`
`kill -9 <PID>` where <PID> is the process ID.

Alternatively, use a different port like 5050. 

Open:

```
[http://127.0.0.1:5050](http://127.0.0.1:5000)
```

---

## Notes

* The dataset is synthetic and stored in `data/fraud_dataset.csv`.
* It is created only if it does not already exist.
* Secrets should never be hard-coded in real systems.
* MLflow artifacts provide traceability and auditability.
  ```
