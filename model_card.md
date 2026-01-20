# Model card

## Model details
- Model name:
- Model type:
- Version:
- Owner:
- MLflow experiment:
- MLflow run id:

## Intended use
- Primary use case:
- Users:
- Decisions supported:
- Out of scope:

## Data
- Source: Synthetic data generated at runtime (sklearn.datasets.make_classification).
- Rows:
- Fraud rate (positive class ratio):
- Feature list:

## Training and evaluation
- Train/test split: Stratified
- Random seed:
- Preprocessing: StandardScaler in a scikit-learn Pipeline
- Threshold for classification metrics:
- Metrics logged:
  - ROC AUC
  - Average precision (PR AUC)
  - Precision (fraud)
  - Recall (fraud)
  - F1 (fraud)
  - Confusion matrix

## Performance notes
- Summary of strengths:
- Summary of weaknesses:
- Known failure modes:

## Fairness and responsible use
- Potential harms:
- Monitoring plan:

## Security and compliance notes
- Secret handling:
  - Bad practice example included for demo (hard-coded key).
  - Secure approach: load from .env via python-dotenv.
- Traceability:
  - Runs, params, metrics, and artefacts stored in MLflow (SQLite backend).
