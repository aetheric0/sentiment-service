# ----------------------------------------------
# XGBoost + spaCy Upgrade (v2, cached)
# ----------------------------------------------
import json
import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from datasets import load_dataset
from joblib import Memory
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from app.spacy_preprocessor import SpacyPreprocessor

# -------------------------------
# 0) Cache setup
# -------------------------------
# You can override CACHE_DIR and CLEAR_CACHE via environment variables:
#   CACHE_DIR=/tmp/sentiment_cache CLEAR_CACHE=1 python notebooks/sentiment_xgb.py
CACHE_DIR = os.getenv("CACHE_DIR", os.path.join(os.getcwd(), "pipeline_cache"))
os.makedirs(CACHE_DIR, exist_ok=True)
memory = Memory(location=CACHE_DIR, verbose=0)

# -------------------------------
# 1) Load & split data
# -------------------------------
dataset = load_dataset("imdb")
df = pd.DataFrame(dataset["train"])
X_train, X_val, y_train, y_val = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# -------------------------------
# 2) Define base pipeline with caching
# -------------------------------
base_pipeline = Pipeline([
    ("spacy", SpacyPreprocessor()),
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("xgb", XGBClassifier(
        eval_metric="logloss",
        n_jobs=-1
    )),
], memory=memory)

# Optionally clear previous cached steps (useful after code/param changes)
if os.getenv("CLEAR_CACHE", "0") == "1":
    try:
        memory.clear(warn=False)
        print(f"🧹 Cleared pipeline cache at: {CACHE_DIR}")
    except Exception as e:
        print(f"Cache clear skipped: {e}")

# -------------------------------
# 3) Hyperparameter tuning
# -------------------------------
param_grid = {
    "xgb__n_estimators": [100, 200],
    "xgb__max_depth": [4, 6],
    "xgb__learning_rate": [0.1, 0.01],
}

grid = GridSearchCV(
    base_pipeline,
    param_grid,
    cv=3,
    scoring="accuracy",
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)

# -------------------------------
# 4) Probability calibration on validation set
# -------------------------------
best_pipeline = grid.best_estimator_

tfidf = best_pipeline.named_steps["tfidf"]
xgb_base = best_pipeline.named_steps["xgb"]

calibrator = CalibratedClassifierCV(
    estimator=xgb_base,
    method="isotonic",
    cv="prefit"
)

# Transform once and reuse (also benefits from joblib cache)
X_val_vec = tfidf.transform(X_val)
calibrator.fit(X_val_vec, y_val)

# -------------------------------
# 5) Wrap calibrated model back into full pipeline
# -------------------------------
calibrated_pipeline = Pipeline([
    ("spacy", best_pipeline.named_steps["spacy"]),
    ("tfidf", tfidf),
    ("xgb", calibrator)
], memory=memory)

# -------------------------------
# 6) Reliability diagram
# -------------------------------
probs = calibrated_pipeline.predict_proba(X_val)[:, 1]
frac_pos, mean_pred = calibration_curve(y_val, probs, n_bins=10)

plt.figure(figsize=(6, 6))
plt.plot(mean_pred, frac_pos, marker="o", label="XGB Calibrated")
plt.plot([0, 1], [0, 1], "--", label="Perfectly Calibrated")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Reliability Diagram")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# 7) Metrics & save
# -------------------------------
y_pred = calibrated_pipeline.predict(X_val)
metrics = {
    "accuracy": accuracy_score(y_val, y_pred),
    "precision": precision_score(y_val, y_pred),
    "recall": recall_score(y_val, y_pred),
    "f1": f1_score(y_val, y_pred)
}

print(classification_report(y_val, y_pred))

os.makedirs("models", exist_ok=True)
with open("models/xgb_metrics_v2.json", "w") as f:
    json.dump(metrics, f, indent=2)

joblib.dump(calibrated_pipeline, "models/sentiment_model_xgb_v2.pkl")
print(f"✅ Calibrated spaCy+XGBoost model saved.\n🗂  Pipeline cache directory: {CACHE_DIR}")
