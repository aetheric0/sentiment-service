# ----------------------------------------------
# XGBoost + spaCy Upgrade (v2)
# ----------------------------------------------
import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from app.spacy_preprocessor import SpacyPreprocessor

# 1. Load & split
dataset = load_dataset("imdb")
df = pd.DataFrame(dataset["train"])
X_train, X_val, y_train, y_val = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# 2. Define pipeline with spaCy -> TFIDF -> XGBoost
pipeline = Pipeline(
    [
        ("spacy", SpacyPreprocessor()),
        ("tfidf", TfidfVectorizer(max_features=5000)),
        (
            "xgb",
            XGBClassifier(
                use_label_encoder=False,
                eval_metrics="logloss",
                n_jobs=-1
            ),
        ),
    ]
)

# 3. (Optional) Hyperparameter tuning
param_grid = {
    "xgb__n_estimators": [100, 200],
    "xgb__max_depth": [4, 6],
    "xgb__learning_rate": [0.1, 0.01],
}
grid = GridSearchCV(
    pipeline, param_grid, cv=3, scoring="accuracy", verbose=1, n_jobs=-1
)
grid.fit(X_train, y_train)
best_pipeline = grid.best_estimator_
print("Best params", grid.best_params_)

# 3.5 Calibrate the best model's probabilities

# extract trained steps
tfidf = best_pipeline.naemd_steps["tfidf"]
xgb_base = best_pipeline.named_steps["xgb"]

# Wrap in calibrator (using isotonic / prefitting on validation set)
calibrator = CalibratedClassifierCV(
    base_estimator=xgb_base, method="isotonic", cv="prefit"
)
# transform val set into TF-IDF vectors
X_val_vec = tfidf.transform(X_val)
calibrator.fit(X_val_vec, y_val)

probs = calibrator.predict_proba(X_val_vec)[:, 1]

# Compute observed vs. predicted
frac_pos, mean_pred = calibration_curve(y_val, probs, n_bins=10)

# Plotting
plt.figure(figsize=(6, 6))
plt.plot(mean_pred, frac_pos, marker="o", label="XGB Calibrated")
plt.plot([0, 1], [0, 1], "--", label="Perfectly Calibrated")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Reliability Diagram")
plt.legend()
plt.grid(True)
plt.show()

best_pipeline.steps[-1] = ("xgb", calibrator)


# 4. Evaluate & log v2 metrics
y_pred = best_pipeline.predict(X_val)
metrics = {
    "accuracy": accuracy_score(y_val, y_pred),
    "precision": precision_score(y_val, y_pred),
    "recall": recall_score(y_val, y_pred),
    "f1": f1_score(y_val, y_pred),
}
print(classification_report(y_val, y_pred))

os.makedirs("models/", exist_ok=True)
with open("models/xgb_metrics_v2.json", "w") as f:
    json.dump(metrics, f, indent=2)

# 5. Save the upgraded model
joblib.dump(best_pipeline, "models/sentiment_model_xgb_v2.pkl")
print("✅ spaCy+XGBoost model saved.")
