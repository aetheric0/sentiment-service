# benchmark.py
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

# ---- helpers -----------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """y_true in {0,1}; proba is P(class=1) shape (N,) or (N,2)."""
    if proba.ndim == 2:  # (N,2) -> take positive class column 1
        p1 = proba[:, 1]
    else:
        p1 = proba
    y_pred = (p1 >= 0.5).astype(int)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    # Optional metrics that need probabilities
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, p1))
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["brier"] = float(brier_score_loss(y_true, p1))
    except Exception:
        out["brier"] = float("nan")
    return out


def maybe_plot_curves(y_true: np.ndarray, p1_baseline: np.ndarray, p1_v2: np.ndarray) -> None:
    """Create ROC + reliability plots if matplotlib is available; otherwise skip."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import RocCurveDisplay

        # ROC
        RocCurveDisplay.from_predictions(y_true, p1_baseline, name="baseline")
        RocCurveDisplay.from_predictions(y_true, p1_v2, name="v2 (xgb)")
        plt.title("ROC curves")
        plt.savefig(REPORTS / "roc.png", dpi=160, bbox_inches="tight")
        plt.close()

        # Reliability (calibration) diagram
        frac_pos_b, mean_pred_b = calibration_curve(y_true, p1_baseline, n_bins=10)
        frac_pos_v, mean_pred_v = calibration_curve(y_true, p1_v2, n_bins=10)
        plt.plot(mean_pred_b, frac_pos_b, marker="o", label="baseline")
        plt.plot(mean_pred_v, frac_pos_v, marker="o", label="v2 (xgb)")
        plt.plot([0, 1], [0, 1], "--", label="perfect")
        plt.xlabel("Mean predicted probability (positive)")
        plt.ylabel("Fraction of positives")
        plt.title("Reliability diagram")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(REPORTS / "reliability.png", dpi=160, bbox_inches="tight")
        plt.close()
    except Exception:
        # Matplotlib not installed or headless issues; silently skip plotting.
        pass


def eval_pipeline(pipeline, X: pd.Series, y: pd.Series) -> Tuple[Dict[str, float], np.ndarray]:
    proba = pipeline.predict_proba(X)
    # normalize to positive class prob vector (N,)
    p1 = proba[:, 1] if proba.ndim == 2 else proba
    metrics = compute_metrics(y.to_numpy(), proba)
    return metrics, p1


# ---- load data ---------------------------------------------------------------
print("Downloading IMDB test split (first run will cache it)...")
imdb = load_dataset("imdb")
test_df = pd.DataFrame(imdb["test"])
X_test, y_test = test_df["text"], test_df["label"]

# ---- BASELINE: TF-IDF + LogisticRegression -----------------------------------
baseline_vec_path = MODELS / "tfidf_baseline.pkl"
baseline_model_path = MODELS / "logreg_baseline.pkl"
baseline_metrics_path = MODELS / "baseline_metrics.json"

if not (baseline_vec_path.exists() and baseline_model_path.exists()):
    print("No baseline artifacts found; training a quick baseline...")
    # Train on the IMDB train split for a fair test on IMDB test split.
    train_df = pd.DataFrame(imdb["train"])
    X_train, y_train = train_df["text"], train_df["label"]

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    clf = LogisticRegression(max_iter=2000, n_jobs=None)
    baseline = make_pipeline(vec, clf)
    baseline.fit(X_train, y_train)

    # Persist artifacts separately (to mirror your v1 style)
    joblib.dump(vec, baseline_vec_path)
    joblib.dump(clf, baseline_model_path)
    print("✅ Baseline artifacts saved.")

# Recreate baseline pipeline from artifacts
baseline_vec = joblib.load(baseline_vec_path)
baseline_clf = joblib.load(baseline_model_path)
baseline_pipeline = make_pipeline(baseline_vec, baseline_clf)

# Evaluate baseline
print("Evaluating baseline...")
baseline_metrics, p1_baseline = eval_pipeline(baseline_pipeline, X_test, y_test)
# Ensure the JSON exists for compatibility with older code
with open(baseline_metrics_path, "w") as f:
    json.dump(baseline_metrics, f, indent=2)
print(f"✅ Wrote {baseline_metrics_path.name}")

# ---- V2: your current vectorizer + model -------------------------------------
v2_vec_path = MODELS / "vectorizer_v2.pkl"
v2_model_path = MODELS / "sentiment_model_v2.pkl"
v2_metrics_path = MODELS / "xgb_metrics_v2.json"  # keep your filename

if not (v2_vec_path.exists() and v2_model_path.exists()):
    raise FileNotFoundError(
        "Expected v2 artifacts not found:\n"
        f"  - {v2_vec_path}\n  - {v2_model_path}\n"
        "Make sure your upgraded artifacts are in place."
    )

v2_vec = joblib.load(v2_vec_path)
v2_model = joblib.load(v2_model_path)
v2_pipeline = make_pipeline(v2_vec, v2_model)

print("Evaluating v2 (vectorizer_v2 + sentiment_model_v2)...")
v2_metrics, p1_v2 = eval_pipeline(v2_pipeline, X_test, y_test)
with open(v2_metrics_path, "w") as f:
    json.dump(v2_metrics, f, indent=2)
print(f"✅ Wrote {v2_metrics_path.name}")

# ---- Combined comparison table -----------------------------------------------
comparison = pd.DataFrame(
    [
        {"version": "v1_baseline", **baseline_metrics},
        {"version": "v2_spacy_xgb", **v2_metrics},
    ]
)
comparison.to_csv(REPORTS / "model_comparison.csv", index=False)
print(f"📄 Saved metrics: {REPORTS/'model_comparison.csv'}")

# ---- Side-by-side prediction sample ------------------------------------------
# Keep it small to avoid giant files; change n if you want more.
n = min(1000, len(test_df))
sample = test_df.sample(n=n, random_state=123).copy()
sample["baseline_p1"] = baseline_pipeline.predict_proba(sample["text"])[:, 1]
sample["v2_p1"] = v2_pipeline.predict_proba(sample["text"])[:, 1]
sample["baseline_pred"] = (sample["baseline_p1"] >= 0.5).astype(int)
sample["v2_pred"] = (sample["v2_p1"] >= 0.5).astype(int)
sample["agree"] = (sample["baseline_pred"] == sample["v2_pred"]).astype(int)
sample.rename(columns={"label": "gold"}, inplace=True)
sample.to_csv(REPORTS / "prediction_compare.csv", index=False)
print(f"📄 Saved per-row comparison: {REPORTS/'prediction_compare.csv'}")

# ---- Optional plots -----------------------------------------------------------
maybe_plot_curves(y_test.to_numpy(), p1_baseline, p1_v2)

print("✅ Benchmark complete.")
