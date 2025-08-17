```markdown
# 📊 Model and prediction reports

A compact summary of model performance, key metrics, plots, and how to reproduce results.

---

## 🧩 Metrics explained

| Metric   | What it means                                                   | How to read it                     |
|----------|------------------------------------------------------------------|------------------------------------|
| accuracy | Percent of all predictions that were correct                     | Higher = fewer mistakes overall    |
| precision| Out of predicted positives, how many were actually positive      | Higher = fewer false positives     |
| recall   | Out of actual positives, how many were correctly identified      | Higher = fewer false negatives     |
| f1       | Harmonic mean of precision and recall                            | Higher = balanced precision/recall |
| roc_auc  | Area under ROC curve (ranking separation between classes)        | Closer to 1 = better separation    |
| brier    | Mean squared error of predicted probabilities vs. outcomes       | Lower = better calibration         |

---

## 🥊 Model comparison

| Model              | Accuracy  | ROC AUC | Brier  |
|--------------------|-----------|---------|--------|
| Baseline (TF–IDF + Logistic Regression) | 0.88868  | 0.956   | 0.0886 |
| spaCy + XGBoost    | 0.88256   | 0.953   | 0.0917 |

> Interpretation:
> - Baseline logistic model slightly leads across accuracy, ROC AUC, and Brier.
> - spaCy + XGBoost is close; with tuning or more data it may catch up or surpass.
> - Calibration (Brier) favors the baseline; expect steadier probability quality there.

---

## 🖼️ Plots

- **ROC curve:** `reports/roc.png`
- **Reliability diagram:** `reports/reliability.png`

> Tip: Reliability helps decide whether to use probability thresholds confidently (e.g., abstain below 0.6).

---

## 🔁 Reproduce results

- **Data:** Ensure the review dataset is available locally (see project root README for setup).
- **Notebooks:** Run `notebooks/sentiment_xgb.py` to train/evaluate the spaCy + XGB pipeline.
- **CLI/script:** Use `benchmark.py` to generate comparison metrics (update paths if needed).
- **Artifacts:** Serialized models live in `models/` and metrics in `models/*.json`.

---

## 🔎 Error analysis ideas

- **Confusion slices:** Inspect false positives vs. false negatives by review length and sentiment intensity.
- **Token impact:** Check top TF–IDF features driving positive/negative predictions for the baseline.
- **Compare misses:** Identify examples baseline gets right but XGB misses (and the reverse).

---

## 🚀 Next steps

- **XGBoost tuning:** Learning rate, max depth, n_estimators, subsampling.
- **Ensembling:** Average calibrated probabilities from baseline and XGB.
- **Thresholding:** Choose operating points for high-precision or high-recall use cases.
- **Calibration:** Platt scaling/Isotonic on a holdout set if deployment needs sharp probabilities.

---

## 📁 Prediction artifacts

- **Comparison CSV:** `reports/prediction_compare.csv` summarizes side-by-side predictions and probabilities.
- **How to use:** Filter rows where models disagree to prioritize qualitative review and labeling.
