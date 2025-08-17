
## MODEL COMPARISON
📊 Metrics Explained
Metric	What it means	How to read it
accuracy	% of all predictions that were correct	Higher = fewer mistakes overall
precision	Out of all predicted positives, how many were actually positive	Higher = fewer false positives
recall	Out of all actual positives, how many were caught	Higher = fewer false negatives
f1	Harmonic mean of precision & recall (balances both)	Higher = good balance between precision & recall
roc_auc	Probability the model ranks a positive higher than a negative (area under ROC curve)	Closer to 1 = better separation of classes
brier	Mean squared error of predicted probabilities vs actual outcomes	Lower = better calibrated probabilities

🔎 Comparing Models

Baseline (v1_baseline)

Accuracy: 0.88868 (~88.9%)

ROC AUC: 0.956 → very strong separation ability

Brier: 0.0886 → good calibration

Spacy + XGBoost (v2_spacy_xgb)

Accuracy: 0.88256 (~88.3%) → slightly worse than baseline

ROC AUC: 0.953 → very close, only a bit lower

Brier: 0.0917 → slightly worse probability calibration


⚖️ Interpretation

Baseline logistic model (probably TF-IDF + Logistic Regression) is slightly better across most metrics.

Spacy + XGBoost didn’t beat the baseline here, though results are close.

But, XGBoost might generalize better on larger datasets or more nuanced language if tuned.



🚀 Next Steps

Try hyperparameter tuning for XGBoost (learning rate, depth, n_estimators).

Consider ensembling (averaging predictions from baseline & XGB).

Add error analysis → check which examples baseline got right but XGB missed (and vice versa).

## MODEL PREDICTION
