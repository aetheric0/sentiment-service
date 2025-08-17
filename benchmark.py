import json

import pandas as pd

base = json.load(open("models/baseline_metrics.json"))
v2 = json.load(open("models/xgb_metrics_v2.json"))

df = pd.DataFrame(
    [{"version": "v1_baseline", **base}, {"version": "v2_spacy_xgb", **v2}]
)
df.to_csv("models/model_comparison.csv", index=False)
print(df)
