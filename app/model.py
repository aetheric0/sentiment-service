from pathlib import Path

import joblib
from sklearn.pipeline import make_pipeline

# Path to top-level models folder
BASE_DIR = Path(__file__).parent.parent  # parent of 'app/'
MODEL_PATH = BASE_DIR / "models" / "sentiment_model_v2.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer_v2.pkl"

# Load artifacts
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Combine into one scikit-learn Pipeline
model_pipeline = make_pipeline(vectorizer, model)
