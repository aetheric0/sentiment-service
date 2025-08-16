import joblib
from pathlib import Path

# Path to top-level models folder
BASE_DIR = Path(__file__).parent.parent  # parent of 'app/'
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer.pkl"

# Load artifacts
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
