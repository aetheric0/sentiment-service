#!/usr/bin/env python3
from typing import Dict

from fastapi import FastAPI, HTTPException

from app.schemas import PredictionResponse, TextRequest

from .model import model_pipeline

# Map numberic labels -> human-readable class names
CLASS_NAMES = {0: "negative", 1: "positive"}

app = FastAPI(
    title="Sentiment Classifier API",
    description=(
        "Predict sentiment using a spaCy-preprocessed "
        "TF-IDF + XGBoost pipeline"
    ),
    version="2.0.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model_pipeline is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: TextRequest):
    try:
        # Turn the raw text into a list for scikit-learn
        texts = [request.text]
        # 1) Predict class label
        pred_label = model_pipeline.predict(texts)[0]
        prediction = CLASS_NAMES[pred_label]

        # 2) Predict class probabilities
        if hasattr(model_pipeline, "predict_proba"):
            proba = model_pipeline.predict_proba(texts)[0]
            # Build a dict: {"negative": 0.0136, "positive": 0.9864}
            probabilities: Dict[str, float] = {
                CLASS_NAMES[i]: float(prob) for i, prob in enumerate(proba)
            }
        else:
            probabilities = {}

        # 3. Return exactly what PredictionResponse expects
        return {"prediction": prediction, "probabilities": probabilities}
    except Exception as e:
        # Bubble up any inference errors as HTTP 500
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Allows: python3 app/main.py
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
