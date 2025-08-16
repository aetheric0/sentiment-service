from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model + vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

app = FastAPI(title="Sentiment Classifier API")

class Review(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(review: Review):
    text_vec = vectorizer.transform([review.text])
    prediction = model.predict(text_vec)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    return {"Sentiment": sentiment}
