from typing import Dict

from pydantic import BaseModel


class TextRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    prediction: str
    probabilities: Dict[str, float]
