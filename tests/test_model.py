import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model import model_pipeline

client = TestClient(app)

TEXT_CASES = [
    "",  # empty string
    "👍" * 1000,  # extreme length
    "El niño comió piña.",  # non-ASCII characters
]


def test_pipeline_predicts():
    text = "This is fantastic!"
    pred = model_pipeline.predict([text])
    assert pred in [0, 1]


@pytest.mark.parametrize("text", TEXT_CASES)
def test_predict_returns_full_probs(text):
    resp = client.post("/predict", json={"text": text})
    assert resp.status_code == 200
    body = resp.json()

    # Check top-level keys
    assert set(body.keys()) == {"prediction", "probabilities"}

    # Check that both classes appear
    probs = body["probabilities"]
    assert set(probs.keys()) == {"negative", "positive"}

    # Sum-to-1 sanity check
    assert abs(sum(probs.values()) - 1.0) < 1e-6


@pytest.mark.parametrize("text", TEXT_CASES)
def test_predict_edge_cases(text):
    resp = client.post("/predict", json={"text": text})
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert "probabilities" in body


def test_invalid_payload():
    # Missing 'text' key
    resp = client.post("/predict", json={"txt": "hello"})
    assert resp.status_code == 422  # Unprocessable Entity
