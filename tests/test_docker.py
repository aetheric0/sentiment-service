import requests


def test_docker_api_health():
    url = "http://localhost:8000/predict"
    resp = requests.post(url, json={"text": "Great product!"})
    assert resp.status_code == 200
    assert "prediction" in resp.json()
