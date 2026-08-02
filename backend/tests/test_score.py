from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_score_invalid_txid():
    response = client.post("/score/", json={"txid": 999999999999})
    assert response.status_code == 404

def test_sample_endpoint_returns_list():
    response = client.get("/score/sample")
    assert response.status_code == 200
    assert isinstance(response.json(), list)