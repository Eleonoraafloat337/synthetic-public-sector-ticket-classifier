from fastapi.testclient import TestClient
import pytest

from inference.app import app, classifier
from ticket_classifier.labels import LABELS


def test_classify_with_scores() -> None:
    classifier.load()
    client = TestClient(app)
    response = client.post("/classify", json={"text": "I cannot login to the staff portal", "include_scores": True})
    assert response.status_code == 200
    body = response.json()
    assert {"label", "confidence", "model_version", "latency_ms", "scores"} <= set(body)
    assert set(body["scores"]) == set(LABELS)
    assert sum(body["scores"].values()) == pytest.approx(1.0, abs=0.001)


def test_validation_edges() -> None:
    client = TestClient(app)
    assert client.post("/classify", json={"text": ""}).status_code == 422
    assert client.post("/classify", json={"text": "x" * 2001}).status_code == 422
    assert client.post("/classify", content=b'{"text":"\\xff"}', headers={"content-type": "application/json"}).status_code in {400, 422}


def test_batch_health_metrics_and_drift() -> None:
    client = TestClient(app)
    response = client.post("/classify/batch", json={"texts": ["vpn outage", "software license request"], "include_scores": False})
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert client.get("/health").json()["status"] == "ok"
    assert "classify_requests_total" in client.get("/metrics").text
    assert "js_divergence" in client.get("/metrics/drift").json()
