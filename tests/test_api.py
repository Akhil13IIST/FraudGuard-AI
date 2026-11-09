import json

import pytest
from fastapi.testclient import TestClient

from api import database
from api.main import app, feature_columns


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "api_test.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)

    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["feature_count"] == len(feature_columns)


def test_home_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_model_metrics_endpoint(client):
    response = client.get("/model/metrics")

    assert response.status_code == 200
    assert "average_precision" in response.json()


def test_samples_endpoint(client):
    response = client.get("/samples?limit=5")

    assert response.status_code == 200
    assert len(response.json()["samples"]) <= 5


def test_invalid_sample_limit(client):
    response = client.get("/samples?limit=0")

    assert response.status_code == 422


def test_missing_features_rejected(client):
    response = client.post(
        "/predict",
        json={
            "tx_id": "test_tx",
            "time_step": 10,
            "features": {}
        }
    )

    assert response.status_code == 400
    assert "missing_features" in response.json()["detail"]


def test_extra_features_rejected(client):
    features = {name: 0.0 for name in feature_columns}
    features["unexpected_feature"] = 1.0

    response = client.post(
        "/predict",
        json={
            "tx_id": "test_tx",
            "time_step": 10,
            "features": features
        }
    )

    assert response.status_code == 400
    assert "extra_features" in response.json()["detail"]



def test_non_finite_feature_rejected(client):
    features = {name: 0.0 for name in feature_columns}
    features[feature_columns[0]] = float("nan")

    payload = {
        "tx_id": "test_tx",
        "time_step": 10,
        "features": features,
    }

    response = client.post(
        "/predict",
        content=json.dumps(payload, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    assert "finite number" in response.json()["detail"]



def test_invalid_time_step_rejected(client):
    response = client.post(
        "/predict",
        json={
            "tx_id": "test_tx",
            "time_step": 50,
            "features": {}
        }
    )

    assert response.status_code == 422
