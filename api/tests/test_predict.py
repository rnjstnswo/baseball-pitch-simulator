"""
Tests for the /predict endpoint and the other API endpoints.

Covers:
    - Valid request returns HTTP 200 with correct response shape
    - Invalid pitch type returns HTTP 422
    - Out-of-range plate location returns HTTP 422
    - Missing required field returns HTTP 422
    - BIP outcome is non-null only when pitch prediction is in_play
    - /health, /pitchers, /arsenal, /usage return expected schemas
    - Unknown pitcher returns HTTP 404
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app

# Gerrit Cole — present in the regenerated lookup tables (latest season 2024).
PITCHER_ID = 543037

VALID_REQUEST = {
    "pitcher_id": PITCHER_ID,
    "pitch_type": "FF",
    "plate_x": -0.5,
    "plate_z": 2.8,
    "batter_hand": "R",
    "batter_quality_tier": "average",
    "balls": 1,
    "strikes": 2,
    "outs": 1,
    "inning": 6,
    "score_diff": -1,
    "on_1b": True,
    "on_2b": False,
    "on_3b": False,
}


@pytest.fixture(scope="module")
def client():
    # `with` triggers the lifespan handler, loading model artifacts into state.
    with TestClient(app) as c:
        yield c


def test_valid_predict_returns_200(client):
    response = client.post("/predict", json=VALID_REQUEST)
    assert response.status_code == 200
    data = response.json()
    assert "pitch_outcome" in data
    assert "probabilities" in data["pitch_outcome"]
    assert sum(data["pitch_outcome"]["probabilities"].values()) == pytest.approx(
        1.0, abs=1e-3
    )
    assert len(data["top_shap_factors"]) == 4
    assert data["updated_state"]["balls"] is not None


def test_invalid_pitch_type_returns_422(client):
    payload = {**VALID_REQUEST, "pitch_type": "NOTREAL"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_out_of_range_plate_x_returns_422(client):
    payload = {**VALID_REQUEST, "plate_x": 99.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_missing_required_field_returns_422(client):
    payload = {k: v for k, v in VALID_REQUEST.items() if k != "pitcher_id"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_bip_outcome_null_when_not_in_play(client):
    """If pitch outcome prediction is not in_play, bip_outcome must be null."""
    data = client.post("/predict", json=VALID_REQUEST).json()
    if data["pitch_outcome"]["prediction"] != "in_play":
        assert data["bip_outcome"] is None
    else:
        assert data["bip_outcome"] is not None
        assert sum(data["bip_outcome"]["probabilities"].values()) == pytest.approx(
            1.0, abs=1e-3
        )


def test_unknown_pitcher_returns_404(client):
    payload = {**VALID_REQUEST, "pitcher_id": 1}
    response = client.post("/predict", json=payload)
    assert response.status_code == 404


def test_health(client):
    data = client.get("/health").json()
    assert data == {"status": "ok", "model_loaded": True, "version": "0.2.0"}


def test_pitchers_search(client):
    data = client.get("/pitchers", params={"search": "Cole"}).json()
    assert "pitchers" in data
    ids = {p["pitcher_id"] for p in data["pitchers"]}
    assert PITCHER_ID in ids


def test_arsenal_shape(client):
    data = client.get(f"/pitchers/{PITCHER_ID}/arsenal").json()
    assert data["pitcher_id"] == PITCHER_ID
    assert len(data["arsenal"]) > 0
    entry = data["arsenal"][0]
    assert {"pitch_type", "pitch_name", "usage_pct", "avg_speed"} <= entry.keys()


def test_usage_filter(client):
    data = client.get(f"/pitchers/{PITCHER_ID}/usage", params={"count": "0-0"}).json()
    assert data["pitcher_id"] == PITCHER_ID
    assert all(u["count"] == "0-0" for u in data["usage_by_count"])


def test_arsenal_unknown_pitcher_404(client):
    assert client.get("/pitchers/1/arsenal").status_code == 404
