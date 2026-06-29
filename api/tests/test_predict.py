"""
Tests for the /predict endpoint and inference chain.

Covers:
    - Valid request returns HTTP 200 with correct response shape
    - Invalid pitch type returns HTTP 422
    - Out-of-range plate location returns HTTP 422
    - Missing required field returns HTTP 422
    - BIP outcome is non-null only when pitch prediction is in_play
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

VALID_REQUEST = {
    "pitcher_id": 543037,
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


@pytest.mark.skip(reason="Not implemented — Phase 4")
def test_valid_predict_returns_200():
    response = client.post("/predict", json=VALID_REQUEST)
    assert response.status_code == 200
    data = response.json()
    assert "pitch_outcome" in data
    assert "probabilities" in data["pitch_outcome"]
    assert sum(data["pitch_outcome"]["probabilities"].values()) == pytest.approx(
        1.0, abs=1e-3
    )


@pytest.mark.skip(reason="Not implemented — Phase 4")
def test_invalid_pitch_type_returns_422():
    payload = {**VALID_REQUEST, "pitch_type": "NOTREAL"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.skip(reason="Not implemented — Phase 4")
def test_out_of_range_plate_x_returns_422():
    payload = {**VALID_REQUEST, "plate_x": 99.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.skip(reason="Not implemented — Phase 4")
def test_missing_required_field_returns_422():
    payload = {k: v for k, v in VALID_REQUEST.items() if k != "pitcher_id"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.skip(reason="Not implemented — Phase 4")
def test_bip_outcome_null_when_not_in_play():
    """If pitch outcome prediction is not in_play, bip_outcome must be null."""
    response = client.post("/predict", json=VALID_REQUEST)
    data = response.json()
    if data["pitch_outcome"]["prediction"] != "in_play":
        assert data["bip_outcome"] is None
