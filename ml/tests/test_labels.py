"""Unit tests for ml/labels.py — label mapping coverage and taxonomy."""

import pandas as pd
import pytest

from ml.labels import (
    BIP_OUTCOME_CLASSES,
    PITCH_OUTCOME_CLASSES,
    PITCH_OUTCOME_MAP,
    add_labels,
    map_bip_outcome,
    map_pitch_outcome,
    validate_coverage,
)


def test_pitch_outcome_map_targets_are_valid_classes():
    assert set(PITCH_OUTCOME_MAP.values()) <= set(PITCH_OUTCOME_CLASSES)


def test_map_pitch_outcome_covers_all_known_descriptions():
    df = pd.DataFrame({"description": list(PITCH_OUTCOME_MAP)})
    out = map_pitch_outcome(df)
    assert out.notna().all()
    assert set(out) <= set(PITCH_OUTCOME_CLASSES)


def test_validate_coverage_passes_on_known_descriptions():
    df = pd.DataFrame({"description": list(PITCH_OUTCOME_MAP)})
    validate_coverage(df)  # should not raise


def test_validate_coverage_raises_on_unknown_description():
    df = pd.DataFrame({"description": ["ball", "some_new_description"]})
    with pytest.raises(ValueError, match="some_new_description"):
        validate_coverage(df)


def test_map_bip_outcome_field_error_is_out():
    df = pd.DataFrame({"events": ["single", "home_run", "field_error", "force_out"]})
    out = map_bip_outcome(df)
    assert out.tolist() == ["single", "home_run", "out", "out"]
    assert set(out) <= set(BIP_OUTCOME_CLASSES)


def test_add_labels_bip_only_for_in_play():
    df = pd.DataFrame(
        {
            "description": ["ball", "hit_into_play", "swinging_strike"],
            "events": [None, "single", None],
        }
    )
    out = add_labels(df)
    assert out["pitch_outcome"].tolist() == ["ball", "in_play", "swinging_strike"]
    assert out.loc[out["pitch_outcome"] == "in_play", "bip_outcome"].tolist() == ["single"]
    assert out.loc[out["pitch_outcome"] != "in_play", "bip_outcome"].isna().all()
