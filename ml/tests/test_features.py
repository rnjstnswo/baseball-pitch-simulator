"""Unit tests for ml/features.py — derived features, pipeline, and split."""

import numpy as np
import pandas as pd

from ml.features import (
    build_pipeline,
    engineer_features,
    get_feature_names,
    make_train_val_test_split,
    woba_to_ordinal,
)


def _sample_df(n=12):
    """Minimal Statcast-shaped frame covering every column engineer_features reads."""
    rng = np.random.default_rng(0)
    years = [2021, 2022, 2023, 2024]
    return pd.DataFrame(
        {
            "game_date": [f"{years[i % 4]}-06-01" for i in range(n)],
            "game_year": [years[i % 4] for i in range(n)],
            "batter": [100 + (i % 3) for i in range(n)],
            "pitcher": [200 + (i % 2) for i in range(n)],
            "pitch_type": ["FF", "SL"] * (n // 2),
            "pitch_name": ["4-Seam Fastball", "Slider"] * (n // 2),
            "release_speed": rng.uniform(85, 100, n),
            "release_spin_rate": rng.uniform(2000, 2600, n),
            "pfx_x": rng.uniform(-1, 1, n),
            "pfx_z": rng.uniform(-1, 1, n),
            "release_pos_x": rng.uniform(-2, 2, n),
            "release_pos_z": rng.uniform(5, 6, n),
            "release_extension": rng.uniform(5.5, 7, n),
            "spin_axis": rng.uniform(0, 360, n),
            "plate_x": rng.uniform(-1.5, 1.5, n),
            "plate_z": rng.uniform(1.5, 3.5, n),
            "zone": [1, 5, 9, 11] * (n // 4),
            "sz_top": rng.uniform(3.2, 3.6, n),
            "sz_bot": rng.uniform(1.4, 1.8, n),
            "balls": [i % 4 for i in range(n)],
            "strikes": [i % 3 for i in range(n)],
            "outs_when_up": [i % 3 for i in range(n)],
            "inning": [1 + (i % 11) for i in range(n)],
            "inning_topbot": ["Top", "Bot"] * (n // 2),
            "bat_score": rng.integers(0, 8, n),
            "fld_score": rng.integers(0, 8, n),
            "on_1b": [101 if i % 2 else pd.NA for i in range(n)],
            "on_2b": [pd.NA] * n,
            "on_3b": [103 if i % 3 == 0 else pd.NA for i in range(n)],
            "p_throws": ["R", "L"] * (n // 2),
            "stand": ["L", "R"] * (n // 2),
            "woba_value": rng.choice([0.0, 0.9, 1.25], n),
            "woba_denom": [1] * n,
        }
    )


def test_woba_to_ordinal_buckets():
    assert woba_to_ordinal(0.250) == 0
    assert woba_to_ordinal(0.310) == 1
    assert woba_to_ordinal(0.340) == 2
    assert woba_to_ordinal(0.370) == 3
    assert np.isnan(woba_to_ordinal(np.nan))


def test_engineer_features_derived_columns():
    out = engineer_features(_sample_df())
    assert out["runners_encoded"].between(0, 7).all()
    assert set(out["count_state"]).issubset({f"{b}-{s}" for b in range(4) for s in range(3)})
    assert out["inning_capped"].max() <= 9
    assert out["score_diff"].between(-10, 10).all()


def test_pipeline_produces_no_nans():
    df = engineer_features(_sample_df())
    pipe = build_pipeline()
    X = pipe.fit_transform(df)
    assert not np.isnan(X).any()
    assert X.shape[0] == len(df)
    assert len(get_feature_names(pipe)) == X.shape[1]


def test_split_no_year_overlap():
    df = _sample_df()
    train, val, test = make_train_val_test_split(df)
    assert set(pd.to_datetime(train["game_date"]).dt.year) <= {2021, 2022}
    assert set(pd.to_datetime(val["game_date"]).dt.year) == {2023}
    assert set(pd.to_datetime(test["game_date"]).dt.year) == {2024}
    assert len(train) + len(val) + len(test) == len(df)
