"""
Chained model inference for the /predict endpoint.

Loads serialized artifacts, runs the two-model chain (pitch outcome →
optional BIP outcome), generates SHAP explanations, and attaches usage
context from the precomputed arsenal/usage tables.

The feature row is assembled manually from the request plus the precomputed
lookup tables — we do NOT reuse ml.features.engineer_features, whose groupby
aggregates are meaningless on a single row. Features not derivable from the
request come from the lookups (per-pitcher medians) or are left NaN for the
fitted median imputer (see the module constants for the two exceptions).

Public interface:
    load_artifacts(artifacts_dir) -> ModelBundle
    run_inference(request, bundle) -> PredictResponse
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from api.schemas import (
    BIPOutcomeResult,
    PitchOutcomeResult,
    PredictRequest,
    PredictResponse,
    ShapFactor,
    UpdatedState,
    UsageContext,
)
from ml.explain import (
    build_explainer,
    compute_shap_values,
    to_plain_english,
    top_factors,
)
from ml.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, WOBA_TIER_ORDINAL

ARTIFACTS_DIR = Path(__file__).parent.parent / "ml" / "artifacts"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# League-median strike-zone bounds (feet). Batter-driven, so no pitcher-level
# value exists; used as constants for the location-derived features. Computed
# once from the raw Statcast data (median of sz_top / sz_bot across 2021–2024).
LEAGUE_SZ_TOP = 3.39
LEAGUE_SZ_BOT = 1.589
LEAGUE_SZ_MID = (LEAGUE_SZ_TOP + LEAGUE_SZ_BOT) / 2
HALF_PLATE = 0.83  # half strike-zone width incl. ball radius (feet)


class UnknownPitcherError(Exception):
    """Raised when a pitcher_id is absent from the arsenal/pitcher tables."""


class InvalidPitchTypeError(Exception):
    """Raised when a pitch_type is not in the pitcher's arsenal."""


@dataclass
class ModelBundle:
    """Container for all loaded artifacts needed at inference time."""

    pitch_outcome_model: object
    bip_model: object
    preprocessor: object
    pitch_outcome_explainer: object
    arsenal_df: pd.DataFrame
    usage_df: pd.DataFrame
    pitcher_woba_df: pd.DataFrame
    pitchers_df: pd.DataFrame
    feature_names: list[str]


def load_artifacts(artifacts_dir: Path = ARTIFACTS_DIR) -> ModelBundle:
    """Load all serialized model artifacts and lookup tables.

    Builds the Model 1 SHAP explainer with no background dataset (the
    tree_path_dependent path, validated < 200ms). Passing an interventional
    background sample would exceed the latency budget by 3–8×.

    Raises:
        FileNotFoundError: If any required artifact is missing.
    """
    import joblib

    pitch_model = joblib.load(artifacts_dir / "pitch_outcome_model.joblib")
    bip_model = joblib.load(artifacts_dir / "bip_model.joblib")
    preprocessor = joblib.load(artifacts_dir / "preprocessor.joblib")

    return ModelBundle(
        pitch_outcome_model=pitch_model,
        bip_model=bip_model,
        preprocessor=preprocessor,
        pitch_outcome_explainer=build_explainer(pitch_model),
        arsenal_df=pd.read_parquet(PROCESSED_DIR / "arsenal.parquet"),
        usage_df=pd.read_parquet(PROCESSED_DIR / "usage.parquet"),
        pitcher_woba_df=pd.read_parquet(PROCESSED_DIR / "pitcher_woba.parquet"),
        pitchers_df=pd.read_parquet(PROCESSED_DIR / "pitchers.parquet"),
        feature_names=preprocessor.get_feature_names_out().tolist(),
    )


def _statcast_zone(plate_x: float, plate_z: float) -> str:
    """Approximate the Statcast zone code (1–9 inside, 11–14 outside).

    Uses league-median zone bounds since the batter (and thus the true zone)
    is unknown at request time. Orientation is from the catcher's view.
    """
    in_x = abs(plate_x) <= HALF_PLATE
    in_z = LEAGUE_SZ_BOT <= plate_z <= LEAGUE_SZ_TOP
    if in_x and in_z:
        col = 0 if plate_x < -HALF_PLATE / 3 else (2 if plate_x > HALF_PLATE / 3 else 1)
        third = (LEAGUE_SZ_TOP - LEAGUE_SZ_BOT) / 3
        row = (
            0
            if plate_z >= LEAGUE_SZ_TOP - third
            else (2 if plate_z <= LEAGUE_SZ_BOT + third else 1)
        )
        return str(1 + row * 3 + col)
    top = plate_z >= LEAGUE_SZ_MID
    left = plate_x < 0
    return {(True, True): "11", (True, False): "12", (False, True): "13"}.get(
        (top, left), "14"
    )


def _build_feature_row(request: PredictRequest, bundle: ModelBundle) -> pd.DataFrame:
    """Convert a PredictRequest into a single-row feature DataFrame.

    Pulls pitch-characteristic and usage features from the precomputed lookup
    tables (using the pitcher's latest season) and derives the rest from the
    request. Raises UnknownPitcherError / InvalidPitchTypeError on bad lookups.
    """
    pid = request.pitcher_id
    arsenal = bundle.arsenal_df[bundle.arsenal_df["pitcher_id"] == pid]
    if arsenal.empty:
        raise UnknownPitcherError(f"pitcher_id {pid} not found")

    season = int(arsenal["season"].max())
    arsenal = arsenal[arsenal["season"] == season]
    pitch = arsenal[arsenal["pitch_type"] == request.pitch_type]
    if pitch.empty:
        raise InvalidPitchTypeError(
            f"pitch_type {request.pitch_type!r} not in pitcher's arsenal"
        )
    p = pitch.iloc[0]

    count_state = f"{request.balls}-{request.strikes}"
    usage = bundle.usage_df[
        (bundle.usage_df["pitcher_id"] == pid)
        & (bundle.usage_df["season"] == season)
        & (bundle.usage_df["count_state"] == count_state)
        & (bundle.usage_df["pitch_type"] == request.pitch_type)
    ]
    usage_in_count = float(usage["usage_pct"].iloc[0]) if not usage.empty else 0.0

    woba = bundle.pitcher_woba_df[
        (bundle.pitcher_woba_df["pitcher_id"] == pid)
        & (bundle.pitcher_woba_df["season"] == season)
    ]
    pitcher_woba_tier = float(woba["woba_tier"].iloc[0]) if not woba.empty else np.nan

    meta = bundle.pitchers_df[bundle.pitchers_df["pitcher_id"] == pid]
    p_throws = meta["p_throws"].iloc[0] if not meta.empty else "R"

    on_1b, on_2b, on_3b = int(request.on_1b), int(request.on_2b), int(request.on_3b)
    row = {
        "release_speed": p["avg_speed"],
        "release_spin_rate": p["avg_spin"],
        "pfx_x": p["avg_pfx_x"],
        "pfx_z": p["avg_pfx_z"],
        "release_pos_x": p["median_release_pos_x"],
        "release_pos_z": p["median_release_pos_z"],
        "release_extension": p["median_release_extension"],
        "spin_axis": p["median_spin_axis"],
        "plate_x": request.plate_x,
        "plate_z": request.plate_z,
        "in_zone": int(
            abs(request.plate_x) <= HALF_PLATE
            and LEAGUE_SZ_BOT <= request.plate_z <= LEAGUE_SZ_TOP
        ),
        "dist_from_center": float(
            np.hypot(request.plate_x, request.plate_z - LEAGUE_SZ_MID)
        ),
        "sz_top": LEAGUE_SZ_TOP,
        "sz_bot": LEAGUE_SZ_BOT,
        "balls": request.balls,
        "strikes": request.strikes,
        "outs_when_up": request.outs,
        "inning_capped": min(request.inning, 9),
        "inning_top": np.nan,  # unknown from request → median-imputed
        "score_diff": max(-10, min(10, request.score_diff)),
        "on_1b": on_1b,
        "on_2b": on_2b,
        "on_3b": on_3b,
        "runners_encoded": 4 * on_3b + 2 * on_2b + on_1b,
        "p_throws_enc": int(p_throws == "R"),
        "stand_enc": int(request.batter_hand == "R"),
        "batter_woba_tier": WOBA_TIER_ORDINAL[request.batter_quality_tier.value],
        "pitcher_woba_tier": pitcher_woba_tier,
        "pitch_usage_pct": p["usage_pct"],
        "pitch_usage_in_count_pct": usage_in_count,
        "avg_speed_for_pitch": p["avg_speed"],
        "avg_spin_for_pitch": p["avg_spin"],
        "pitch_type": request.pitch_type,
        "zone_str": _statcast_zone(request.plate_x, request.plate_z),
        "count_state": count_state,
    }
    return pd.DataFrame([row], columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def _probabilities(model: object, X) -> dict[str, float]:
    """Return a {class_label: probability} dict for a single-row transform."""
    proba = model.predict_proba(X)[0]
    return {str(c): float(pr) for c, pr in zip(model.classes_, proba, strict=True)}


def _map_shap_factors(factors: list[dict], feature_row: pd.DataFrame) -> list[dict]:
    """Map transformed SHAP feature names back to source column + raw value.

    ColumnTransformer names look like "num__strikes" or "cat__pitch_type_FF".
    We strip the prefix, resolve the source column, and attach its raw value
    from the pre-transform feature row.
    """
    mapped = []
    for f in factors:
        name = f["feature"]
        source = name
        if name.startswith("num__"):
            source = name[len("num__") :]
        elif name.startswith("cat__"):
            for col in CATEGORICAL_FEATURES:
                if name.startswith(f"cat__{col}_"):
                    source = col
                    break
        raw = feature_row[source].iloc[0] if source in feature_row else None
        if isinstance(raw, (np.generic,)):
            raw = raw.item()
        mapped.append(
            {
                "feature": source,
                "value": raw,
                "shap_value": f["shap_value"],
                "direction": f["direction"],
            }
        )
    return mapped


def run_inference(request: PredictRequest, bundle: ModelBundle) -> PredictResponse:
    """Execute the full prediction chain for a single request."""
    feature_row = _build_feature_row(request, bundle)
    X = bundle.preprocessor.transform(feature_row)

    pitch_probs = _probabilities(bundle.pitch_outcome_model, X)
    pitch_pred = max(pitch_probs, key=pitch_probs.get)

    bip_outcome = None
    bip_pred = None
    if pitch_pred == "in_play":
        bip_probs = _probabilities(bundle.bip_model, X)
        bip_pred = max(bip_probs, key=bip_probs.get)
        bip_outcome = BIPOutcomeResult(prediction=bip_pred, probabilities=bip_probs)

    class_idx = list(bundle.pitch_outcome_model.classes_).index(pitch_pred)
    shap_values = compute_shap_values(bundle.pitch_outcome_explainer, X)
    raw_factors = top_factors(shap_values, bundle.feature_names, class_idx, n=4)
    factors = _map_shap_factors(raw_factors, feature_row)

    usage_ctx = _usage_context(request, bundle, feature_row)

    return PredictResponse(
        pitch_outcome=PitchOutcomeResult(
            prediction=pitch_pred, probabilities=pitch_probs
        ),
        bip_outcome=bip_outcome,
        explanation=to_plain_english(pitch_pred, factors),
        top_shap_factors=[ShapFactor(**f) for f in factors],
        usage_context=usage_ctx,
        updated_state=_compute_updated_state(request, pitch_pred, bip_pred),
    )


def _usage_context(
    request: PredictRequest, bundle: ModelBundle, feature_row: pd.DataFrame
) -> UsageContext:
    """Assemble the usage_context block from the arsenal and usage tables."""
    pid = request.pitcher_id
    count_state = f"{request.balls}-{request.strikes}"
    season = int(
        bundle.arsenal_df[bundle.arsenal_df["pitcher_id"] == pid]["season"].max()
    )
    usage = bundle.usage_df[
        (bundle.usage_df["pitcher_id"] == pid)
        & (bundle.usage_df["season"] == season)
        & (bundle.usage_df["count_state"] == count_state)
        & (bundle.usage_df["pitch_type"] == request.pitch_type)
    ]
    return UsageContext(
        pitch_usage_overall_pct=float(feature_row["pitch_usage_pct"].iloc[0]),
        pitch_usage_in_count_pct=float(feature_row["pitch_usage_in_count_pct"].iloc[0]),
        count=count_state,
        sample_size=int(usage["sample_size"].iloc[0]) if not usage.empty else 0,
    )


def _compute_updated_state(
    request: PredictRequest, pitch_prediction: str, bip_prediction: str | None
) -> UpdatedState:
    """Determine the resulting count / base / out state after the pitch.

    Real baseball terminal logic (strike 3 = strikeout, ball 4 = walk). Base
    advancement is simplified — forced runners advance on walk/HBP; on a hit,
    all runners and the batter advance by the hit's base count. Precise force
    and scoring rules are out of MVP scope (spec §10).
    """
    balls, strikes, outs = request.balls, request.strikes, request.outs
    bases = [request.on_1b, request.on_2b, request.on_3b]
    result: str | None = None

    def walk_or_hbp(bases: list[bool]) -> list[bool]:
        # Batter to 1b; push only forced runners ahead of him.
        on1, on2, on3 = bases
        if on1 and on2:
            on3 = True
        if on1:
            on2 = True
        return [True, on2, on3]

    if pitch_prediction == "ball":
        balls += 1
        if balls == 4:
            result, balls, strikes = "walk", 0, 0
            bases = walk_or_hbp(bases)
    elif pitch_prediction in ("called_strike", "swinging_strike"):
        strikes += 1
        if strikes == 3:
            result, balls, strikes, outs = "strikeout", 0, 0, outs + 1
    elif pitch_prediction == "foul":
        strikes = min(strikes + 1, 2)
    elif pitch_prediction == "hit_by_pitch":
        result, balls, strikes = "hit_by_pitch", 0, 0
        bases = walk_or_hbp(bases)
    elif pitch_prediction == "in_play":
        balls, strikes = 0, 0
        advance = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
        if bip_prediction == "out":
            result, outs = "out", outs + 1
        elif bip_prediction in advance:
            result = bip_prediction
            n = advance[bip_prediction]
            new_bases = [False, False, False]
            for i, occupied in enumerate(bases):  # existing runners advance by n
                if occupied and i + n <= 2:
                    new_bases[i + n] = True
            if n <= 3:  # batter ends on 1b/2b/3b; home_run (n=4) scores
                new_bases[n - 1] = True
            bases = new_bases

    return UpdatedState(
        balls=balls,
        strikes=strikes,
        outs=outs,
        on_1b=bases[0],
        on_2b=bases[1],
        on_3b=bases[2],
        at_bat_result=result,
    )
