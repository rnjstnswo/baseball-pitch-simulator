"""
Pydantic v2 request and response models for the Baseball Pitch Simulator API.

All schemas match the frozen /predict contract in docs/PROJECT_SPEC.md §6.
Do NOT change field names or types without updating the spec and bumping the
API version.

Classes:
    BatterQualityTier   — enum for wOBA tier labels
    PredictRequest      — POST /predict request body
    PitchOutcomeResult  — Model 1 prediction + probabilities
    BIPOutcomeResult    — Model 2 prediction + probabilities (nullable)
    ShapFactor          — single SHAP explanation entry
    UsageContext        — pitcher's historical usage for this count
    UpdatedState        — resulting count/base/out state after pitch
    PredictResponse     — POST /predict response body
    PitcherSummary      — item in GET /pitchers response
    ArsenalEntry        — item in GET /pitchers/{id}/arsenal response
    UsageEntry          — item in GET /pitchers/{id}/usage response
    HealthResponse      — GET /health response
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class BatterQualityTier(str, Enum):
    below_avg = "below_avg"
    average = "average"
    above_avg = "above_avg"
    elite = "elite"


class PredictRequest(BaseModel):
    pitcher_id: int = Field(..., description="MLB MLBAM pitcher ID")
    pitch_type: str = Field(..., description="Pitch type code (e.g. FF, SL, CH)")
    plate_x: float = Field(
        ..., ge=-2.0, le=2.0, description="Horizontal location (feet from center)"
    )
    plate_z: float = Field(
        ..., ge=0.5, le=5.0, description="Vertical location (feet from ground)"
    )
    batter_hand: str = Field(
        ..., pattern="^[LR]$", description="Batter handedness: L or R"
    )
    batter_quality_tier: BatterQualityTier
    balls: int = Field(..., ge=0, le=3)
    strikes: int = Field(..., ge=0, le=2)
    outs: int = Field(..., ge=0, le=2)
    inning: int = Field(..., ge=1, le=12)
    score_diff: int = Field(
        ..., ge=-10, le=10, description="Batter team score − pitcher team score"
    )
    on_1b: bool
    on_2b: bool
    on_3b: bool


class PitchOutcomeResult(BaseModel):
    prediction: str
    probabilities: dict[str, float]


class BIPOutcomeResult(BaseModel):
    prediction: str
    probabilities: dict[str, float]


class ShapFactor(BaseModel):
    feature: str
    value: float | str | bool
    shap_value: float
    direction: str


class UsageContext(BaseModel):
    pitch_usage_overall_pct: float
    pitch_usage_in_count_pct: float
    count: str
    sample_size: int


class UpdatedState(BaseModel):
    balls: int
    strikes: int
    outs: int
    on_1b: bool
    on_2b: bool
    on_3b: bool
    at_bat_result: str | None


class PredictResponse(BaseModel):
    pitch_outcome: PitchOutcomeResult
    bip_outcome: BIPOutcomeResult | None
    explanation: str
    top_shap_factors: list[ShapFactor]
    usage_context: UsageContext
    updated_state: UpdatedState


class PitcherSummary(BaseModel):
    pitcher_id: int
    full_name: str
    team: str
    p_throws: str


class ArsenalEntry(BaseModel):
    pitch_type: str
    pitch_name: str
    usage_pct: float
    avg_speed: float
    avg_spin: float
    avg_pfx_x: float
    avg_pfx_z: float
    sample_size: int


class UsageEntry(BaseModel):
    count: str
    pitch_type: str
    usage_pct: float
    sample_size: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
