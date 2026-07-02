"""
FastAPI application entry point.

Registers all endpoints and configures CORS, lifespan (model loading), and
OpenAPI metadata. Start with: uvicorn api.main:app --reload

Endpoints:
    GET  /health
    GET  /pitchers
    GET  /pitchers/{pitcher_id}/arsenal
    GET  /pitchers/{pitcher_id}/usage
    POST /predict
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.predict import (
    InvalidPitchTypeError,
    UnknownPitcherError,
    load_artifacts,
    run_inference,
)
from api.schemas import (
    ArsenalEntry,
    ArsenalResponse,
    HealthResponse,
    PitchersResponse,
    PitcherSummary,
    PredictRequest,
    PredictResponse,
    UsageEntry,
    UsageResponse,
)

VERSION = "0.2.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = load_artifacts()
    yield


app = FastAPI(
    title="Baseball Pitch Simulator API",
    description="Predicts MLB pitch outcomes using real Statcast data.",
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # extended in prod via env var
    allow_methods=["*"],
    allow_headers=["*"],
)


def _latest_season(bundle, pitcher_id: int) -> int:
    rows = bundle.arsenal_df[bundle.arsenal_df["pitcher_id"] == pitcher_id]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"pitcher {pitcher_id} not found")
    return int(rows["season"].max())


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model_loaded=True, version=VERSION)


@app.get("/pitchers", response_model=PitchersResponse)
async def list_pitchers(search: str | None = None):
    df = app.state.bundle.pitchers_df
    if search:
        df = df[df["full_name"].str.contains(search, case=False, na=False)]
    pitchers = [
        PitcherSummary(
            pitcher_id=int(r.pitcher_id),
            full_name=r.full_name,
            team=r.team,
            p_throws=r.p_throws,
        )
        for r in df.itertuples()
    ]
    return PitchersResponse(pitchers=pitchers)


@app.get("/pitchers/{pitcher_id}/arsenal", response_model=ArsenalResponse)
async def get_arsenal(pitcher_id: int):
    bundle = app.state.bundle
    season = _latest_season(bundle, pitcher_id)
    rows = bundle.arsenal_df[
        (bundle.arsenal_df["pitcher_id"] == pitcher_id)
        & (bundle.arsenal_df["season"] == season)
    ]
    meta = bundle.pitchers_df[bundle.pitchers_df["pitcher_id"] == pitcher_id]
    full_name = meta["full_name"].iloc[0] if not meta.empty else ""
    arsenal = [
        ArsenalEntry(
            pitch_type=r.pitch_type,
            pitch_name=r.pitch_name,
            usage_pct=float(r.usage_pct),
            avg_speed=float(r.avg_speed),
            avg_spin=float(r.avg_spin),
            avg_pfx_x=float(r.avg_pfx_x),
            avg_pfx_z=float(r.avg_pfx_z),
            sample_size=int(r.sample_size),
        )
        for r in rows.sort_values("usage_pct", ascending=False).itertuples()
    ]
    return ArsenalResponse(
        pitcher_id=pitcher_id, full_name=full_name, season=season, arsenal=arsenal
    )


@app.get("/pitchers/{pitcher_id}/usage", response_model=UsageResponse)
async def get_usage(
    pitcher_id: int, pitch_type: str | None = None, count: str | None = None
):
    bundle = app.state.bundle
    season = _latest_season(bundle, pitcher_id)
    rows = bundle.usage_df[
        (bundle.usage_df["pitcher_id"] == pitcher_id)
        & (bundle.usage_df["season"] == season)
    ]
    if pitch_type:
        rows = rows[rows["pitch_type"] == pitch_type]
    if count:
        rows = rows[rows["count_state"] == count]
    usage_by_count = [
        UsageEntry(
            count=r.count_state,
            pitch_type=r.pitch_type,
            usage_pct=float(r.usage_pct),
            sample_size=int(r.sample_size),
        )
        for r in rows.itertuples()
    ]
    return UsageResponse(pitcher_id=pitcher_id, usage_by_count=usage_by_count)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        return run_inference(request, app.state.bundle)
    except UnknownPitcherError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except InvalidPitchTypeError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
