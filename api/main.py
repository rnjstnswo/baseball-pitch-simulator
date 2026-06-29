"""
FastAPI application entry point.

Registers all routers and configures CORS, lifespan (model loading), and
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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model artifacts and arsenal tables into app.state on startup
    raise NotImplementedError
    yield
    # Cleanup on shutdown (if needed)


app = FastAPI(
    title="Baseball Pitch Simulator API",
    description="Predicts MLB pitch outcomes using real Statcast data.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # extended in prod via env var
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    raise NotImplementedError


@app.get("/pitchers")
async def list_pitchers(search: str | None = None):
    raise NotImplementedError


@app.get("/pitchers/{pitcher_id}/arsenal")
async def get_arsenal(pitcher_id: int):
    raise NotImplementedError


@app.get("/pitchers/{pitcher_id}/usage")
async def get_usage(
    pitcher_id: int, pitch_type: str | None = None, count: str | None = None
):
    raise NotImplementedError


@app.post("/predict")
async def predict(
    request: object,
):  # replace `object` with PredictRequest from schemas.py
    raise NotImplementedError
