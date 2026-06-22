# Baseball Pitch Simulator

An interactive ML web app that predicts the outcome of any MLB pitch using real Statcast data.

Select a pitcher, choose a pitch from their real arsenal, click a location on the strike zone, and the app returns a calibrated probability distribution over pitch outcomes — plus a plain-English SHAP explanation.

**Status:** Phase 0 complete (spec + scaffold). Phase 1 (data pipeline) in progress.

---

## Demo

> _Coming in Phase 6._

---

## Architecture

Two chained classifiers trained on MLB Statcast data (2021–2024):

1. **Pitch Outcome Model** — predicts `ball | called_strike | swinging_strike | foul | in_play | hit_by_pitch`
2. **Ball-in-Play Model** — predicts `out | single | double | triple | home_run` (conditioned on contact)

See [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md) for the full specification.

```
pybaseball → Parquet → Feature Pipeline → XGBoost/LightGBM → Calibration → SHAP
                                                                              ↓
                                                           FastAPI (/predict) ↓
                                                              React + SVG Strike Zone
```

---

## Local Setup

### Prerequisites

- Python 3.10+
- Node.js 18+ (Phase 5+)
- Docker + docker-compose (Phase 6)

### 1. Clone & create virtual environment

```bash
git clone https://github.com/rnjstnswo/baseball-pitch-simulator.git
cd baseball-pitch-simulator
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install pre-commit hooks

```bash
pre-commit install
```

### 4. Pull Statcast data (Phase 1+)

```bash
python ml/ingest.py --start-year 2021 --end-year 2024
```

### 5. Train models (Phase 3+)

```bash
python ml/train_pitch_outcome.py --input data/processed/labeled.parquet
python ml/train_bip_outcome.py   --input data/processed/labeled_bip.parquet
```

### 6. Start the API (Phase 4+)

```bash
uvicorn api.main:app --reload
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 7. Start the frontend (Phase 5+)

```bash
cd frontend
npm install
npm run dev
# App available at http://localhost:5173
```

### 8. Run with Docker (Phase 6+)

```bash
docker-compose up --build
```

---

## Project Structure

```
baseball-pitch-simulator/
├── docs/PROJECT_SPEC.md     ← Source of truth for all design decisions
├── ml/                      ← Data pipeline, feature engineering, training
├── api/                     ← FastAPI backend
├── frontend/                ← React + TypeScript frontend (scaffolded in Phase 5)
├── data/                    ← Gitignored; raw and processed Parquet files
└── notebooks/               ← EDA only; not part of reproducible pipeline
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data | pybaseball, pandas, pyarrow (Parquet) |
| ML | scikit-learn, XGBoost / LightGBM, SHAP, joblib |
| Backend | FastAPI, Pydantic v2, uvicorn |
| Frontend | React, TypeScript, Vite, Tailwind CSS, Recharts, TanStack Query |
| Infra | Docker, GitHub Actions, Render/Fly.io, Vercel |

---

## License

MIT
