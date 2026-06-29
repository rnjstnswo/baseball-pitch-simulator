# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## guideline
1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

State your assumptions explicitly. If uncertain, ask.
If multiple interpretations exist, present them - don't pick silently.
If a simpler approach exists, say so. Push back when warranted.
If something is unclear, stop. Name what's confusing. Ask.
2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

No features beyond what was asked.
No abstractions for single-use code.
No "flexibility" or "configurability" that wasn't requested.
No error handling for impossible scenarios.
If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, comments, or formatting.
Don't refactor things that aren't broken.
Match existing style, even if you'd do it differently.
If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused.
Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.

4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

"Add validation" → "Write tests for invalid inputs, then make them pass"
"Fix the bug" → "Write a test that reproduces it, then make it pass"
"Refactor X" → "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
pre-commit install

# Lint /e format
ruff check . --fix
black .
isort .
pre-commit run --all-files   # runs black + ruff + isort + file checks

# Run tests
pytest                        # all tests in api/tests/
pytest api/tests/test_predict.py::test_valid_predict_returns_200   # single test

# ML pipeline (Phase 1+)
python ml/ingest.py --start-year 2021 --end-year 2024
python ml/labels.py --input data/raw/statcast_2021.parquet
python ml/arsenal.py --input data/processed/labeled.parquet
python ml/train_pitch_outcome.py --input data/processed/labeled.parquet
python ml/train_bip_outcome.py   --input data/processed/labeled_bip.parquet

# Start API
uvicorn api.main:app --reload   # http://localhost:8000, docs at /docs

# Docker (Phase 6+)
docker-compose up --build
```

## Architecture

**`docs/PROJECT_SPEC.md` is the source of truth.** All modeling decisions, the API contract, feature definitions, and wOBA tier thresholds are specified there. Update the spec before changing any of those things.

### Two-model inference chain

```
POST /predict
  → build feature row (request fields + arsenal lookup)
  → preprocessor.transform(X)                      # ml/artifacts/preprocessor.joblib
  → pitch_outcome_model.predict_proba(X)           # 6-class: ball/called_strike/swinging_strike/foul/in_play/hit_by_pitch
  → if prediction == "in_play":
        bip_model.predict_proba(X)                 # 5-class: out/single/double/triple/home_run
  → SHAP values → plain-English explanation
  → usage lookup from data/processed/usage.parquet
  → return PredictResponse
```

`api/predict.py` owns the chain. `api/schemas.py` holds all Pydantic models — the `/predict` request/response schema there is frozen (matches §6 of the spec).

### ML module responsibilities

| File | Responsibility |
|---|---|
| `ml/ingest.py` | pybaseball pull → `data/raw/statcast_{year}.parquet`; deterministic, idempotent |
| `ml/labels.py` | Map Statcast `description`/`events` to Model 1 and Model 2 label spaces |
| `ml/features.py` | sklearn Pipeline (imputation, encoding, scaling) + chronological train/val/test split |
| `ml/arsenal.py` | Precompute per-pitcher pitch stats and count-level usage → `data/processed/` |
| `ml/train_pitch_outcome.py` | Baseline → XGBoost/LightGBM → calibration → serialize to `ml/artifacts/` |
| `ml/train_bip_outcome.py` | Same pipeline, filtered to in-play rows only |
| `ml/explain.py` | SHAP TreeExplainer → `top_factors()` → `to_plain_english()` |

### Data flow

```
data/raw/statcast_{year}.parquet       ← ml/ingest.py
data/processed/labeled.parquet         ← ml/labels.py + ml/features.py
data/processed/labeled_bip.parquet     ← ml/labels.py (in-play subset)
data/processed/arsenal.parquet         ← ml/arsenal.py
data/processed/usage.parquet           ← ml/arsenal.py
data/processed/pitcher_woba.parquet    ← ml/arsenal.py (pitcher wOBA-against tier lookup)
ml/artifacts/preprocessor.joblib       ← ml/features.py (fit on train split)
ml/artifacts/pitch_outcome_model.joblib ← ml/train_pitch_outcome.py
ml/artifacts/bip_model.joblib          ← ml/train_bip_outcome.py
```

`data/` is gitignored except `.gitkeep` placeholders. `ml/artifacts/` is gitignored.

### Key constraints

- **No leakage:** train/val/test split is chronological by `game_date` (2021–2022 train, 2023 val, 2024 test). Never split randomly at row level.
- **Batter representation:** No per-batter IDs in features — use `stand` (L/R) + `batter_woba_tier` (ordinal 0–3). Thresholds: <.310 / .310–.339 / .340–.369 / ≥.370.
- **Calibration required:** Both models must pass through `CalibratedClassifierCV` before serialization. Raw booster probabilities are not acceptable for display.
- **Scripts, not notebooks:** `notebooks/` is for EDA only. Every reproducible pipeline step must be a runnable `ml/*.py` script.
- **`/predict` contract is frozen.** The Pydantic schemas in `api/schemas.py` match `docs/PROJECT_SPEC.md §6` exactly. Any change requires updating the spec and bumping `version` in `api/main.py`.
