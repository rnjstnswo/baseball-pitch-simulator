# Baseball Pitch Simulator — Project Specification

> **Source of truth for all phases.** When any decision below conflicts with a PR or notebook, update this document first.

---

## Table of Contents

1. [Project Overview & Goals](#1-project-overview--goals)
2. [Full User Flow](#2-full-user-flow)
3. [Two-Model Architecture](#3-two-model-architecture)
4. [Feature List](#4-feature-list)
5. [MVP Batter Representation](#5-mvp-batter-representation)
6. [Frozen /predict API Contract](#6-frozen-predict-api-contract)
7. [All Other API Endpoints](#7-all-other-api-endpoints)
8. [Engineering Principles](#8-engineering-principles)
9. [Phased Plan (Phases 0–6)](#9-phased-plan-phases-06)
10. [Limitations & Future Work](#10-limitations--future-work)
11. [Metrics Plan](#11-metrics-plan)

---

## 1. Project Overview & Goals

**Baseball Pitch Simulator** is an interactive, ML-powered web application that predicts the outcome of any MLB pitch using real Statcast data. Given a pitcher, a pitch type from their actual arsenal, a target location in the strike zone, and the current game situation, the app returns a probability distribution over pitch outcomes and — when contact is made — ball-in-play outcomes, together with a plain-English SHAP-based explanation.

### Portfolio Goals

- Demonstrate an end-to-end production ML pipeline: data ingestion → feature engineering → model training → calibration → REST API → interactive frontend.
- Show domain-aware ML design: leakage-aware splitting, calibrated probabilities, chained classifiers that mirror how baseball actually works.
- Produce a deployable, publicly accessible application suitable for a data science / ML engineering portfolio.

### Non-Goals (for MVP)

- Real-time pitch tracking or live game integration.
- Per-batter modeling (avoided due to sparsity; see §5).
- Pitcher fatigue, sequencing effects, or catcher framing.
- Seasons prior to 2021 (Statcast data quality and availability vary).

---

## 2. Full User Flow

```
[1] Select Pitcher
      ↓
[2] Select Batter Configuration
      Handedness (L / R) + Quality Tier (see §5)
      ↓
[3] Set Game Situation
      balls (0–3), strikes (0–2), outs (0–2),
      inning (1–9+), score_diff (-∞ to +∞),
      runners on 1B / 2B / 3B (bool × 3)
      ↓
[4] Choose Pitch Type
      Populated from pitcher's real Statcast arsenal
      ↓
[5] Click Strike-Zone Location
      Hand-drawn SVG strike zone; click → (plate_x, plate_z)
      ↓
[6] App calls POST /predict
      ↓
[7] Display Results
      • Pitch-outcome probability bar chart
      • If in_play: ball-in-play probability bar chart
      • Plain-English explanation (top SHAP drivers)
      • Pitcher's historical usage for this count / situation
      • Optional: updated count / base / out state shown
```

---

## 3. Two-Model Architecture

### Rationale

Chaining two classifiers mirrors real baseball:
1. First ask: what happens to the pitch itself? (swing? ball? contact?)
2. If contact: what happens to the batted ball?

Keeping them separate maintains class balance in each model and allows independent calibration.

### Model 1 — Pitch Outcome Classifier

**Task:** Multiclass classification over what happens to every pitched ball.

| Label | Description |
|---|---|
| `ball` | Pitch outside the zone, no swing |
| `called_strike` | Pitch in (or bordering) the zone, no swing |
| `swinging_strike` | Batter swings and misses (includes bunts) |
| `foul` | Foul ball (any count) |
| `in_play` | Batted ball put in play (any result) |
| `hit_by_pitch` | Batter struck by pitch |

**Input:** All features in §4.  
**Output:** Softmax probability vector over the 6 classes above.

### Model 2 — Ball-in-Play Outcome Classifier

**Task:** Multiclass classification conditioned only on pitches where Model 1 predicts `in_play`.

| Label | Description |
|---|---|
| `out` | Any out (groundout, flyout, lineout, sac, DP, etc.) |
| `single` | Single |
| `double` | Double |
| `triple` | Triple |
| `home_run` | Home run |

**Input:** Same feature set as Model 1, filtered to `in_play` rows only.  
**Output:** Softmax probability vector over the 5 classes above.

### Inference Chain

```
predict(request)
  → pitch_outcome_probs = model1.predict_proba(features)
  → if argmax(pitch_outcome_probs) == "in_play":
        bip_probs = model2.predict_proba(features)
    else:
        bip_probs = None
  → explanations = SHAP(model1, features) [+ model2 if applicable]
  → return PredictResponse
```

---

## 4. Feature List

All features are derived from MLB Statcast pitch-level data pulled via `pybaseball.statcast()`.

### 4a. Pitch Characteristics

| Feature Name | Type | Statcast Column(s) | Transformation |
|---|---|---|---|
| `pitch_type` | categorical | `pitch_type` | One-hot encode (FF, SL, CH, CU, SI, FC, KC, ST, …) |
| `release_speed` | float | `release_speed` | None (mph) |
| `release_spin_rate` | float | `release_spin_rate` | None (rpm); median-impute missing |
| `pfx_x` | float | `pfx_x` | None (horizontal movement, inches) |
| `pfx_z` | float | `pfx_z` | None (vertical movement, inches) |
| `release_pos_x` | float | `release_pos_x` | None (release point x, feet) |
| `release_pos_z` | float | `release_pos_z` | None (release point z, feet) |
| `release_extension` | float | `release_extension` | None (feet toward plate) |
| `spin_axis` | float | `spin_axis` | None (degrees 0–360); median-impute |

### 4b. Strike-Zone Location

| Feature Name | Type | Statcast Column(s) | Transformation |
|---|---|---|---|
| `plate_x` | float | `plate_x` | None (feet from center; negative = arm side) |
| `plate_z` | float | `plate_z` | None (feet from ground) |
| `zone` | categorical | `zone` | One-hot encode (Statcast zones 1–14) |
| `in_zone` | bool | derived | `1` if zone ∈ {1–9} |
| `dist_from_center` | float | derived | `sqrt(plate_x² + (plate_z − sz_mid)²)` |
| `sz_top` | float | `sz_top` | Batter-specific strike-zone top (feet) |
| `sz_bot` | float | `sz_bot` | Batter-specific strike-zone bottom (feet) |

### 4c. Count & Game Situation

| Feature Name | Type | Statcast Column(s) | Transformation |
|---|---|---|---|
| `balls` | int | `balls` | None (0–3) |
| `strikes` | int | `strikes` | None (0–2) |
| `count_state` | categorical | derived | `f"{balls}-{strikes}"` one-hot (12 states) |
| `outs_when_up` | int | `outs_when_up` | None (0–2) |
| `inning` | int | `inning` | Cap at 9, then encode as `min(inning, 9)` |
| `inning_top` | bool | `inning_topbot` | `1` if top of inning |
| `score_diff` | int | derived | `bat_score − fld_score`; clip to [−10, 10] |
| `on_1b` | bool | `on_1b` | `1` if runner on 1B |
| `on_2b` | bool | `on_2b` | `1` if runner on 2B |
| `on_3b` | bool | `on_3b` | `1` if runner on 3B |
| `runners_encoded` | int | derived | `4·on_3b + 2·on_2b + on_1b` (0–7, base state) |

### 4d. Pitcher & Batter Identity

| Feature Name | Type | Statcast Column(s) | Transformation |
|---|---|---|---|
| `pitcher_id` | int | `pitcher` | Used for arsenal lookup; NOT fed directly to model (causes leakage) |
| `p_throws` | categorical | `p_throws` | Binary encode: L=0, R=1 |
| `stand` | categorical | `stand` | Binary encode: L=0, R=1 (batter handedness) |
| `batter_woba_tier` | categorical | derived | See §5; ordinal encode 0–3 |
| `pitcher_woba_tier` | categorical | derived | Same tier schema applied to pitcher's season wOBA-against; ordinal encode 0–3 |

### 4e. Arsenal Context (precomputed)

| Feature Name | Type | Source | Transformation |
|---|---|---|---|
| `pitch_usage_pct` | float | arsenal table | Pitcher's season-level usage % for this pitch type |
| `pitch_usage_in_count_pct` | float | arsenal table | Usage % for this pitch type in this exact count |
| `avg_speed_for_pitch` | float | arsenal table | Pitcher's average release speed for this pitch type |
| `avg_spin_for_pitch` | float | arsenal table | Pitcher's average spin rate for this pitch type |

---

## 5. MVP Batter Representation

### Decision

Individual batter IDs are **not** used as model features. With ~800 active batters per season and thousands of pitchers, direct encoding causes extreme sparsity and data leakage across the train/test date split.

Instead, each batter at inference time is represented by two features:

1. **`stand`** — batting handedness (L / R), directly from user selection.
2. **`batter_woba_tier`** — a quality tier derived from the batter's season wOBA (weighted On-Base Average), grouped into 4 ordinal buckets.

### wOBA Tier Definitions

Thresholds are calibrated to MLB season distributions (approximately):

| Tier | Label | wOBA Range | Ordinal |
|---|---|---|---|
| 0 | `below_avg` | < .310 | 0 |
| 1 | `average` | .310 – .339 | 1 |
| 2 | `above_avg` | .340 – .369 | 2 |
| 3 | `elite` | ≥ .370 | 3 |

At inference, the user selects a tier label (e.g. "Elite / MVP"). The API maps this to the ordinal value before running the model.

### Why wOBA?

wOBA is the best single-number summary of a batter's overall offensive value because it weights each outcome (walk, single, double, triple, HR) by its run value. It is widely available per-season and correlates strongly with the features our model learns to distinguish (swing decisions, contact quality).

---

## 6. Frozen /predict API Contract

> **This contract is frozen.** Frontend development may begin against this schema immediately using mocked responses. Any change requires updating this document first and bumping the API version.

### Endpoint

```
POST /predict
Content-Type: application/json
```

### Request Schema

```json
{
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
  "on_1b": true,
  "on_2b": false,
  "on_3b": false
}
```

#### Request Field Table

| Field | Type | Required | Valid Values | Notes |
|---|---|---|---|---|
| `pitcher_id` | int | yes | Any valid MLB pitcher MLBAM ID | Looked up from /pitchers |
| `pitch_type` | string | yes | Pitcher's actual arsenal types | Validated against pitcher's arsenal |
| `plate_x` | float | yes | −2.0 to 2.0 (feet) | Horizontal location; negative = arm side |
| `plate_z` | float | yes | 0.5 to 5.0 (feet) | Vertical location from ground |
| `batter_hand` | string | yes | `"L"`, `"R"` | Batting handedness |
| `batter_quality_tier` | string | yes | `"below_avg"`, `"average"`, `"above_avg"`, `"elite"` | Maps to ordinal 0–3 |
| `balls` | int | yes | 0, 1, 2, 3 | Current ball count |
| `strikes` | int | yes | 0, 1, 2 | Current strike count |
| `outs` | int | yes | 0, 1, 2 | Outs when pitch is thrown |
| `inning` | int | yes | 1–12 | Capped at 9 internally |
| `score_diff` | int | yes | −10 to 10 | Batter's team score − Pitcher's team score; clipped |
| `on_1b` | bool | yes | true, false | Runner on first base |
| `on_2b` | bool | yes | true, false | Runner on second base |
| `on_3b` | bool | yes | true, false | Runner on third base |

### Response Schema

```json
{
  "pitch_outcome": {
    "prediction": "swinging_strike",
    "probabilities": {
      "ball": 0.12,
      "called_strike": 0.08,
      "swinging_strike": 0.45,
      "foul": 0.22,
      "in_play": 0.12,
      "hit_by_pitch": 0.01
    }
  },
  "bip_outcome": {
    "prediction": "out",
    "probabilities": {
      "out": 0.62,
      "single": 0.21,
      "double": 0.10,
      "triple": 0.02,
      "home_run": 0.05
    }
  },
  "explanation": "This pitch is likely to produce a swinging strike. The biggest factors are: the two-strike count (+), the pitch location low and away from a right-handed batter (+), and the high spin rate of this four-seamer (+). The elite batter tier slightly reduces swing-miss probability (−).",
  "top_shap_factors": [
    {"feature": "strikes", "value": 2, "shap_value": 0.18, "direction": "increases_whiff"},
    {"feature": "plate_x", "value": -0.5, "shap_value": 0.14, "direction": "increases_whiff"},
    {"feature": "release_spin_rate", "value": 2450, "shap_value": 0.11, "direction": "increases_whiff"},
    {"feature": "batter_quality_tier", "value": "elite", "shap_value": -0.07, "direction": "decreases_whiff"}
  ],
  "usage_context": {
    "pitch_usage_overall_pct": 0.34,
    "pitch_usage_in_count_pct": 0.51,
    "count": "1-2",
    "sample_size": 312
  },
  "updated_state": {
    "balls": 1,
    "strikes": 3,
    "outs": 1,
    "on_1b": true,
    "on_2b": false,
    "on_3b": false,
    "at_bat_result": null
  }
}
```

#### Response Field Table

| Field | Type | Nullable | Notes |
|---|---|---|---|
| `pitch_outcome.prediction` | string | no | Argmax of probability vector |
| `pitch_outcome.probabilities` | object | no | 6-class calibrated probabilities; sum to 1.0 |
| `bip_outcome` | object | yes | `null` unless `pitch_outcome.prediction == "in_play"` |
| `bip_outcome.prediction` | string | yes | Argmax of BIP probability vector |
| `bip_outcome.probabilities` | object | yes | 5-class calibrated probabilities; sum to 1.0 |
| `explanation` | string | no | Plain-English SHAP summary (1–3 sentences) |
| `top_shap_factors` | array | no | Top 4 SHAP contributors; each has `feature`, `value`, `shap_value`, `direction` |
| `usage_context.pitch_usage_overall_pct` | float | no | Season-level usage % for this pitch type |
| `usage_context.pitch_usage_in_count_pct` | float | no | Usage % in this specific count |
| `usage_context.count` | string | no | e.g. `"1-2"` |
| `usage_context.sample_size` | int | no | Number of pitches in arsenal table for this count |
| `updated_state` | object | no | Resulting game state after this pitch (strikes=3 → strikeout, etc.) |

---

## 7. All Other API Endpoints

### GET /pitchers

Returns the list of pitchers available in the model's training data.

**Request:** No parameters (optional query: `?search=<name>`)

**Response:**
```json
{
  "pitchers": [
    {"pitcher_id": 543037, "full_name": "Gerrit Cole", "team": "NYY", "p_throws": "R"},
    {"pitcher_id": 605400, "full_name": "Shohei Ohtani", "team": "LAD", "p_throws": "R"}
  ]
}
```

### GET /pitchers/{pitcher_id}/arsenal

Returns the pitcher's pitch type arsenal with aggregate statistics.

**Response:**
```json
{
  "pitcher_id": 543037,
  "full_name": "Gerrit Cole",
  "season": 2023,
  "arsenal": [
    {
      "pitch_type": "FF",
      "pitch_name": "4-Seam Fastball",
      "usage_pct": 0.34,
      "avg_speed": 97.2,
      "avg_spin": 2450,
      "avg_pfx_x": -4.2,
      "avg_pfx_z": 14.1,
      "sample_size": 1204
    }
  ]
}
```

### GET /pitchers/{pitcher_id}/usage

Returns pitch usage broken down by count and game situation.

**Query params:** `pitch_type` (optional), `count` (optional, e.g. `"1-2"`)

**Response:**
```json
{
  "pitcher_id": 543037,
  "usage_by_count": [
    {"count": "0-0", "pitch_type": "FF", "usage_pct": 0.55, "sample_size": 340},
    {"count": "1-2", "pitch_type": "SL", "usage_pct": 0.42, "sample_size": 198}
  ]
}
```

### GET /health

Liveness check for the API.

**Response:**
```json
{"status": "ok", "model_loaded": true, "version": "0.1.0"}
```

---

## 8. Engineering Principles

### 8.1 Leakage-Aware Splitting

- **Never** split train/test randomly at the row level — adjacent pitches to the same batter/pitcher are correlated.
- **Default split strategy:** chronological cutoff. Use pitches from 2021–2022 for training, 2023 for validation, 2024 for test. This simulates real deployment (model trained on past data, evaluated on future data).
- **Alternative for pitcher-level evaluation:** group-by-pitcher split to measure generalization to new pitchers.
- Verify no feature is derived from test-set labels before the split point.

### 8.2 Probability Calibration

- Raw classifier outputs (especially XGBoost/LightGBM) are not well-calibrated probabilities.
- Apply `sklearn.calibration.CalibratedClassifierCV` (isotonic or Platt scaling) after fitting.
- Validate calibration with reliability diagrams and Expected Calibration Error (ECE) on the validation set.
- Both models (pitch outcome and BIP) must be calibrated independently.

### 8.3 Baselines First

- Before tuning any boosted tree, fit: (a) majority-class dummy, (b) logistic regression baseline, (c) single decision tree.
- Record baseline metrics in §11 results table before moving to complex models.
- A complex model that doesn't beat the logistic baseline on log-loss is not worth deploying.

### 8.4 Scripts, Not Just Notebooks

- Any reproducible step — data ingestion, label mapping, feature engineering, training, evaluation — must have a corresponding runnable Python script in `ml/`.
- Notebooks in `notebooks/` are for EDA and one-off exploration only; they are never the source of truth for a pipeline step.
- Scripts must be runnable with a single `python ml/<script>.py` command and must not require manual state (no "run cells 1–5 first").

### 8.5 Freeze the /predict Contract Early

- The response schema in §6 is frozen as of Phase 0.
- Frontend development may begin against a mock server immediately after Phase 0.
- Changes to the contract require a doc update + API version bump + frontend update in the same PR.

### 8.6 Deterministic Ingestion

- `ml/ingest.py` must produce the same Parquet files on every run given the same date range.
- Cache raw Statcast pulls to `data/raw/` so re-runs do not re-hit the API.
- Log row counts and date ranges at each ingestion step.

---

## 9. Phased Plan (Phases 0–6)

### Phase 0 — Foundation

**Goals:** Repo structure, tooling, and spec finalized before any code is written.

**Tasks:**
- [x] Write `docs/PROJECT_SPEC.md` (this document).
- [x] Scaffold repo structure: directories, stub files, configs.
- [x] Set up `pyproject.toml` (black, ruff, isort), `.pre-commit-config.yaml`, `.gitignore`.
- [x] Decide and document outcome taxonomy (§3).
- [x] Freeze `/predict` contract (§6).
- [ ] Initialize git, push to GitHub.

**Exit Criteria:**
- `pre-commit run --all-files` passes on all stub files.
- `docs/PROJECT_SPEC.md` is committed and reviewed.
- All stub files exist with correct docstrings and function signatures.

---

### Phase 1 — Data Pipeline

**Goals:** Pull, cache, and validate raw Statcast data.

**Tasks:**
- Implement `ml/ingest.py`: pull Statcast data for 2021–2024 via `pybaseball.statcast()`, cache to `data/raw/*.parquet`.
- Add CLI flags: `--start-year`, `--end-year`, `--force-refresh`.
- Log row counts, date range, null rates per column.
- Run EDA in `notebooks/01_eda.ipynb`: pitch type distributions, outcome distributions by year, null analysis, class imbalance check.

**Exit Criteria:**
- `python ml/ingest.py` runs end-to-end without error.
- Parquet files land in `data/raw/` with correct schema.
- EDA notebook documents: class distribution for Model 1 labels, class distribution for Model 2 labels, top-10 null-rate columns, pitch type counts by pitcher.
- No leakage: raw data contains no derived label columns.

---

### Phase 2 — Features & Labels

**Goals:** Map raw Statcast columns to model-ready features and labels.

**Tasks:**
- Implement `ml/labels.py`: map `description` and `events` columns to Model 1 and Model 2 taxonomy (§3).
- Implement `ml/features.py`: full preprocessing pipeline (imputation, encoding, scaling) using `sklearn.Pipeline`.
- Implement `ml/arsenal.py`: precompute per-pitcher pitch type stats and count-level usage tables, write to `data/processed/arsenal.parquet`.
- Unit tests: verify label mapping covers all Statcast `description` values; verify features produce no NaNs post-pipeline.
- Date-based train/val/test split logic.

**Exit Criteria:**
- `python ml/labels.py` outputs a labeled Parquet with no unmapped rows.
- `python ml/arsenal.py` outputs `data/processed/arsenal.parquet`.
- Feature pipeline produces zero NaN values on validation split.
- Label distribution logged and matches expected class ratios.

---

### Phase 3 — Modeling

**Goals:** Train, calibrate, and explain both classifiers.

**Tasks:**
- Train baselines (dummy, logistic regression, single tree) for both models.
- Train tuned XGBoost or LightGBM for both models (hyperparameter search with `optuna` or `GridSearchCV`).
- Calibrate both final models with `CalibratedClassifierCV`.
- Compute SHAP values; implement `ml/explain.py` to generate plain-English summaries.
- Serialize: `ml/artifacts/pitch_outcome_model.joblib`, `ml/artifacts/bip_model.joblib`, `ml/artifacts/preprocessor.joblib`.
- Evaluate on held-out 2024 test split.

**Exit Criteria:**
- Results table (§11) filled in for all models.
- Tuned model beats logistic regression on log-loss for both Model 1 and Model 2.
- Calibration curve (reliability diagram) shows ECE < 0.05 for both models.
- SHAP explanations generate in < 200ms per prediction (precompute background dataset).
- All artifacts are serialized and loadable from a clean Python environment.

---

### Phase 4 — Backend API

**Goals:** Expose models via a production-ready FastAPI service.

**Tasks:**
- Implement `api/main.py`: wire up all endpoints (§6, §7).
- Implement `api/schemas.py`: Pydantic v2 models for all request/response types.
- Implement `api/predict.py`: load serialized artifacts, run inference chain, call SHAP explanation.
- Add input validation: validate `pitch_type` against pitcher's actual arsenal.
- Write `api/tests/test_predict.py`: at least (a) valid request → 200, (b) invalid pitch type → 422, (c) out-of-range plate location → 422.
- Add `api/Dockerfile`.

**Exit Criteria:**
- `pytest api/tests/` passes.
- `uvicorn api.main:app` starts without errors.
- `POST /predict` with the example request in §6 returns a valid response.
- `GET /pitchers`, `/arsenal`, `/usage`, `/health` all return expected schemas.
- Docker image builds and container responds to requests.

---

### Phase 5 — Frontend

**Goals:** Build the interactive React/TypeScript UI.

**Tasks:**
- `npm create vite@latest frontend -- --template react-ts` inside `frontend/`.
- Implement pitcher selector (search dropdown).
- Implement batter configuration panel (handedness + tier).
- Implement situation panel (count, outs, inning, score diff, runners).
- Implement pitch type selector (populated from `/arsenal`).
- Implement clickable SVG strike zone → `(plate_x, plate_z)`.
- Implement prediction result panel: Recharts bar charts, explanation text, usage context.
- Optional: updated count/base/out state display.
- Configure TanStack Query for all API calls with loading/error states.
- Style with Tailwind CSS.

**Exit Criteria:**
- Full user flow (§2) completable end-to-end in browser.
- Strike zone click accurately captures `(plate_x, plate_z)` within ±0.05 feet.
- Prediction result renders within 1s on local network (excluding model inference).
- No TypeScript compilation errors (`tsc --noEmit` passes).
- Responsive on 1280px+ screen widths.

---

### Phase 6 — Polish, Deploy & Docs

**Goals:** Production-quality app, deployed and documented.

**Tasks:**
- Design pass: consistent color scheme, baseball-themed typography.
- Frontend unit tests (Vitest): strike zone click accuracy, probability chart rendering.
- `docker-compose up` brings up API + frontend together.
- Deploy API to Render or Fly.io; deploy frontend to Vercel.
- GitHub Actions CI: lint → test → build on every PR.
- Write portfolio `README.md`: demo GIF, architecture diagram, local setup instructions, live link.
- Update `docs/architecture.md` with final system diagram.

**Exit Criteria:**
- Live URL accessible and functional.
- CI passes on `main` branch.
- `README.md` has demo GIF, architecture description, and setup instructions under 10 steps.
- `docker-compose up` starts the full stack from a cold machine in < 5 minutes.

---

## 10. Limitations & Future Work

### Current Limitations

| Limitation | Impact | Mitigation in MVP |
|---|---|---|
| **Correlation, not causation** | Model learns historical patterns; a high-probability swinging strike is not guaranteed — it reflects what happened in similar situations, not pitcher intent. | Disclaimer in UI explanation text. |
| **MVP batter simplification** | Replacing individual batters with wOBA tiers loses information about specific batter tendencies (pull hitter, high-K%, etc.). | Acknowledged; enables generalization with reasonable accuracy. |
| **No fatigue or sequencing** | Model ignores pitch count, prior pitches in the at-bat, days of rest, or game-day warm-up. | Out of scope for MVP; noted in UI. |
| **Static season averages** | Arsenal and wOBA tiers are season-level averages; month-to-month variation (hot/cold streaks) is ignored. | Acceptable for portfolio scope. |
| **Limited seasons (2021–2024)** | Model does not capture multi-year trends or players whose peak years were earlier. | Prioritizes Statcast data quality. |
| **Outcome taxonomy simplification** | "Out" in Model 2 lumps groundouts, flyouts, sacrifice flies, and double plays together. | Acceptable for binary in-play modeling depth. |

### Future Work

- **Per-batter modeling** once per-batter sample sizes grow (with regularization / hierarchical models).
- **Pitch sequencing model:** incorporate prior pitch(es) in at-bat as context features.
- **Catcher framing:** incorporate called-strike probability given catcher identity.
- **Live game integration:** connect to MLB Stats API for real-time situation pre-fill.
- **Pitcher fatigue proxy:** pitch count in game, days since last appearance.
- **Multi-season ensemble:** average predictions across seasons with temporal weighting.

---

## 11. Metrics Plan

### Primary Metrics

| Model | Metric | Rationale |
|---|---|---|
| Pitch Outcome (M1) | **Log-loss** | Primary; penalizes confident wrong predictions |
| Pitch Outcome (M1) | **Macro-F1** | Ensures minority classes (HBP, triple) are not ignored |
| Pitch Outcome (M1) | **ECE** (Expected Calibration Error) | Ensures probabilities are trustworthy |
| BIP Outcome (M2) | **Log-loss** | Same rationale |
| BIP Outcome (M2) | **Macro-F1** | Triple/home_run are rare; macro weighting matters |
| BIP Outcome (M2) | **ECE** | Calibration required for display to users |

### Secondary Diagnostics

- Confusion matrices for both models on 2024 test split.
- Reliability diagrams (calibration curves) for both models.
- Per-pitch-type breakdown of Model 1 performance.
- Per-count-state breakdown of Model 1 performance.

### Results Table (to be filled in Phase 3)

#### Model 1 — Pitch Outcome

| Model | Log-loss | Macro-F1 | ECE |
|---|---|---|---|
| Majority-class dummy | — | — | — |
| Logistic Regression (baseline) | — | — | — |
| Single Decision Tree (baseline) | — | — | — |
| XGBoost / LightGBM (tuned) | — | — | — |
| XGBoost / LightGBM + Calibration | — | — | **target < 0.05** |

#### Model 2 — Ball-in-Play Outcome

| Model | Log-loss | Macro-F1 | ECE |
|---|---|---|---|
| Majority-class dummy | — | — | — |
| Logistic Regression (baseline) | — | — | — |
| Single Decision Tree (baseline) | — | — | — |
| XGBoost / LightGBM (tuned) | — | — | — |
| XGBoost / LightGBM + Calibration | — | — | **target < 0.05** |

---

*Last updated: Phase 0 — Foundation*
