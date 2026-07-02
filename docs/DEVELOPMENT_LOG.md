# Development Log

A phase-by-phase record of the work done on the Baseball Pitch Simulator: the
goal of each phase, what exists once the phase is complete, the files touched,
the role each file plays, and the specific edits made.

> Source of truth for *decisions* is [`docs/PROJECT_SPEC.md`](PROJECT_SPEC.md).
> This log records *what was built and changed*, not the spec itself.

---

## Phase 0 — Foundation

**Commit:** `21a0dbd` Phase 0: project spec and repo scaffold
(+ `ba2e911`, `371cbf8` adding `CLAUDE.md`)

**Goal:** Lock the spec, repo structure, and tooling before writing pipeline
code. Freeze the `/predict` contract and the outcome taxonomy so later phases
build against a fixed target.

**State after phase:**
- Full project spec written; `/predict` request/response schema frozen.
- Repo scaffolded with stub modules (correct docstrings + function signatures,
  bodies raise `NotImplementedError`).
- Lint/format/test tooling configured; `pre-commit run --all-files` passes on
  the stubs.

**Files created and their roles:**

| File | Role |
|---|---|
| `docs/PROJECT_SPEC.md` | Source of truth: architecture, feature list (§4), batter representation (§5), frozen `/predict` contract (§6), phased plan (§9), metrics plan (§11). |
| `docs/architecture.md` | Placeholder for the final system diagram. |
| `README.md` | Portfolio README (filled out later in Phase 6). |
| `.gitignore` | Ignores `data/` (except `.gitkeep`), `ml/artifacts/`, `.venv/`, caches. |
| `.pre-commit-config.yaml` | Runs black + ruff + isort + file checks. |
| `pyproject.toml` | black / ruff / isort / pytest config. |
| `ml/ingest.py` | Stub: Statcast pull → `data/raw/`. |
| `ml/labels.py` | Stub: outcome taxonomy mapping. |
| `ml/features.py` | Stub: preprocessing pipeline + split. |
| `ml/arsenal.py` | Stub: per-pitcher arsenal/usage tables. |
| `ml/train_pitch_outcome.py` | Stub: Model 1 training. |
| `ml/train_bip_outcome.py` | Stub: Model 2 training. |
| `ml/explain.py` | Stub: SHAP explanation. |
| `api/main.py`, `api/predict.py`, `api/schemas.py` | Stub FastAPI app, inference chain, Pydantic schemas. |
| `api/tests/test_predict.py` | Stub API tests. |
| `docker-compose.yml`, `frontend/README.md` | Placeholders for later phases. |
| `requirements.txt` | Pinned dependency set (ML + backend + tooling). |
| `CLAUDE.md` | Coding guidelines + command/architecture guidance for the assistant. |

---

## Phase 1 — Data Pipeline

**Commits:** `699f229` (ingest + EDA), `dd08757` (execute notebook, fix lint,
fix build-backend)

**Goal:** Pull, cache, and validate raw Statcast data for 2021–2024; run EDA to
confirm class distributions and null rates before any modeling.

**State after phase:**
- `python ml/ingest.py` pulls Statcast 2021–2024 and caches deterministic
  Parquet files to `data/raw/` (**3,073,685 rows**).
- EDA notebook executed in-place with outputs: pitch-type distributions,
  Model 1 / Model 2 label distributions, null-rate analysis, per-pitcher counts.
- Build backend bug fixed so the package is installable.

**Files edited and their roles / edits:**

| File | Role | Edit made |
|---|---|---|
| `ml/ingest.py` | pybaseball pull → `data/raw/statcast_{year}.parquet`; idempotent, logs row counts & null rates. | Implemented full pull logic + CLI flags `--start-year` / `--end-year` / `--force-refresh`. |
| `notebooks/01_eda.ipynb` | EDA only (not a pipeline step). | Authored cells; executed in-place; fixed lint (removed unused `numpy`, reordered imports, removed empty f-string, renamed unused loop vars). |
| `pyproject.toml` | Build + tooling config. | Fixed invalid `build-backend` → `setuptools.build_meta`. |
| `api/*`, `ml/labels.py`, `ml/arsenal.py`, `ml/train_*.py` | Stubs. | Lint-only touch-ups from `pre-commit` auto-fixes. |

---

## Phase 2 — Features & Labels

**Commit:** `873396b` Implement Phase 2: labels, features, arsenal tables + unit tests

**Goal:** Map raw Statcast columns to the model label space and a model-ready
feature matrix; precompute the per-pitcher serving tables; add a leakage-aware
chronological split.

**State after phase:**
- `python ml/labels.py` → `labeled.parquet` (3.07M rows) + `labeled_bip.parquet`
  (in-play subset), **zero unmapped `description` values**.
- `python ml/arsenal.py` → `arsenal.parquet`, `usage.parquet`,
  `pitcher_woba.parquet`.
- Feature pipeline produces **zero NaN on the validation split** (75 features,
  fit on train / transform on val).
- 10 unit tests covering label coverage/taxonomy and feature/pipeline/split.

**Files edited and their roles / edits:**

| File | Role | Edit made |
|---|---|---|
| `ml/labels.py` | Map `description` → 6-class `pitch_outcome`; `events` → 5-class `bip_outcome`; coverage validation. | Implemented `PITCH_OUTCOME_MAP` / `BIP_OUTCOME_MAP`, `map_pitch_outcome`, `map_bip_outcome`, `add_labels`, `validate_coverage`, CLI. Decisions: `foul_tip`/`bunt_foul_tip`/`foul_pitchout`→`foul`; `automatic_ball`→`ball`; `automatic_strike`→`called_strike`; `field_error`→`out`. |
| `ml/features.py` | sklearn preprocessing pipeline + derived features + chronological split. | Implemented `engineer_features` (all §4 derived cols), `compute_season_woba`, `woba_to_ordinal`, `build_pipeline` (impute+scale / one-hot), `get_feature_names`, `make_train_val_test_split` (2021–22 train / 2023 val / 2024 test). |
| `ml/arsenal.py` | Precompute per-pitcher serving tables. | Implemented `build_arsenal_table`, `build_usage_table`, `build_pitcher_woba_table`, `save_tables` (3 tables), loaders. Added `ml.features` import with script-mode fallback. |
| `ml/tests/test_labels.py` | Unit tests for label mapping. | Created (6 tests). |
| `ml/tests/test_features.py` | Unit tests for features/pipeline/split. | Created (5 tests, synthetic fixture). |
| `ml/tests/__init__.py` | Test package marker. | Created (empty). |
| `docs/PROJECT_SPEC.md` | Source of truth. | Added "Label Mapping Notes (Phase 2)" subsection; documented `pitcher_woba.parquet` serving artifact. |
| `CLAUDE.md` | Guidance. | Added `labeled_bip.parquet` and `pitcher_woba.parquet` to the data-flow block. |
| `pyproject.toml` | Tooling config. | `testpaths = ["api/tests", "ml/tests"]`. |

**Post-Phase-2 follow-up commit:** `dd7d7ea` added `nbconvert` + `ipykernel` to
`requirements.txt` so the Phase 1 EDA notebook step is reproducible from a clean
clone.

---

## Pre–Phase-3 setup — Virtual environment

**Goal:** Reproducible environment for the modeling stack (no venv existed;
packages had been installed into system Python).

**Work done (not yet committed as code — environment + `requirements.txt`):**
- Created `.venv` (Python 3.13.5) and installed all of `requirements.txt`,
  including the Phase 3 stack: LightGBM 4.6.0, XGBoost 3.3.0, SHAP 0.52.0,
  Optuna 4.9.0, scikit-learn 1.9.0.
- Verified imports + existing suite (10 passed, 5 skipped).

---

## Phase 3 — Modeling

**Status:** Complete. All code implemented and unit-tested (18 ml tests pass);
lint-clean; full training run finished; §11 results tables filled; all exit
criteria met.

**Goal:** Train, calibrate, and explain both classifiers, then serialize
loadable artifacts. Exit criteria (spec §9): results table filled; tuned booster
beats logistic on log-loss for both models; calibrated ECE < 0.05; SHAP < 200ms;
all artifacts load from a clean environment.

**Confirmed design decisions:**
- **LightGBM** for both models (spec allows XGBoost *or* LightGBM).
- Shared **`ml/modeling.py`** holds logic common to both training scripts.
- **Calibrate on the validation split** (spec §8.2); report final metrics on the
  untouched 2024 test split.
- **SHAP runs on the raw booster**, not the `CalibratedClassifierCV` wrapper
  (TreeExplainer can't explain the calibrator; calibration is monotonic per
  class so it doesn't change which features drive a class). `build_explainer`
  unwraps the booster from the calibrated model automatically.
- **Model 2 engineers features on the full frame, then filters to in-play**, so
  the §4e arsenal-context denominators count every pitch (filtering first would
  corrupt usage %).
- **Natural base rates (no `class_weight`)** for the booster: log-loss is the
  primary metric and calibrated probabilities must reflect true outcome
  frequencies for display. Rare-class macro-F1 is modest and reported honestly.
- Isotonic calibration via **`FrozenEstimator`** (the modern replacement for the
  removed `cv="prefit"` in scikit-learn 1.9).

**Files created / edited and their roles / edits:**

| File | Role | Edit made |
|---|---|---|
| `ml/modeling.py` *(new)* | Shared training/eval helpers used by both models. | Implemented `load_features` (engineer→filter→split), `train_baselines` (dummy / logistic / decision tree), `tune_lgbm` (Optuna over LightGBM, val log-loss objective, native early stopping), `calibrate` (isotonic via `FrozenEstimator`), `evaluate` (log-loss / macro-F1 / ECE), `compute_ece` (top-label confidence binning). |
| `ml/train_pitch_outcome.py` | Model 1 (6-class) orchestration. | Implemented `_run` (load → fit preprocessor on train → baselines → tune → calibrate-on-val → eval-on-test → serialize), `serialize` (model + preprocessor), exit-criteria check, metrics JSON dump, CLI (`--n-trials`, `--sample-frac`). Public interface re-exported from `modeling`. |
| `ml/train_bip_outcome.py` | Model 2 (5-class, in-play only) orchestration. | Same as M1 but **reuses the saved `preprocessor.joblib`** (transform only) and loads via `in_play_only=True`; `serialize` writes `bip_model.joblib`. |
| `ml/explain.py` | SHAP explanation generation. | Implemented `_unwrap_booster`, `build_explainer` (TreeExplainer on the booster), `compute_shap_values`, `top_factors` (top-n by \|shap\|), `to_plain_english`. |
| `ml/tests/test_modeling.py` *(new)* | Unit tests for modeling helpers. | Created: ECE on perfectly-/badly-calibrated input, finite metrics, train→calibrate→predict end-to-end. |
| `ml/tests/test_explain.py` *(new)* | Unit tests for explanations. | Created: top-factor ordering/sizing, plain-English phrasing, <200ms SHAP latency. |

**Artifacts produced by the run (gitignored, under `ml/artifacts/`):**
- `preprocessor.joblib` — fitted feature pipeline (shared by both models).
- `pitch_outcome_model.joblib` — calibrated Model 1.
- `bip_model.joblib` — calibrated Model 2.
- `metrics_pitch_outcome.json`, `metrics_bip_outcome.json` — test metrics.

**Results (2024 test split, LightGBM, 30 Optuna trials):**

| | Model 1 (pitch outcome) | Model 2 (ball-in-play) |
|---|---|---|
| Logistic (baseline) log-loss | 1.1126 | 0.9162 |
| LightGBM tuned log-loss | 0.6749 | 0.9104 |
| Calibrated log-loss / macro-F1 / ECE | 0.6742 / 0.6548 / 0.0179 | 0.9091 / 0.1631 / 0.0028 |

**Exit criteria — all met:**
- [x] §11 results tables filled with real test metrics.
- [x] Tuned beats logistic on log-loss for both models.
- [x] Calibrated ECE < 0.05 for both (0.0179 / 0.0028).
- [x] SHAP explanation < 200ms (unit test `test_shap_latency_under_200ms`).
- [x] Artifacts load and run a full chain from a clean process.

**Honest finding:** Model 2 only marginally beats the dummy (0.9091 vs 0.9320).
Batted-ball outcome is low-signal from pre-contact features alone (exit velocity /
launch angle excluded as leakage), so rare-class macro-F1 stays low. Accepted MVP
limitation (spec §10).

---

---

## Phase 4 — Backend API

**Status:** Complete. Two-model inference chain served via FastAPI; all 5
endpoints implemented; 11 API tests pass (29 total across the repo); lint-clean.
Docker image builds and the container serves all endpoints (verified 2026-07-02).

**Goal:** Expose both models via a production FastAPI service honoring the frozen
`/predict` contract (spec §6) and the §7 endpoints. Exit criteria (spec §9):
pytest passes; app boots; `/predict` returns a valid response; `/pitchers`,
`/arsenal`, `/usage`, `/health` return expected schemas; Docker image builds.

**Data-prep prerequisite discovered during planning:**
The `/pitchers` contract needs `full_name`/`team`/`p_throws` and the Model 1
feature row needs `p_throws` + `release_pos_x/z` + `release_extension` +
`spin_axis` + `sz_top/bot` — none of which existed in the processed tables (only
in raw Statcast). `labeled.parquet` holds raw columns + labels only; the
engineered features are computed at train time, so the inference layer replicates
that engineering manually (it can't reuse `engineer_features`, whose groupby
aggregates are meaningless on one row).

**Confirmed design decisions:**
- **Missing pitch-physics features → per-pitcher medians** (Option A): extended
  `ml/arsenal.py` to store `median_release_pos_x/z`, `median_release_extension`,
  `median_spin_axis` per (pitcher, season, pitch_type). `sz_top`/`sz_bot` are
  batter-driven (no pitcher-level value) → single league-median constants in
  `api/predict.py`. `inning_top` is unknown from the request → left NaN for the
  fitted median imputer.
- **New `pitchers.parquet`** (name/team/handedness) built in the same arsenal pass.
- **`updated_state` uses real baseball logic** (strike 3 → strikeout, ball 4 →
  walk, in-play resolves on the BIP prediction); runner advancement is simplified
  (documented, spec §10). Fixed the internally-inconsistent §6 example to match.
- **SHAP explainer built with no background dataset** (the validated <200ms
  `tree_path_dependent` path). Only Model 1 is explained (the pitch outcome).
- **Season rule:** each pitcher's latest season in the arsenal, applied
  consistently across all lookups.
- **Error codes:** unknown pitcher → 404; pitch_type not in arsenal → 422.
- **Contract version bumped to 0.2.0** (defined `updated_state` semantics).

**Files created / edited and their roles / edits:**

| File | Role | Edit made |
|---|---|---|
| `ml/arsenal.py` | Precompute lookup tables. | Added 4 release/spin `median_*` columns to `build_arsenal_table`; new `build_pitchers_table` (name/team/p_throws) + `load_pitchers`; `save_tables` now writes `pitchers.parquet`. Re-ran the pipeline. |
| `api/schemas.py` | Pydantic models. | Added `PitchersResponse`, `ArsenalResponse`, `UsageResponse` wrappers (additive; frozen `/predict` models untouched). |
| `api/predict.py` | Inference chain. | Implemented `load_artifacts` (loads 3 joblib + 4 parquet, builds the SHAP explainer), `_build_feature_row` (assembles the 32 numeric + 3 categorical features from request + lookups), `_statcast_zone`, `run_inference` (2-model chain + SHAP + usage + state), `_map_shap_factors` (transformed→raw name/value), `_compute_updated_state`. |
| `api/main.py` | App + endpoints. | `lifespan` loads the bundle into `app.state`; implemented `/health`, `/pitchers`, `/pitchers/{id}/arsenal`, `/pitchers/{id}/usage`, `/predict` with 404/422 handling; version 0.2.0. |
| `api/tests/test_predict.py` | API tests. | Un-skipped the 5 stubs; module-scoped `client` fixture (triggers lifespan); added `/health`, `/pitchers`, `/arsenal`, `/usage`, and unknown-pitcher-404 tests. |
| `api/Dockerfile` *(new)* | Container image. | slim Python + libgomp1, installs requirements, copies `api/`/`ml/`/artifacts/processed parquets, runs uvicorn. |
| `.dockerignore` *(new)* | Build context. | Excludes `data/raw/`, notebooks, venv, caches so the image stays lean. |

**Verified:** `pytest` 29 passed; `TestClient` boots via lifespan and `/health`
returns `{"status":"ok","model_loaded":true,"version":"0.2.0"}`; the §6 example
request returns a valid `PredictResponse` (probabilities sum to 1.0). Docker:
`docker build -f api/Dockerfile -t pitch-sim-api .` succeeds (image 3.65GB) and
the running container answers `/health`, `/predict`, `/pitchers`, `/arsenal`,
`/usage` correctly, including 404 on an unknown pitcher. (Required a WSL2 kernel
update — `wsl --update` — before Docker Desktop's engine would start.)

---

## Phase 5 — Frontend

**Status:** Complete. Full §2 user flow works end-to-end in the browser against
the live API; `tsc -b` clean; zone-click accuracy verified within ±0.05 ft.

**Goal:** Interactive React/TypeScript UI per spec Phase 5: pitcher search,
batter config, situation panel, arsenal-driven pitch selector, clickable SVG
strike zone, prediction results with charts.

**Setup:** Node.js 24.18.0 LTS installed via winget (was absent from the
machine). Scaffolded with `create-vite` (react-ts template), Vite 8 + React 19.
Deps: `tailwindcss` + `@tailwindcss/vite` (v4 plugin path, no config file),
`recharts`, `@tanstack/react-query`.

**Structure (`frontend/src/`):**

| File | Role |
|---|---|
| `api/types.ts` | TS mirrors of every Pydantic schema (frozen §6 contract noted) |
| `api/client.ts` | fetch wrapper over `VITE_API_URL` (default `http://localhost:8000`), `ApiError` with FastAPI `detail` |
| `api/hooks.ts` | TanStack Query hooks: `usePitchers` / `useArsenal` / `useUsage` / `usePredict` (mutation) |
| `components/ButtonGroup.tsx` | shared segmented single-select row |
| `components/PitcherSelector.tsx` | search combobox, client-side filter, first 50 matches |
| `components/BatterPanel.tsx` | L/R + wOBA tier button groups |
| `components/SituationPanel.tsx` | count/outs button groups, inning select (1–12), score-diff input (clamped ±10), runner toggles |
| `components/PitchTypeSelector.tsx` | arsenal buttons with usage % + avg velo |
| `components/StrikeZone.tsx` | SVG spanning the full valid range (x ±2, z 0.5–5); zone rect at rulebook width and the API's `LEAGUE_SZ_TOP/BOT`; click mapped screen→feet via inverse `getScreenCTM` |
| `components/ProbabilityChart.tsx` | Recharts horizontal bars, single hue, sorted desc, % labels at tips |
| `components/ResultsPanel.tsx` | headline prediction, both charts (BIP conditional), SHAP explanation, usage sentence, updated state line |
| `App.tsx` | all state (`useState`), predict fires on zone click; 3-column grid at `lg:` |

**Design decisions:**
- Predict fires on zone click only (re-click to re-predict); pitcher change
  resets pitch type, location, and the previous result.
- Charts follow the dataviz method: single measure → one hue (`#2a78d6`,
  validated), no legend, ≤24px bars with 4px rounded data-ends, hairline grid,
  values at bar tips in ink (not series color).
- Strike zone drawn at the exact league-median vertical bounds the model
  assumes, so what the user sees matches what the model is told.
- No server-side `?search=` — the full pitcher list is one fetch; filtering is
  client-side.

**Verified (Playwright vs system Edge, headless):** full flow — search "Gerrit"
→ select Cole → LHB/elite → 1-2 count, 1 out, runner on 1B → 4-Seam Fastball →
zone click at (−0.5, 2.8) → readout (−0.49, 2.80) (±0.05 ft criterion) →
prediction panel renders chart + explanation + usage + state; second click
re-predicts; zero browser console errors. `tsc -b` passes. Screenshots reviewed.

---

*Last updated: Phase 5 — Frontend complete — 2026-07-02*
