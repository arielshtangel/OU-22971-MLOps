# Capstone Project Summary
## MLOps Unit 8 – NYC Green Taxi Tip Prediction

---

## ✅ What Has Been Done

### Phase 0 — Solution Plan
- Created **`8/solution_steps.md`** — a 19-step ordered execution contract aligned with the Design Doc.
  All implementation decisions trace back to specific Design Doc sections.

### Phase 1–2 — Repository Review & Design Doc Analysis
- Reviewed all existing Unit 6 and Unit 7 source files:
  - `green_taxi_drift_lib.py` — feature engineering, integrity checks, drift utilities
  - `train_initial.py`, `retrain.py`, `train_register.py`, `flip_aliases.py`
- Extracted all functional requirements, constraints, and anti-footgun rules from `8/design_doc.md`.

### Phase 3–4 — Implementation

#### Environment
| File | Change |
|------|--------|
| `Tools/MLOps/environment.yml` | Added `metaflow` and `nannyml` to pip dependencies |

#### New files created in `8/capstone/`

| File | What it does |
|------|-------------|
| `capstone_lib.py` | Shared utilities — re-exports Unit 6 helpers, adds: `rmse()`, `write_decision_json()`, `run_hard_integrity_checks()`, `run_nannyml_soft_checks()`, `ensure_registered_model()`, `get_champion_version()` |
| `capstone_flow.py` | **Main Metaflow workflow** — full implementation of flow steps A→G (see below) |
| `inference.py` | Offline batch inference — loads `@champion`, applies feature pipeline, saves `predictions.parquet`, logs to MLflow |
| `README.md` | Setup guide, all 3 demo run commands, MLflow UI navigation guide |
| `.gitignore` | Excludes `data/` from version control |
| `summary.md` | This file |

#### Flow Steps Implemented (`capstone_flow.py`)

```
start → load_data → integrity_gate ──(hard fail)──→ end
                         │
                  feature_engineering
                         │
                  load_champion  ←── (bootstrap initial model if no @champion)
                         │
                  model_gate ──(no retrain needed)──→ candidate_acceptance → end
                         │
                       retrain  (Optuna 15-trial hyperparameter tuning)
                         │
                  candidate_acceptance  (P1–P4 gates → promote or reject)
                         │
                         end
```

| Step | Description |
|------|-------------|
| **start** | Creates MLflow run, logs flow params, stores `mlflow_run_id` as Metaflow artifact |
| **load_data** | Loads reference + batch `.parquet` files |
| **integrity_gate** | Layer 1 hard rules (fail-fast → `action="reject_batch"`); Layer 2 NannyML soft checks (missingness spike + unseen categoricals) |
| **feature_engineering** | `make_tip_frame` + `align_feature_frame` + `cast_ints_to_float` on both splits; logs `feature_spec.json` |
| **load_champion** | Loads `@champion` alias; if absent: trains bootstrap model, registers it, sets alias |
| **model_gate** | Evaluates champion on batch; logs `rmse_champion`, `rmse_baseline`, `rmse_increase_pct`; writes `model_gate/decision.json` |
| **retrain** | Rolling window training (ref + batch); Optuna tuning; registers candidate with all required tags; logs `rmse_val` for future baseline retrieval |
| **candidate_acceptance** | Checks P1 (valid metrics), P2 (beats champion by ≥1%), P3 (stable on reference slice), P4 (no integrity failure); flips `@champion` alias on success; writes final `decision.json` |
| **end** | Calls `client.set_terminated(FINISHED)` unless already FAILED |

#### Key Design Decisions
- **MLflow run lifecycle**: Run created in `start` via `mlflow.start_run()`, then deactivated with `mlflow.end_run()`. All subsequent logging uses `MlflowClient` API directly (passing `run_id`) — this avoids run-context loss across Metaflow step boundaries.
- **NannyML integration**: Uses `SummaryStatsNullValuesCalculator` with synthesized timestamps; falls back to manual missingness comparison if NannyML unavailable or raises an error.
- **`rmse_val` key in retrain run**: The retrain step explicitly logs `mlflow.log_metric("rmse_val", rmse_candidate)` so that if this candidate is later promoted to `@champion`, the *next* flow run can retrieve `rmse_baseline` correctly.

### Phase 5 — Independent Verification
- Ran verification sub-agent against all Design Doc requirements.
- **68 ✅ / 12 ⚠️ / 1 ❌** items identified.
- Critical defect fixed: `rmse_val` not logged in retrain run (would break performance gate on all runs after first promotion).
- Additional fixes applied: `align_feature_frame` in `inference.py`, README stack header, Demo Run 3 resume note.

---

## ✅ Completed — All Demo Runs Executed Successfully on Google Colab

### Execution environment
- Ran on **Google Colab** (CPU runtime) — no local Conda environment needed.
- All packages installed via `pip install` in Cell 1 of `capstone_colab.py`.
- Code uploaded manually via `google.colab.files.upload()`.
- Data downloaded directly from NYC TLC CDN.
- MLflow server started as background process; UI tunnelled via ngrok.

### Demo Run 1 — Baseline ✅
- Batch: `green_tripdata_2023-04.parquet`, threshold: `0.99`
- **Result**: Bootstrap model created (v1, `@champion`), RMSE increase 1.0% ≤ threshold.
- MLflow evidence: `retrain_recommended=false`, `promotion_recommended=false`, `action="no_action"`.

### Demo Run 2 — Retrain + Promotion ✅
- Batch: `green_tripdata_2023-10.parquet`, threshold: `0.001`
- **Result**: Champion v1 RMSE increase triggered retrain. Candidate v2 trained (Optuna, 15 trials). P1–P4 gates all passed. v2 promoted to `@champion`.
- MLflow evidence: `retrain_recommended=true`, `promotion_recommended=true`, `action="promote"`. Model registry shows v2 with `@champion` alias.

### Demo Run 3 — Failure + Resume ✅
- Injected `RuntimeError` in `retrain` step using unique TARGET (`import optuna` line).
- Flow failed at `retrain` step (run-id `1778181108414775`).
- After removing injection, `resume retrain` restored steps start→model_gate from cache; only `retrain` + `candidate_acceptance` ran fresh.
- MLflow handled gracefully: failed run logged + new resumed run completed.

### Batch Inference ✅
- Ran `inference.py` on `green_tripdata_2023-10.parquet`.
- Output: `predictions.parquet` — **41,545 rows** with `predicted_tip_amount` and `actual_tip_amount` columns.
- Logged as MLflow artifact in `8_capstone_inference` experiment.

---

## 🔲 What Still Needs to Be Done

### 1. Record the submission video (5–10 min) *(mandatory)*
Required by the Design Doc (§8 Deliverables). Must show:
- **Code walkthrough**: `capstone_flow.py` — integrity gate, model gate, retrain, promotion logic
- **MLflow UI** (use ngrok URL or Cell 15/16 programmatic output):
  - Metrics: `rmse_champion`, `rmse_candidate`, `rmse_increase_pct`
  - Artifacts: `decision.json`, `checks/nannyml_soft.json`
  - Tags: `retrain_recommended`, `promotion_recommended`, `integrity_warn`
  - Model registry: `green_taxi_tip_model` → version 2 with `@champion` alias
- **All 3 demo runs** (replay output from Colab or re-run and record screen)
- **Inference demo**: show `predictions.parquet` preview (Cell 14 output)

### 2. Push to GitHub *(mandatory)*
The repo `arielshtangel/OU-22971-MLOps` already exists. Push the latest files (including `capstone_colab.py`):
```bash
git add Tools/MLOps/8/capstone/
git add Tools/MLOps/environment.yml
git commit -m "capstone: add colab runner and fix injection target"
git push
```
Include the repo link in your submission.

### 3. Download MLflow DB from Colab for submission reference
Run Cell 17 in Colab to download `mlflow.db` — keep it for the video and submission.

### 4. Optional stretch goals (from Design Doc)
- **Stretch A**: Auto-trigger via folder-watching polling script + cron
- **Stretch B**: Giskard model vulnerability scan as extra promotion gate
- **Stretch C**: Containerized deployment to cloud

---

## 🚀 How to Re-Run on Google Colab

### Step 1 — Open a new Colab session and run Cell 1 (install packages)

### Step 2 — Upload files (Cell 3)
Upload these 4 files from your Windows machine:
- `C:\...\Tools\MLOps\8\capstone\capstone_flow.py`
- `C:\...\Tools\MLOps\8\capstone\capstone_lib.py`
- `C:\...\Tools\MLOps\8\capstone\inference.py`
- `C:\...\Tools\MLOps\6\green_taxi_drift_lib.py`

### Step 3 — Download TLC data (Cell 4)
Cells auto-download from the NYC TLC CDN — no manual download needed.

### Step 4 — Start MLflow server (Cell 5)
Starts as background process at `http://127.0.0.1:5000`.

### Step 5 — Expose MLflow UI (Cell 6, optional)
Requires a free ngrok token from https://ngrok.com — paste it into Cell 6.

### Step 6 — Copy drift lib (Cell 6b)
Copies `green_taxi_drift_lib.py` into `/content/capstone/` so Metaflow subprocesses can find it.

### Step 7–12 — Run Demos 1, 2, and 3
Run cells 7–12 in order. See inline comments for expected evidence per step.

### Step 13–14 — Batch inference and verification
Cell 13 runs inference; Cell 14 previews `predictions.parquet`.

### Step 15–16 — MLflow programmatic inspection
Cells 15–16 print run summary and model registry state without needing the UI.

### Step 17 — Download MLflow DB
Downloads `mlflow.db` to your local machine for submission reference.

---

## 📁 File Reference

```
Tools/MLOps/
├── environment.yml              ← Updated: metaflow + nannyml added
├── 8/
│   ├── design_doc.md            ← Project specification (source of truth)
│   ├── solution_steps.md        ← 19-step execution plan
│   └── capstone/
│       ├── capstone_flow.py     ← Main Metaflow workflow (Steps A→G)
│       ├── capstone_lib.py      ← Shared utilities
│       ├── inference.py         ← Offline batch inference
│       ├── capstone_colab.py    ← Google Colab runner (17 cells, copy into notebook)
│       ├── README.md            ← Setup + run guide
│       ├── summary.md           ← This file
│       ├── .gitignore
│       └── data/                ← TLC parquet files (gitignored; auto-downloaded in Colab)
```

