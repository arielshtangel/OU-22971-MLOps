# MLOps Capstone – NYC Green Taxi Tip Prediction
## Unit 8: Manual Monitoring → Optional Retraining → Champion Promotion

**Stack:** Metaflow · MLflow 3.8 · NannyML · Optuna · Scikit-learn (DecisionTreeRegressor)

---

## Overview

This project implements a **manually triggered MLOps workflow** for a tip-prediction regression model on NYC Green Taxi data. Each time a new monthly data batch arrives, you run the flow from the command line. The flow:

1. Checks batch integrity (hard rules + NannyML soft gate)
2. Applies consistent feature engineering
3. Evaluates the current champion model
4. Optionally retrains a candidate model
5. Promotes the candidate to `@champion` if it passes all quality gates
6. Logs every decision to MLflow

---

## Project Structure

```
8/capstone/
  capstone_flow.py    ← Metaflow FlowSpec (main workflow)
  capstone_lib.py     ← Shared utilities (integrity, features, registry helpers)
  inference.py        ← Offline batch inference script
  README.md           ← This file
  .gitignore
  data/               ← Local TLC parquet files (gitignored)
```

---

## Prerequisites

### 1. Conda environment

```bash
# From the Tools/MLOps directory:
conda env update -f environment.yml --prune
conda activate 22971-mlflow
```

### 2. MLflow tracking server

Open a **separate terminal** and run:

```bash
conda activate 22971-mlflow
mlflow server --host 127.0.0.1 --port 5000
```

Leave this running throughout all demo runs.

MLflow UI: [http://localhost:5000](http://localhost:5000)

### 3. Download TLC Green Taxi data

Download parquet files from:
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Place them in `8/capstone/data/`:

| File | Purpose |
|------|---------|
| `green_tripdata_2023-01.parquet` | Reference dataset (all runs) |
| `green_tripdata_2023-04.parquet` | Demo Run 1 (no retrain) |
| `green_tripdata_2023-10.parquet` | Demo Run 2 & 3 (retrain + promotion) |

---

## Running the Workflow

All commands below should be run from `8/capstone/` with the `22971-mlflow` environment active.

### Demo Run 1 — Baseline (no retrain, no promotion)

Champion performance is acceptable → flow completes with no model changes.

```bash
python capstone_flow.py run \
    --reference-path data/green_tripdata_2023-01.parquet \
    --batch-path     data/green_tripdata_2023-04.parquet \
    --retrain-rmse-threshold 0.99
```

**Verify in MLflow UI (`8_capstone` experiment):**
- Tag `retrain_recommended = false`
- Tag `promotion_recommended = false`
- Artifact `decision.json` with `action = "no_action"`

---

### Demo Run 2 — Retrain + Promotion (automatic within the flow)

Low threshold forces the flow to detect degradation and retrain.

```bash
python capstone_flow.py run \
    --reference-path data/green_tripdata_2023-01.parquet \
    --batch-path     data/green_tripdata_2023-10.parquet \
    --retrain-rmse-threshold 0.001
```

**Verify in MLflow UI:**
- Tag `retrain_recommended = true`
- Tag `promotion_recommended = true` (if candidate passes all P1–P4 gates)
- New model version in **Models → green_taxi_tip_model**
- `@champion` alias updated to the new version
- Artifact `decision.json` with `action = "promote"`

---

### Demo Run 3 — Failure + Resume (workflow robustness)

**Step 1:** Inject a deliberate failure in `retrain` (add one line at the top of the `retrain` step):

```python
# In capstone_flow.py, inside retrain():
raise RuntimeError("injected failure for demo run 3")
```

**Step 2:** Run the flow — it fails at the `retrain` step:

```bash
python capstone_flow.py run \
    --reference-path data/green_tripdata_2023-01.parquet \
    --batch-path     data/green_tripdata_2023-10.parquet \
    --retrain-rmse-threshold 0.001
```

**Step 3:** Remove the injected exception, then resume from the failed step:

```bash
python capstone_flow.py resume retrain
```

> **Note:** If you have multiple prior failed runs, Metaflow picks the most recent one.
> To be explicit, pass the run ID: `python capstone_flow.py resume retrain --origin-run-id <N>`
> (the run ID is printed when the flow starts, e.g. `CapstoneFlow/42`)

**Verify:**
- Steps `start → load_data → integrity_gate → feature_engineering → load_champion → model_gate`
  are **NOT re-executed** (Metaflow restores step artifacts)
- `retrain` and `candidate_acceptance` run fresh
- MLflow shows the completed run with all artifacts

---

## Batch Inference

Run predictions against the current `@champion` model:

```bash
python inference.py \
    --batch-path data/green_tripdata_2024-01.parquet \
    --tracking-uri http://localhost:5000
```

Output: `predictions.parquet` saved locally and logged as an MLflow artifact in the `8_capstone_inference` experiment.

---

## MLflow UI Guide

| What to look at | Where |
|----------------|-------|
| All flow runs | Experiments → `8_capstone` |
| Inference runs | Experiments → `8_capstone_inference` |
| Key metrics | Run → Metrics: `rmse_champion`, `rmse_candidate`, `rmse_increase_pct` |
| Decision artifact | Run → Artifacts → `decision.json` |
| Integrity report | Run → Artifacts → `checks/nannyml_soft.json` |
| Decision tags | Run → Tags: `retrain_recommended`, `promotion_recommended`, `integrity_warn` |
| Model registry | Models → `green_taxi_tip_model` |
| Champion alias | Models → `green_taxi_tip_model` → Aliases → `@champion` |

---

## Flow Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--reference-path` | *(required)* | Path to reference parquet |
| `--batch-path` | *(required)* | Path to new batch parquet |
| `--model-name` | `green_taxi_tip_model` | MLflow registered model name |
| `--tracking-uri` | `http://localhost:5000` | MLflow server URI |
| `--min-improvement` | `0.01` | Min RMSE improvement for promotion (1%) |
| `--retrain-rmse-threshold` | `0.10` | RMSE increase that triggers retrain (10%) |
| `--experiment-name` | `8_capstone` | MLflow experiment name |

---

## Design Document Reference

Full project specification: [`8/design_doc.md`](../design_doc.md)
