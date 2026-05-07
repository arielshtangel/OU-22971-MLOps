# Capstone Solution Steps
## Manual MLOps Workflow: NYC Green Taxi Tip Prediction
> **Design Doc:** `8/design_doc.md`
> **Implementation root:** `Tools/MLOps/8/capstone/`
> **Conda environment:** `22971-mlflow`

---

## Step 1 — Create project folder structure
**What:** Create `8/capstone/` with sub-folders and placeholder files.
**Why:** Establishes the execution root for the Metaflow flow and all supporting scripts.
**Ref:** Design Doc §8 Deliverables

```
8/capstone/
  data/               # Raw TLC parquet files (gitignored)
  capstone_lib.py     # Shared utilities (adapts Unit 6 green_taxi_drift_lib.py)
  capstone_flow.py    # Metaflow FlowSpec — main workflow
  inference.py        # Offline batch inference script
  README.md           # Submission README
```

---

## Step 2 — Update environment.yml with new dependencies
**What:** Add `metaflow` and `nannyml` to the pip section of `environment.yml`.
**Why:** Flow orchestration (Metaflow) and drift detection (NannyML) are required by the Design Doc.
**Ref:** Design Doc §Goal; Appendix: Metaflow primer

Update `Tools/MLOps/environment.yml` and run:
```bash
conda env update -f environment.yml --prune
```

---

## Step 3 — Acquire TLC Green Taxi data
**What:** Download ≥3 monthly parquet files from TLC Trip Record Data into `capstone/data/`.
**Why:** Need reference + at least 2 batch months to demonstrate the 3 required demo runs.
**Ref:** Design Doc §Problem description

| Role            | File                               | Demo Run |
|-----------------|------------------------------------|----------|
| Reference       | green_tripdata_2023-01.parquet     | All runs |
| Batch (no retrain) | green_tripdata_2023-04.parquet  | Run 1    |
| Batch (retrain) | green_tripdata_2023-10.parquet     | Runs 2 & 3 |

Add `capstone/data/` to `.gitignore`.

---

## Step 4 — Build `capstone_lib.py`
**What:** Shared utilities module that re-exports Unit 6 helpers and adds capstone-specific functions.
**Why:** Single source of truth for feature engineering, integrity checks, and decision logging.
**Ref:** Design Doc §Step B, §Step C, §Step D, §Anti-footgun rules

Functions to implement:
- `rmse(y_true, y_pred)` — RMSE metric helper
- `write_decision_json(path, action, criteria, metrics, decision_reason)` — serialize decision artifact
- `run_hard_integrity_checks(df) -> (passed, reasons)` — Layer 1 fail-fast rules
- `run_nannyml_soft_checks(ref_df, batch_df) -> (warn, report)` — Layer 2 NannyML + unseen categoricals
- `get_champion_version(client, model_name)` — safe alias lookup
- `ensure_registered_model(client, model_name)` — idempotent model creation

Re-export from Unit 6 `green_taxi_drift_lib.py`:
- `load_taxi_table`, `make_tip_frame`, `align_feature_frame`
- `cast_ints_to_float`, `resolve_input_path`, `run_integrity_checks`
- `EXPECTED_SCHEMA`, `RANGE_SPECS`, `RAW_DATETIME_COLS`

---

## Step 5 — Implement `capstone_flow.py` — flow skeleton + Parameters
**What:** Create `CapstoneFlow(FlowSpec)` with all Parameters and stubbed @step methods.
**Why:** Establishes the control graph before adding step logic.
**Ref:** Design Doc §Appendix: Metaflow primer; Metaflow starter

Parameters:
- `reference-path` (required)
- `batch-path` (required)
- `model-name` (default: `green_taxi_tip_model`)
- `tracking-uri` (default: `http://localhost:5000`)
- `min-improvement` (default: `0.01`) — 1% threshold, Ref §P2
- `retrain-rmse-threshold` (default: `0.10`) — 10% RMSE degradation triggers retrain
- `experiment-name` (default: `8_capstone`)

Validate flow graph: `python capstone_flow.py show`

---

## Step 6 — Implement Step A: `start` + `load_data`
**What:** Create the MLflow run and load reference + batch datasets.
**Why:** Steps A of the Design Doc workflow.
**Ref:** Design Doc §Step A

- `start`: `mlflow.start_run()` → store `self.mlflow_run_id` → log flow parameters via `MlflowClient`
- `load_data`: `load_taxi_table(ref)`, `load_taxi_table(batch)` → store as Metaflow artifacts
  `self.ref_df`, `self.batch_df`, `self.batch_id`

---

## Step 7 — Implement Step B: `integrity_gate`
**What:** Two-layer integrity check on the raw batch.
**Why:** Step B of the Design Doc — catch schema/pipeline issues before feature engineering.
**Ref:** Design Doc §Step B; §Anti-footgun rules

**Layer 1 (hard rules — fail-fast):**
- Required columns present
- No `trip_distance < 0` (> 5% threshold)
- No inverted datetimes (> 5% threshold)
- No critical missingness (> 50% threshold)
- On failure: log reason, write `decision.json` with `action="reject_batch"`, `self.next(self.end)`

**Layer 2 (NannyML soft checks):**
- Missingness spike vs reference (`SummaryStatsNullValuesCalculator` or fallback)
- Unseen categoricals in `payment_type`, `RatecodeID`, `trip_type`
- On warning: set `integrity_warn=true` tag, log `checks/nannyml_soft.json` — do NOT stop flow

---

## Step 8 — Implement Step C: `feature_engineering`
**What:** Apply identical feature transformation pipeline to both reference and batch.
**Why:** Step C — stable schema for training, evaluation and inference.
**Ref:** Design Doc §Feature engineering; §Step C

- Call `make_tip_frame(ref_df, credit_card_only=True)` → `X_ref, y_ref, feature_cols`
- Call `make_tip_frame(batch_df, ...)` + `align_feature_frame(X_batch, feature_cols)`
- Store `self.X_ref`, `self.y_ref`, `self.X_batch`, `self.y_batch`, `self.feature_cols`
- Log `feature_spec.json` (column names + dtypes) to MLflow run

---

## Step 9 — Implement Step D: `load_champion`
**What:** Load @champion from registry, or bootstrap if none exists.
**Why:** Step D — must always have a champion model before evaluation.
**Ref:** Design Doc §Step D; §Bootstrap (no champion exists yet)

**Normal path:** `mlflow.pyfunc.load_model("models:/green_taxi_tip_model@champion")`

**Bootstrap path** (champion alias absent):
1. Train `Pipeline(SimpleImputer + DecisionTreeRegressor)` on `X_ref`
2. Register under `green_taxi_tip_model`, `await_registration_for=300`
3. Set alias `@champion`, tags: `role=champion`, `promotion_reason=bootstrap`

Store `self.champion_version`, `self.champion_uri`, `self.rmse_baseline`.

---

## Step 10 — Implement Step E: `model_gate`
**What:** Evaluate champion on batch; decide whether to retrain.
**Why:** Step E — performance gate before conditional retrain.
**Ref:** Design Doc §Step E

- `rmse_champion = rmse(y_batch, champion.predict(X_batch))`
- `rmse_increase_pct = (rmse_champion - rmse_baseline) / rmse_baseline`
- `retrain_needed = rmse_increase_pct > retrain_rmse_threshold`
- Log metrics to MLflow: `rmse_champion`, `rmse_baseline`, `rmse_increase_pct`
- Set tag: `retrain_recommended=true/false`
- Write `decision.json` to `model_gate/` artifact path
- Branch: `self.next(self.retrain if retrain_needed else self.candidate_acceptance)`

---

## Step 11 — Implement Step F: `retrain`
**What:** Train a candidate model on reference + batch combined; register it.
**Why:** Step F — conditional retrain triggered by performance degradation.
**Ref:** Design Doc §Step F; §Model registry logic — registration mechanics

- Build rolling training set: `pd.concat([X_ref, X_batch])`
- Hyperparameter tuning: Optuna (15 trials, `TPESampler`) for `max_depth` and `min_samples_leaf`
- Train `Pipeline(SimpleImputer + DecisionTreeRegressor)` with best params
- Evaluate on **same batch as champion** (fair comparison): `rmse_candidate`
- Register under `green_taxi_tip_model`, tags: `role=candidate`, `trained_on_batches`, `eval_batch_id`,
  `validation_status=pending`
- Log `rmse_candidate` and `rmse_champion` metrics to both the training run and main capstone run

---

## Step 12 — Implement Step G: `candidate_acceptance`
**What:** Apply promotion gates P1–P4; flip @champion alias if all pass.
**Why:** Step G — promote only a meaningful, stable, safe candidate.
**Ref:** Design Doc §Step G; §Promotion criteria P1–P4; §Promotion mechanics; §Anti-footgun rules

**If no candidate** (retrain_needed=False): write `decision.json` with `action="no_action"`, exit.

**Gates:**
- **P1**: `rmse_candidate` is finite and eval dataset exists
- **P2**: `rmse_candidate < rmse_champion * (1 - min_improvement)` (1% threshold)
- **P3**: Stability — candidate does not regress reference slice by > 10%
- **P4**: `integrity_passed=True` (no hard failures)

**On promote:** flip `@champion` alias, tag old as `previous_champion`, tag new as `champion`.
**On reject:** tag candidate `validation_status=rejected`.
**Always:** write `decision.json`, log `promotion_recommended=true/false` tag.

---

## Step 13 — Implement `end` step
**What:** Finalize MLflow run status.
**Why:** Clean closure of the run.
**Ref:** Design Doc §Anti-footgun rules

- Call `client.set_terminated(run_id, "FINISHED")` (or "FAILED" if integrity rejected)
- Print summary of run_id and key outcomes

---

## Step 14 — Implement `inference.py`
**What:** Standalone offline batch inference script.
**Why:** Required as part of deliverables (Inference demo, Unit 6 style).
**Ref:** Design Doc §8 Deliverables — "Inference demo (offline)"

- Load `@champion` via `mlflow.pyfunc.load_model("models:/green_taxi_tip_model@champion")`
- Apply `make_tip_frame` + `align_feature_frame`
- Run `model.predict(X)`
- Save predictions as `predictions.parquet`
- Log `predictions.parquet` as MLflow artifact

Usage: `python inference.py --batch-path data/... --tracking-uri http://localhost:5000`

---

## Step 15 — Prepare Demo Run 1: Baseline (no retrain, no promotion)
**What:** Run the flow on a batch month with acceptable champion performance.
**Why:** Required demo run 1.
**Ref:** Design Doc §Required demo pattern — Run 1

Use a month close to reference OR raise `--retrain-rmse-threshold 0.99`.
Verify in MLflow UI: `retrain_recommended=false`, `promotion_recommended=false`, `decision.json`.

---

## Step 16 — Prepare Demo Run 2: Retrain + Promotion
**What:** Run the flow with a degraded batch to trigger automatic retrain and champion promotion.
**Why:** Required demo run 2.
**Ref:** Design Doc §Required demo pattern — Run 2

Use a month with distribution shift OR lower `--retrain-rmse-threshold 0.001`.
Verify in MLflow UI: new model version, `@champion` alias updated, `decision.json` with `action="promote"`.

---

## Step 17 — Prepare Demo Run 3: Failure + Resume
**What:** Inject a deliberate exception in `retrain`, fail the flow, fix, then resume from `retrain`.
**Why:** Required demo run 3 — demonstrates Metaflow checkpoint/resume.
**Ref:** Design Doc §Required demo pattern — Run 3; §Appendix: Metaflow resume

```bash
# After injecting: raise RuntimeError("injected failure")
python capstone_flow.py run --reference-path ... --batch-path ...   # fails at retrain
# Fix the exception, then:
python capstone_flow.py resume retrain
```

Verify: steps A–E not re-executed, retrain + candidate_acceptance complete successfully.

---

## Step 18 — Write `README.md`
**What:** Document setup, run commands, and MLflow UI navigation.
**Why:** Required deliverable.
**Ref:** Design Doc §8 Deliverables

Sections:
1. Prerequisites & environment setup
2. Start MLflow server
3. Download TLC data
4. Run the flow (all 3 demo scenarios)
5. Run inference
6. MLflow UI guide (experiment name, key metrics, decision.json, registry)

---

## Step 19 — Independent verification
**What:** Re-read Design Doc and verify every requirement against implementation.
**Why:** Phase 5 — verification sub-agent check.
**Ref:** Full Design Doc

Produce checklist report of ✅ / ⚠️ / ❌ items.

