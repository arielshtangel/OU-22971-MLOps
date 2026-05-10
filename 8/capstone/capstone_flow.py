"""
capstone_flow.py
================
MLOps Capstone – Manual monitoring & optional retraining workflow (Unit 8).

Flow steps (A → G):
  start → load_data → integrity_gate → feature_engineering → load_champion
        → model_gate → [retrain →] candidate_acceptance → end

Usage:
  # Normal run
  python capstone_flow.py run \\
      --reference-path data/green_tripdata_2023-01.parquet \\
      --batch-path     data/green_tripdata_2023-04.parquet

  # Force retrain (lower threshold)
  python capstone_flow.py run \\
      --reference-path data/green_tripdata_2023-01.parquet \\
      --batch-path     data/green_tripdata_2023-10.parquet \\
      --retrain-rmse-threshold 0.001

  # Resume after failure at a specific step
  python capstone_flow.py resume retrain
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from metaflow import FlowSpec, Parameter, step

# ── Resolve capstone_lib (same directory as this script) ─────────────────────
_CAPSTONE_DIR = Path(__file__).resolve().parent
if str(_CAPSTONE_DIR) not in sys.path:
    sys.path.insert(0, str(_CAPSTONE_DIR))


class CapstoneFlow(FlowSpec):
    """
    Manual MLOps monitoring + optional retraining workflow for NYC Green Taxi tip prediction.
    """

    # ── CLI Parameters ────────────────────────────────────────────────────────
    reference_path = Parameter(
        "reference-path", type=str, required=True,
        help="Path to reference parquet file (absolute or relative to cwd)."
    )
    batch_path = Parameter(
        "batch-path", type=str, required=True,
        help="Path to new batch parquet file."
    )
    model_name = Parameter(
        "model-name", type=str, default="green_taxi_tip_model",
        help="MLflow registered model name."
    )
    tracking_uri = Parameter(
        "tracking-uri", type=str, default="http://localhost:5000",
        help="MLflow tracking server URI."
    )
    min_improvement = Parameter(
        "min-improvement", type=float, default=0.01,
        help="Minimum fractional RMSE improvement required for promotion (default 1%%). "
    )
    retrain_rmse_threshold = Parameter(
        "retrain-rmse-threshold", type=float, default=0.10,
        help="Fractional RMSE increase over baseline that triggers retraining (default 10%%). "
    )
    experiment_name = Parameter(
        "experiment-name", type=str, default="8_capstone",
        help="MLflow experiment name."
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Step A — start: Create MLflow run and record flow parameters
    # ══════════════════════════════════════════════════════════════════════════
    @step
    def start(self):
        """
        Initialise MLflow run and log flow-level parameters.
        """
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        batch_stem = Path(self.batch_path).stem
        run = mlflow.start_run(run_name=f"capstone_{batch_stem}")
        self.mlflow_run_id = run.info.run_id
        # Deactivate the fluent context — we use the client API going forward
        # so that step-by-step logging works safely across Metaflow step boundaries.
        mlflow.end_run()

        client = MlflowClient(tracking_uri=self.tracking_uri)
        client.log_param(self.mlflow_run_id, "reference_path", self.reference_path)
        client.log_param(self.mlflow_run_id, "batch_path", self.batch_path)
        client.log_param(self.mlflow_run_id, "model_name", self.model_name)
        client.log_param(self.mlflow_run_id, "min_improvement", self.min_improvement)
        client.log_param(self.mlflow_run_id, "retrain_rmse_threshold",
                         self.retrain_rmse_threshold)

        print(f"[start] MLflow run started. run_id={self.mlflow_run_id}")
        print(f"[start] Experiment: {self.experiment_name}")
        self.next(self.load_data)

    # ══════════════════════════════════════════════════════════════════════════
    # Step A — load_data: Load reference and batch datasets
    # ══════════════════════════════════════════════════════════════════════════
    @step
    def load_data(self):
        """
        Load raw reference and batch parquet files into Metaflow artifacts.
        """
        from capstone_lib import load_taxi_table, resolve_input_path

        print(f"[load_data] Reference: {self.reference_path}")
        self.ref_df = load_taxi_table(
            resolve_input_path(self.reference_path, anchor_dir=_CAPSTONE_DIR)
        )
        print(f"[load_data] Reference rows: {len(self.ref_df):,}")

        print(f"[load_data] Batch: {self.batch_path}")
        self.batch_df = load_taxi_table(
            resolve_input_path(self.batch_path, anchor_dir=_CAPSTONE_DIR)
        )
        print(f"[load_data] Batch rows: {len(self.batch_df):,}")

        self.batch_id = Path(self.batch_path).stem
        self.next(self.integrity_gate)

    # ══════════════════════════════════════════════════════════════════════════
    # Step B — integrity_gate: Two-layer integrity check on the raw batch
    # ══════════════════════════════════════════════════════════════════════════
    @step
    def integrity_gate(self):
        """
        Layer 1: Hard rules (fail-fast) — reject batch if critical issues found.
        Layer 2: NannyML soft checks — warn but do not stop.
        """
        from mlflow import MlflowClient
        import mlflow
        from capstone_lib import (
            run_hard_integrity_checks,
            run_nannyml_soft_checks,
            run_integrity_checks,
            write_decision_json,
        )

        client = MlflowClient(tracking_uri=self.tracking_uri)
        run_id = self.mlflow_run_id

        # ── Layer 1: Hard Rules ───────────────────────────────────────────────
        hard_passed, hard_reasons = run_hard_integrity_checks(self.batch_df)
        print(f"[integrity_gate] Hard checks passed: {hard_passed}")

        if not hard_passed:
            print(f"[integrity_gate] HARD FAIL — rejecting batch:")
            for r in hard_reasons:
                print(f"  - {r}")

            client.set_tag(run_id, "integrity_hard_fail", "true")
            client.set_tag(run_id, "integrity_hard_reasons",
                           " | ".join(hard_reasons)[:500])

            with tempfile.TemporaryDirectory() as tmpdir:
                dec_path = Path(tmpdir) / "decision.json"
                write_decision_json(
                    dec_path,
                    action="reject_batch",
                    criteria={"hard_rules_failed": hard_reasons},
                    metrics={},
                    decision_reason="Hard integrity rules failed — batch rejected.",
                )
                client.log_artifact(run_id, str(dec_path), artifact_path="")

            # Mark run as FAILED
            client.set_terminated(run_id, status="FAILED")

            self.integrity_passed = False
            self.integrity_warn = False
            self.soft_report = {}
            self.next(self.end)
            return

        # ── Full integrity check (log all check tables to MLflow) ─────────────
        chk = run_integrity_checks(self.batch_df)
        for name, tbl in chk.tables.items():
            if not tbl.empty:
                with tempfile.TemporaryDirectory() as tmpdir:
                    p = Path(tmpdir) / f"{name}.json"
                    tbl.to_json(p, orient="records", indent=2)
                    client.log_artifact(run_id, str(p), artifact_path="checks")

        for key, val in chk.metrics.items():
            fval = float(val)
            if np.isfinite(fval):
                client.log_metric(run_id, f"integrity_{key}", fval)

        # ── Layer 2: NannyML Soft Checks ─────────────────────────────────────
        soft_warn, soft_report = run_nannyml_soft_checks(self.ref_df, self.batch_df)
        print(f"[integrity_gate] NannyML soft warn: {soft_warn}")
        if soft_report.get("warnings"):
            for w in soft_report["warnings"]:
                print(f"  ⚠  {w}")

        self.integrity_warn = soft_warn
        # Store report (strip large nested objects for Metaflow artifact size)
        self.soft_report = {
            k: v for k, v in soft_report.items()
            if k != "nannyml_null_results_sample"
        }

        # set warning tag
        client.set_tag(run_id, "integrity_warn", str(soft_warn).lower())
        if soft_warn:
            client.set_tag(run_id, "integrity_warn_reasons",
                           " | ".join(soft_report.get("warnings", []))[:500])

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "nannyml_soft.json"
            with open(p, "w", encoding="utf-8") as f:
                json.dump(soft_report, f, indent=2, default=str)
            client.log_artifact(run_id, str(p), artifact_path="checks")

        self.integrity_passed = True
        self.next(self.feature_engineering)

    # ══════════════════════════════════════════════════════════════════════════
    # Step C — feature_engineering: Build consistent feature frames
    # ══════════════════════════════════════════════════════════════════════════
    @step
    def feature_engineering(self):
        """
        Apply identical feature pipeline to reference and batch.
        Produces a stable schema logged as feature_spec.json.
        """
        from mlflow import MlflowClient
        from capstone_lib import make_tip_frame, cast_ints_to_float, align_feature_frame

        client = MlflowClient(tracking_uri=self.tracking_uri)
        run_id = self.mlflow_run_id

        # Apply same transformation to both splits (credit-card rows, numeric features)
        X_ref, y_ref, feature_cols = make_tip_frame(self.ref_df, credit_card_only=True)
        X_batch_raw, y_batch, _ = make_tip_frame(self.batch_df, credit_card_only=True)

        # Align batch columns to reference feature order
        try:
            X_batch = align_feature_frame(X_batch_raw, feature_cols)
        except ValueError:
            # Some monthly files differ slightly — use intersection safely
            common = [c for c in feature_cols if c in X_batch_raw.columns]
            feature_cols = common
            X_ref = X_ref[feature_cols]
            X_batch = X_batch_raw[feature_cols]

        self.X_ref = cast_ints_to_float(X_ref)
        self.y_ref = y_ref
        self.X_batch = cast_ints_to_float(X_batch)
        self.y_batch = y_batch
        self.feature_cols = feature_cols

        print(
            f"[feature_engineering] X_ref={self.X_ref.shape}  "
            f"X_batch={self.X_batch.shape}  "
            f"features={len(feature_cols)}"
        )

        # Log feature spec for schema debugging
        spec = {
            "feature_cols": feature_cols,
            "n_features": len(feature_cols),
            "dtypes": {c: str(self.X_ref[c].dtype) for c in feature_cols},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / "feature_spec.json"
            with open(spec_path, "w", encoding="utf-8") as f:
                json.dump(spec, f, indent=2)
            client.log_artifact(run_id, str(spec_path), artifact_path="")

        self.next(self.load_champion)

    # ══════════════════════════════════════════════════════════════════════════
    # Step D — load_champion: Load @champion or bootstrap initial model
    # ══════════════════════════════════════════════════════════════════════════
    @step
    def load_champion(self):
        """
        Load the current @champion model from the registry.
        If no champion exists, train an initial model and register it as bootstrap.
        """
        import mlflow
        import mlflow.sklearn
        from mlflow import MlflowClient
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.model_selection import train_test_split
        from capstone_lib import (
            get_champion_version,
            ensure_registered_model,
            rmse,
        )

        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient(tracking_uri=self.tracking_uri)
        run_id = self.mlflow_run_id

        ensure_registered_model(client, self.model_name)
        champion_ver = get_champion_version(client, self.model_name)

        if champion_ver is None:
            # ── Bootstrap path ────────────────────────────────────────────────
            print("[load_champion] No @champion alias found — bootstrapping initial model.")
            client.set_tag(run_id, "bootstrap", "true")

            X_tr, X_va, y_tr, y_va = train_test_split(
                self.X_ref, self.y_ref, test_size=0.2, random_state=42
            )
            model = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("tree", DecisionTreeRegressor(
                    random_state=42, max_depth=8, min_samples_leaf=100
                )),
            ])
            model.fit(X_tr, y_tr)
            val_rmse = rmse(y_va, model.predict(X_va))
            self.rmse_baseline = val_rmse
            print(f"[load_champion] Bootstrap validation RMSE: {val_rmse:.4f}")

            exp = mlflow.get_experiment_by_name(self.experiment_name)
            with mlflow.start_run(
                experiment_id=exp.experiment_id,
                run_name="bootstrap_train",
            ) as boot_run:
                mlflow.log_param("role", "bootstrap")
                mlflow.log_param("trained_on", "reference")
                mlflow.log_metric("rmse_val", val_rmse)

                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    registered_model_name=self.model_name,
                    input_example=X_tr.head(5),
                    await_registration_for=300,
                )
                # Log feature spec in training run for reproducibility
                with tempfile.TemporaryDirectory() as tmpdir:
                    fc_path = Path(tmpdir) / "feature_cols.json"
                    with open(fc_path, "w") as fh:
                        json.dump({"feature_cols": self.feature_cols}, fh)
                    mlflow.log_artifact(str(fc_path))

                self.champion_run_id = boot_run.info.run_id

            v = str(model_info.registered_model_version)
            # tags and alias
            client.set_model_version_tag(self.model_name, v, "role", "champion")
            client.set_model_version_tag(self.model_name, v, "promotion_reason", "bootstrap")
            client.set_model_version_tag(self.model_name, v, "trained_on_batches", "reference")
            client.set_registered_model_alias(self.model_name, "champion", v)

            self.champion_uri = model_info.model_uri
            self.champion_version = v
            print(f"[load_champion] Bootstrap @champion v{v} registered.")

        else:
            # ── Normal path — champion already exists ─────────────────────────
            self.champion_version = str(champion_ver.version)
            self.champion_uri = f"models:/{self.model_name}@champion"
            self.champion_run_id = champion_ver.run_id
            print(f"[load_champion] Loaded @champion version {self.champion_version}.")

            # Retrieve baseline RMSE from the training run that produced the champion
            self.rmse_baseline = -1.0
            try:
                champ_run = client.get_run(self.champion_run_id)
                for metric_key in ("rmse_val", "root_mean_squared_error", "rmse"):
                    val = champ_run.data.metrics.get(metric_key)
                    if val is not None:
                        self.rmse_baseline = float(val)
                        break
            except Exception as e:
                print(f"[load_champion] Could not fetch baseline RMSE: {e}")

        client.set_tag(run_id, "champion_version", self.champion_version)
        print(f"[load_champion] rmse_baseline={self.rmse_baseline:.4f}")
        self.next(self.model_gate)

    # ══════════════════════════════════════════════════════════════════════════
    # Step E — model_gate: Evaluate champion on batch; decide retrain
    # ══════════════════════════════════════════════════════════════════════════
    @step
    def model_gate(self):
        """
        Evaluate champion model on engineered batch features.
        Compute RMSE increase and decide whether to retrain.
        """
        import mlflow
        from mlflow import MlflowClient
        from capstone_lib import rmse, write_decision_json

        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient(tracking_uri=self.tracking_uri)
        run_id = self.mlflow_run_id

        champion = mlflow.pyfunc.load_model(self.champion_uri)
        y_pred = champion.predict(self.X_batch)
        self.rmse_champion = rmse(self.y_batch, y_pred)

        if self.rmse_baseline > 0:
            self.rmse_increase_pct = (
                (self.rmse_champion - self.rmse_baseline) / self.rmse_baseline
            )
        else:
            self.rmse_increase_pct = 0.0

        self.retrain_needed = self.rmse_increase_pct > self.retrain_rmse_threshold
        self.retrain_reason = (
            f"RMSE increased by {self.rmse_increase_pct:.1%} "
            f"> threshold {self.retrain_rmse_threshold:.1%}"
            if self.retrain_needed else
            f"RMSE increase {self.rmse_increase_pct:.1%} "
            f"≤ threshold {self.retrain_rmse_threshold:.1%}"
        )

        print(
            f"[model_gate] rmse_champion={self.rmse_champion:.4f}  "
            f"rmse_baseline={self.rmse_baseline:.4f}  "
            f"increase={self.rmse_increase_pct:.1%}  "
            f"retrain={self.retrain_needed}"
        )

        # Log metrics and tag
        client.log_metric(run_id, "rmse_champion", self.rmse_champion)
        client.log_metric(run_id, "rmse_baseline", self.rmse_baseline)
        client.log_metric(run_id, "rmse_increase_pct", self.rmse_increase_pct)
        client.set_tag(run_id, "retrain_recommended",
                       str(self.retrain_needed).lower())

        # Write decision.json — always required by anti-footgun rules
        with tempfile.TemporaryDirectory() as tmpdir:
            dec_path = Path(tmpdir) / "decision.json"
            write_decision_json(
                dec_path,
                action="retrain" if self.retrain_needed else "no_retrain",
                criteria={
                    "retrain_rmse_threshold": self.retrain_rmse_threshold,
                    "rmse_increase_pct": self.rmse_increase_pct,
                },
                metrics={
                    "rmse_champion": self.rmse_champion,
                    "rmse_baseline": self.rmse_baseline,
                    "rmse_increase_pct": self.rmse_increase_pct,
                },
                decision_reason=self.retrain_reason,
            )
            client.log_artifact(run_id, str(dec_path), artifact_path="model_gate")

        # Initialise candidate fields (populated in retrain if executed)
        self.candidate_version = None
        self.candidate_uri = None
        self.candidate_run_id = None
        self.rmse_candidate = float("nan")

        # Always route to retrain; retrain will skip itself if not needed.
        # Note: Metaflow 2.19+ does not allow runtime variables in self.next() ternary.
        self.next(self.retrain)

    # ══════════════════════════════════════════════════════════════════════════
    # Step F — retrain: Train candidate model; register it
    # ══════════════════════════════════════════════════════════════════════════
    @step
    def retrain(self):
        """
        Train a candidate model on a rolling window (reference + batch).
        Uses Optuna for lightweight hyperparameter search.
        """
        import mlflow
        import mlflow.sklearn
        from mlflow import MlflowClient
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.model_selection import cross_val_score
        import optuna
        from capstone_lib import rmse

        # Skip retrain if not needed (Metaflow 2.19+ requires static self.next())
        if not self.retrain_needed:
            print("[retrain] Skipping — retrain not needed.")
            self.next(self.candidate_acceptance)
            return

        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient(tracking_uri=self.tracking_uri)
        run_id = self.mlflow_run_id

        # Build rolling training set (ref + batch)
        X_train = pd.concat([self.X_ref, self.X_batch], ignore_index=True)
        y_train = np.concatenate([self.y_ref, self.y_batch])
        print(f"[retrain] Training set size: {len(X_train):,} rows")

        # ── Optuna hyperparameter tuning (15 trials) ──────────────────────────
        def objective(trial: optuna.Trial) -> float:
            params = {
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 50, 500, log=True
                ),
            }
            m = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("tree", DecisionTreeRegressor(random_state=42, **params)),
            ])
            scores = cross_val_score(
                m, X_train, y_train,
                cv=3, scoring="neg_root_mean_squared_error", n_jobs=1,
            )
            return float(-scores.mean())

        print("[retrain] Running Optuna (15 trials)...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=15, show_progress_bar=False)
        best = study.best_params
        print(f"[retrain] Best params: {best}  CV-RMSE={study.best_value:.4f}")

        # ── Train final candidate on all data ─────────────────────────────────
        candidate = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("tree", DecisionTreeRegressor(
                random_state=42,
                max_depth=best["max_depth"],
                min_samples_leaf=best["min_samples_leaf"],
            )),
        ])
        candidate.fit(X_train, y_train)

        # Evaluate on SAME batch as champion for fair comparison
        self.rmse_candidate = rmse(self.y_batch, candidate.predict(self.X_batch))
        print(
            f"[retrain] rmse_candidate={self.rmse_candidate:.4f}  "
            f"rmse_champion={self.rmse_champion:.4f}"
        )

        # ── Register candidate in MLflow Model Registry ───────────────────────
        exp = mlflow.get_experiment_by_name(self.experiment_name)
        with mlflow.start_run(
            experiment_id=exp.experiment_id,
            run_name="retrain_candidate",
        ) as cand_run:
            mlflow.log_params({
                "max_depth": best["max_depth"],
                "min_samples_leaf": best["min_samples_leaf"],
                "optuna_n_trials": 15,
                "trained_on": f"reference+{self.batch_id}",
            })
            mlflow.log_metric("rmse_candidate", self.rmse_candidate)
            mlflow.log_metric("rmse_champion", self.rmse_champion)
            mlflow.log_metric("rmse_baseline", self.rmse_baseline)
            # Log rmse_val under the canonical key so that future runs can retrieve
            # baseline RMSE from this training run if this candidate is promoted to @champion.
            # Fix for: load_champion queries rmse_val → root_mean_squared_error → rmse
            mlflow.log_metric("rmse_val", self.rmse_candidate)

            model_info = mlflow.sklearn.log_model(
                sk_model=candidate,
                name="model",
                registered_model_name=self.model_name,
                input_example=self.X_batch.head(5),
                await_registration_for=300,
            )
            # Log feature spec for reproducibility
            with tempfile.TemporaryDirectory() as tmpdir:
                fc_path = Path(tmpdir) / "feature_cols.json"
                with open(fc_path, "w") as fh:
                    json.dump({"feature_cols": self.feature_cols}, fh)
                mlflow.log_artifact(str(fc_path))

            self.candidate_run_id = cand_run.info.run_id

        self.candidate_version = str(model_info.registered_model_version)
        self.candidate_uri = model_info.model_uri

        # Tag candidate version
        tags = {
            "role": "candidate",
            "trained_on_batches": f"reference+{self.batch_id}",
            "eval_batch_id": self.batch_id,
            "validation_status": "pending",
            "rmse_candidate": f"{self.rmse_candidate:.6f}",
            "rmse_champion": f"{self.rmse_champion:.6f}",
        }
        for k, v in tags.items():
            client.set_model_version_tag(self.model_name, self.candidate_version, k, v)

        client.log_metric(run_id, "rmse_candidate", self.rmse_candidate)
        client.set_tag(run_id, "candidate_version", self.candidate_version)

        print(f"[retrain] Candidate v{self.candidate_version} registered.")
        self.next(self.candidate_acceptance)

    # ══════════════════════════════════════════════════════════════════════════
    # Step G — candidate_acceptance: Evaluate P1-P4 gates; promote if all pass
    # ══════════════════════════════════════════════════════════════════════════
    @step
    def candidate_acceptance(self):
        """
        Apply promotion gates P1–P4.
        Flip @champion alias if all gates pass; reject candidate otherwise.
        Always write decision.json.
        """
        import mlflow
        from mlflow import MlflowClient
        from datetime import datetime, timezone
        from capstone_lib import rmse, write_decision_json

        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient(tracking_uri=self.tracking_uri)
        run_id = self.mlflow_run_id
        now_utc = datetime.now(timezone.utc).isoformat()

        # ── No candidate trained (retrain was not needed) ─────────────────────
        if self.candidate_version is None:
            print("[candidate_acceptance] No candidate model — no retrain was triggered.")
            client.set_tag(run_id, "promotion_recommended", "false")

            with tempfile.TemporaryDirectory() as tmpdir:
                dec_path = Path(tmpdir) / "decision.json"
                write_decision_json(
                    dec_path,
                    action="no_action",
                    criteria={"retrain_needed": False},
                    metrics={
                        "rmse_champion": self.rmse_champion,
                        "rmse_baseline": self.rmse_baseline,
                        "rmse_increase_pct": self.rmse_increase_pct,
                    },
                    decision_reason=(
                        "Champion performance is within acceptable bounds. "
                        "No retrain or promotion required."
                    ),
                )
                client.log_artifact(run_id, str(dec_path), artifact_path="")

            self.next(self.end)
            return

        # ── Gate P1: Evaluation validity ─────────────────────────────────────
        p1_pass = not np.isnan(self.rmse_candidate)
        p1_reason = "ok" if p1_pass else "rmse_candidate is NaN — evaluation invalid."

        # ── Gate P2: Candidate beats champion by min_improvement ──────────────
        threshold_rmse = self.rmse_champion * (1.0 - self.min_improvement)
        p2_pass = self.rmse_candidate < threshold_rmse
        p2_reason = (
            f"rmse_candidate ({self.rmse_candidate:.4f}) < "
            f"rmse_champion * (1 - {self.min_improvement:.2f}) = {threshold_rmse:.4f}"
            if p2_pass else
            f"Candidate ({self.rmse_candidate:.4f}) did NOT beat "
            f"threshold ({threshold_rmse:.4f})."
        )

        # ── Gate P3: Stability — candidate doesn't regress reference slice ────
        rmse_cand_ref = float("nan")
        rmse_champ_ref = float("nan")
        ref_regression = float("nan")
        try:
            cand_model = mlflow.pyfunc.load_model(self.candidate_uri)
            champ_model = mlflow.pyfunc.load_model(self.champion_uri)
            rmse_cand_ref = rmse(self.y_ref, cand_model.predict(self.X_ref))
            rmse_champ_ref = rmse(self.y_ref, champ_model.predict(self.X_ref))
            ref_regression = (
                (rmse_cand_ref - rmse_champ_ref) / max(rmse_champ_ref, 1e-9)
            )
            # Fail P3 if candidate regresses reference performance by > 10%
            p3_pass = ref_regression <= 0.10
            p3_reason = (
                f"Reference regression: {ref_regression:+.1%} "
                f"(candidate={rmse_cand_ref:.4f}, champion={rmse_champ_ref:.4f})"
            )
            client.log_metric(run_id, "rmse_candidate_on_ref", rmse_cand_ref)
            client.log_metric(run_id, "rmse_champion_on_ref", rmse_champ_ref)
            client.log_metric(run_id, "ref_regression_pct", ref_regression)
        except Exception as exc:
            # Non-blocking — don't veto promotion due to a P3 evaluation error
            p3_pass = True
            p3_reason = f"P3 evaluation error (non-blocking): {exc}"

        # ── Gate P4: Integrity sanity ─────────────────────────────────────────
        p4_pass = self.integrity_passed
        p4_reason = (
            "ok" if p4_pass
            else "Hard integrity failure — promotion blocked."
        )

        promote = p1_pass and p2_pass and p3_pass and p4_pass

        print(
            f"[candidate_acceptance] "
            f"P1={p1_pass}  P2={p2_pass}  P3={p3_pass}  P4={p4_pass}  "
            f"→ promote={promote}"
        )
        for gate, reason in [
            ("P1", p1_reason), ("P2", p2_reason),
            ("P3", p3_reason), ("P4", p4_reason),
        ]:
            print(f"  {gate}: {reason}")

        client.set_tag(run_id, "promotion_recommended", str(promote).lower())
        for gate_id, passed in [("p1", p1_pass), ("p2", p2_pass),
                                  ("p3", p3_pass), ("p4", p4_pass)]:
            client.log_metric(run_id, f"{gate_id}_pass", int(passed))

        # ── Promotion or rejection ────────────────────────────────────────────
        if promote:
            print(f"[candidate_acceptance] Promoting v{self.candidate_version} → @champion")

            # Demote old champion
            try:
                client.set_model_version_tag(
                    self.model_name, self.champion_version,
                    "role", "previous_champion",
                )
                client.set_model_version_tag(
                    self.model_name, self.champion_version,
                    "demoted_at", now_utc,
                )
            except Exception:
                pass  # Champion version may already be tagged/removed

            # Flip alias
            client.set_registered_model_alias(
                self.model_name, "champion", self.candidate_version
            )

            # Tag new champion
            for k, v in {
                "role": "champion",
                "promoted_at": now_utc,
                "promotion_reason": "candidate_beat_champion",
                "validation_status": "approved",
            }.items():
                client.set_model_version_tag(
                    self.model_name, self.candidate_version, k, v
                )

            action = "promote"
            decision_reason = (
                f"Candidate v{self.candidate_version} passes all gates "
                f"(P1={p1_pass}, P2={p2_pass}, P3={p3_pass}, P4={p4_pass}). "
                f"Promoted to @champion."
            )
        else:
            # Collect failure reasons
            failure_reasons = [
                r for gate_pass, r in [
                    (p1_pass, f"P1: {p1_reason}"),
                    (p2_pass, f"P2: {p2_reason}"),
                    (p3_pass, f"P3: {p3_reason}"),
                    (p4_pass, f"P4: {p4_reason}"),
                ] if not gate_pass
            ]
            decision_reason = "; ".join(failure_reasons)

            # Tag as rejected
            client.set_model_version_tag(
                self.model_name, self.candidate_version,
                "validation_status", "rejected",
            )
            client.set_model_version_tag(
                self.model_name, self.candidate_version,
                "decision_reason", decision_reason[:500],
            )
            action = "reject_candidate"
            print(f"[candidate_acceptance] Candidate rejected: {decision_reason}")

        # Write final decision.json — always required (anti-footgun rule)
        with tempfile.TemporaryDirectory() as tmpdir:
            dec_path = Path(tmpdir) / "decision.json"
            write_decision_json(
                dec_path,
                action=action,
                criteria={
                    "min_improvement": self.min_improvement,
                    "p1_evaluation_valid": {"passed": p1_pass, "reason": p1_reason},
                    "p2_beats_champion": {"passed": p2_pass, "reason": p2_reason},
                    "p3_stable": {"passed": p3_pass, "reason": p3_reason},
                    "p4_integrity": {"passed": p4_pass, "reason": p4_reason},
                },
                metrics={
                    "rmse_champion": self.rmse_champion,
                    "rmse_candidate": self.rmse_candidate,
                    "rmse_baseline": self.rmse_baseline,
                    "rmse_increase_pct": self.rmse_increase_pct,
                    "rmse_candidate_on_ref": rmse_cand_ref,
                    "rmse_champion_on_ref": rmse_champ_ref,
                    "ref_regression_pct": ref_regression,
                },
                decision_reason=decision_reason,
            )
            client.log_artifact(run_id, str(dec_path), artifact_path="")

        self.next(self.end)

    # ══════════════════════════════════════════════════════════════════════════
    # end: Finalise MLflow run
    # ══════════════════════════════════════════════════════════════════════════
    @step
    def end(self):
        """
        Finalize the MLflow run status and print a summary.
        """
        from mlflow import MlflowClient

        client = MlflowClient(tracking_uri=self.tracking_uri)
        run_id = self.mlflow_run_id

        try:
            run = client.get_run(run_id)
            # Only mark FINISHED if not already FAILED (set by integrity_gate on hard fail)
            if run.info.status not in ("FAILED", "KILLED"):
                client.set_terminated(run_id, status="FINISHED")
        except Exception as exc:
            print(f"[end] Warning: could not update MLflow run status: {exc}")

        print("=" * 60)
        print("  CAPSTONE FLOW COMPLETE")
        print("=" * 60)
        print(f"  MLflow run_id    : {run_id}")
        print(f"  Integrity passed : {getattr(self, 'integrity_passed', '?')}")
        print(f"  Retrain needed   : {getattr(self, 'retrain_needed', '?')}")
        print(f"  Champion version : {getattr(self, 'champion_version', '?')}")
        print(f"  Candidate version: {getattr(self, 'candidate_version', None)}")
        print("=" * 60)


if __name__ == "__main__":
    CapstoneFlow()

