"""
inference.py
============
Offline batch inference script for the MLOps Capstone project (Unit 8).

Loads the current @champion model from the MLflow Model Registry,
applies the same feature engineering pipeline used during training,
runs predictions, and logs the output predictions.parquet as an MLflow artifact.

Usage:
  python inference.py \\
      --batch-path data/green_tripdata_2024-01.parquet \\
      --tracking-uri http://localhost:5000
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

# ── Resolve capstone_lib ──────────────────────────────────────────────────────
_CAPSTONE_DIR = Path(__file__).resolve().parent
if str(_CAPSTONE_DIR) not in sys.path:
    sys.path.insert(0, str(_CAPSTONE_DIR))

from capstone_lib import (
    load_taxi_table,
    make_tip_frame,
    align_feature_frame,
    cast_ints_to_float,
    resolve_input_path,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run batch inference using the @champion model from MLflow registry."
    )
    p.add_argument(
        "--batch-path", type=str, required=True,
        help="Path to the batch parquet file for inference.",
    )
    p.add_argument(
        "--tracking-uri", type=str, default="http://localhost:5000",
        help="MLflow tracking server URI.",
    )
    p.add_argument(
        "--model-name", type=str, default="green_taxi_tip_model",
        help="Registered model name in MLflow.",
    )
    p.add_argument(
        "--experiment-name", type=str, default="8_capstone_inference",
        help="MLflow experiment name for logging the inference run.",
    )
    p.add_argument(
        "--output-path", type=str, default=None,
        help="Local path to save predictions.parquet (default: cwd/predictions.parquet).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import mlflow
    import mlflow.pyfunc
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    client = MlflowClient(tracking_uri=args.tracking_uri)

    batch_path = resolve_input_path(args.batch_path, anchor_dir=_CAPSTONE_DIR)
    output_path = (
        Path(args.output_path) if args.output_path
        else Path.cwd() / "predictions.parquet"
    )

    with mlflow.start_run(run_name=f"inference_{batch_path.stem}") as run:
        run_id = run.info.run_id
        mlflow.log_param("batch_path", str(batch_path))
        mlflow.log_param("model_name", args.model_name)
        mlflow.log_param("model_alias", "champion")

        # ── Step 1: Load @champion model ──────────────────────────────────────
        model_uri = f"models:/{args.model_name}@champion"
        print(f"[inference] Loading model: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)

        # Retrieve champion version for lineage
        try:
            champ_ver = client.get_model_version_by_alias(args.model_name, "champion")
            mlflow.log_param("champion_version", champ_ver.version)
        except Exception:
            pass

        # ── Step 2: Load and preprocess batch ────────────────────────────────
        print(f"[inference] Loading batch: {batch_path}")
        df_raw = load_taxi_table(batch_path)
        print(f"[inference] Batch rows: {len(df_raw):,}")
        mlflow.log_metric("batch_rows_raw", len(df_raw))

        # Feature engineering (same pipeline as training)
        X, y, feature_cols = make_tip_frame(df_raw, credit_card_only=True)
        X = cast_ints_to_float(X)

        # Try to align feature columns to the order expected by the model.
        # The champion training run stores feature_cols.json — download and align if available.
        try:
            from mlflow import MlflowClient as _MlflowClient
            import json as _json
            _client = _MlflowClient(tracking_uri=args.tracking_uri)
            _champ_ver = _client.get_model_version_by_alias(args.model_name, "champion")
            _art_path = mlflow.artifacts.download_artifacts(
                run_id=_champ_ver.run_id, artifact_path="feature_cols.json"
            )
            with open(_art_path, "r") as _fh:
                _spec = _json.load(_fh)
            _train_cols = _spec.get("feature_cols", [])
            if _train_cols:
                X = align_feature_frame(X, _train_cols)
                print(f"[inference] Aligned to {len(_train_cols)} training columns.")
        except Exception as _e:
            print(f"[inference] Column alignment skipped ({_e}); using columns as-is.")

        print(f"[inference] Feature rows after CC filter: {len(X):,}")
        mlflow.log_metric("batch_rows_cc", len(X))

        # ── Step 3: Predict ───────────────────────────────────────────────────
        print("[inference] Running predictions...")
        predictions = model.predict(X)

        # ── Step 4: Build output DataFrame ───────────────────────────────────
        # Keep original index for join-back to the raw batch
        pred_df = X.copy()
        pred_df["predicted_tip_amount"] = predictions
        if y is not None and len(y) == len(pred_df):
            pred_df["actual_tip_amount"] = y
            import numpy as np
            from sklearn.metrics import mean_squared_error
            rmse_val = float(
                np.sqrt(mean_squared_error(y, predictions))
            )
            mlflow.log_metric("rmse_inference", rmse_val)
            print(f"[inference] RMSE on batch: {rmse_val:.4f}")

        # ── Step 5: Save and log predictions.parquet ─────────────────────────
        pred_df.to_parquet(output_path, index=False)
        print(f"[inference] Predictions saved to: {output_path}")

        mlflow.log_artifact(str(output_path), artifact_path="")
        mlflow.log_metric("n_predictions", len(pred_df))

        print(f"[inference] Run complete. run_id={run_id}")
        print(f"[inference] Predictions logged as MLflow artifact: predictions.parquet")

    print("\nDone. Open MLflow UI to inspect the inference run.")


if __name__ == "__main__":
    main()

