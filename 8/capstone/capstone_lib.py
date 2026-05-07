"""
capstone_lib.py
===============
Shared utilities for the MLOps Capstone project (Unit 8).

Re-exports useful helpers from Unit 6 (green_taxi_drift_lib.py) and adds
capstone-specific functions for decision logging, integrity checks, and
model-registry management.

Ref: Design Doc §Step B, §Step C, §Step D, §Anti-footgun rules
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# ── Re-export Unit 6 utilities ────────────────────────────────────────────────
_LIB6 = Path(__file__).resolve().parent.parent.parent / "6"
if str(_LIB6) not in sys.path:
    sys.path.insert(0, str(_LIB6))

from green_taxi_drift_lib import (  # noqa: E402
    EXPECTED_SCHEMA,
    RANGE_SPECS,
    RAW_DATETIME_COLS,
    add_datetime_features,
    align_feature_frame,
    cast_ints_to_float,
    load_taxi_table,
    make_tip_frame,
    resolve_input_path,
    run_integrity_checks,
)

__all__ = [
    # Re-exported from Unit 6
    "EXPECTED_SCHEMA", "RANGE_SPECS", "RAW_DATETIME_COLS",
    "add_datetime_features", "align_feature_frame", "cast_ints_to_float",
    "load_taxi_table", "make_tip_frame", "resolve_input_path",
    "run_integrity_checks",
    # Capstone-specific
    "rmse",
    "write_decision_json",
    "run_hard_integrity_checks",
    "run_nannyml_soft_checks",
    "get_champion_version",
    "ensure_registered_model",
]


# ── Metrics ───────────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(
        np.asarray(y_true, dtype=float),
        np.asarray(y_pred, dtype=float),
    )))


# ── Decision logging ──────────────────────────────────────────────────────────

def write_decision_json(
    path: "Path | str",
    *,
    action: str,
    criteria: Dict[str, Any],
    metrics: Dict[str, Any],
    decision_reason: str = "",
) -> Path:
    """
    Write decision.json artifact.

    Ref: Design Doc §Anti-footgun rules —
    'Always log decision.json describing: criteria used, metric values, final decision.'
    """
    payload = {
        "action": action,
        "decision_reason": decision_reason,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "criteria": {k: _safe_json(v) for k, v in criteria.items()},
        "metrics": {k: _safe_json(v) for k, v in metrics.items()},
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return p


def _safe_json(val: Any) -> Any:
    """Convert numpy scalars and NaN/Inf to JSON-safe values."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        v = float(val)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(val, float):
        return None if (np.isnan(val) or np.isinf(val)) else val
    return val


# ── Layer 1: Hard integrity checks ───────────────────────────────────────────

def run_hard_integrity_checks(
    df: pd.DataFrame,
    *,
    required_cols: Optional[List[str]] = None,
    missing_frac_threshold: float = 0.50,
    neg_distance_threshold: float = 0.05,
    inverted_datetime_threshold: float = 0.05,
) -> Tuple[bool, List[str]]:
    """
    Fail-fast hard integrity rules on the raw batch.

    Returns (passed, reasons).  If passed=False the batch must be rejected.

    Ref: Design Doc §Step B - Layer 1: hard rules (fail-fast)
    """
    reasons: List[str] = []

    if required_cols is None:
        required_cols = [
            "trip_distance", "PULocationID", "DOLocationID",
            "lpep_pickup_datetime", "lpep_dropoff_datetime", "tip_amount",
        ]

    # 1. Required columns present
    missing_cols = sorted(set(required_cols) - set(df.columns))
    if missing_cols:
        reasons.append(f"Missing required columns: {missing_cols}")

    # 2. Empty batch
    if df.empty:
        reasons.append("Batch dataset is empty (0 rows).")
        return False, reasons

    # 3. Critical missingness (columns that exist)
    for col in ["trip_distance", "tip_amount", "PULocationID", "DOLocationID"]:
        if col in df.columns:
            frac = float(df[col].isna().mean())
            if frac > missing_frac_threshold:
                reasons.append(
                    f"Column '{col}' has {frac:.1%} missing values "
                    f"(threshold {missing_frac_threshold:.0%})."
                )

    # 4. Negative trip_distance
    if "trip_distance" in df.columns:
        neg_frac = float(
            (pd.to_numeric(df["trip_distance"], errors="coerce") < 0).mean()
        )
        if neg_frac > neg_distance_threshold:
            reasons.append(
                f"trip_distance: {neg_frac:.1%} negative values "
                f"(threshold {neg_distance_threshold:.0%})."
            )

    # 5. Inverted datetimes (dropoff before pickup)
    if all(c in df.columns for c in RAW_DATETIME_COLS):
        pickup = pd.to_datetime(df["lpep_pickup_datetime"], errors="coerce")
        dropoff = pd.to_datetime(df["lpep_dropoff_datetime"], errors="coerce")
        dur_sec = (dropoff - pickup).dt.total_seconds()
        inv_frac = float((dur_sec < 0).mean())
        if inv_frac > inverted_datetime_threshold:
            reasons.append(
                f"Inverted datetime (dropoff < pickup) in {inv_frac:.1%} of rows "
                f"(threshold {inverted_datetime_threshold:.0%})."
            )

    return len(reasons) == 0, reasons


# ── Layer 2: NannyML soft checks ──────────────────────────────────────────────

def run_nannyml_soft_checks(
    ref_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    *,
    missingness_spike_threshold: float = 0.10,
) -> Tuple[bool, Dict[str, Any]]:
    """
    NannyML-based soft integrity checks (do NOT stop on warnings).

    Returns (warn, report_dict).

    Ref: Design Doc §Step B - Layer 2: NannyML checks (soft gate)
    """
    report: Dict[str, Any] = {"checks": [], "warnings": [], "nannyml_status": "ok"}

    # ── Missingness spike via NannyML ─────────────────────────────────────────
    try:
        import nannyml as nml  # noqa: F401

        num_cols = [
            c for c in ref_df.columns
            if c in batch_df.columns
            and pd.api.types.is_numeric_dtype(ref_df[c])
            and c not in RAW_DATETIME_COLS
        ]

        if num_cols and len(ref_df) >= 100 and len(batch_df) >= 100:
            ref_work = ref_df[num_cols].copy()
            batch_work = batch_df[num_cols].copy()
            # Synthesize timestamps for NannyML chunking
            ref_work["_ts"] = pd.date_range(
                "2020-01-01", periods=len(ref_work), freq="1s"
            )
            batch_work["_ts"] = pd.date_range(
                "2021-01-01", periods=len(batch_work), freq="1s"
            )

            calc = nml.SummaryStatsNullValuesCalculator(
                column_names=num_cols,
                timestamp_column_name="_ts",
                chunk_size=max(100, len(batch_work)),
            )
            calc.fit(ref_work)
            nml_res = calc.calculate(batch_work)
            report["nannyml_null_results_sample"] = str(nml_res.to_df().head(5).to_dict())
            report["nannyml_status"] = "ok"
        else:
            report["nannyml_status"] = "skipped_insufficient_data"

    except ImportError:
        report["nannyml_status"] = "not_installed_using_fallback"

    except Exception as e:
        report["nannyml_status"] = f"error: {e}"

    # ── Missingness spike (manual — always run as ground truth) ──────────────
    for col in ref_df.columns:
        if col not in batch_df.columns or col in RAW_DATETIME_COLS:
            continue
        ref_null = float(ref_df[col].isna().mean())
        batch_null = float(batch_df[col].isna().mean())
        spike = batch_null - ref_null
        if spike > missingness_spike_threshold:
            msg = (
                f"Missingness spike in '{col}': "
                f"ref={ref_null:.2%} → batch={batch_null:.2%} (+{spike:.2%})"
            )
            report["warnings"].append(msg)
            report["checks"].append({
                "column": col, "check": "missingness_spike",
                "ref_null_frac": ref_null, "batch_null_frac": batch_null,
                "spike": spike, "warn": True,
            })
        else:
            report["checks"].append({
                "column": col, "check": "missingness_spike",
                "ref_null_frac": ref_null, "batch_null_frac": batch_null,
                "spike": spike, "warn": False,
            })

    # ── Unseen categoricals ───────────────────────────────────────────────────
    cat_cols = ["payment_type", "RatecodeID", "trip_type", "store_and_fwd_flag"]
    for col in cat_cols:
        if col not in ref_df.columns or col not in batch_df.columns:
            continue
        ref_vals = set(ref_df[col].dropna().unique())
        batch_vals = set(batch_df[col].dropna().unique())
        unseen = batch_vals - ref_vals
        if unseen:
            msg = (
                f"Unseen values in '{col}': "
                f"{sorted(str(v) for v in unseen)}"
            )
            report["warnings"].append(msg)
            report["checks"].append({
                "column": col, "check": "unseen_categoricals",
                "unseen_values": sorted(str(v) for v in unseen), "warn": True,
            })
        else:
            report["checks"].append({
                "column": col, "check": "unseen_categoricals",
                "unseen_values": [], "warn": False,
            })

    warn = len(report["warnings"]) > 0
    if warn:
        report["warning_count"] = len(report["warnings"])

    return warn, report


# ── Model Registry helpers ────────────────────────────────────────────────────

def ensure_registered_model(client: Any, model_name: str) -> None:
    """Create the registered model if it does not already exist."""
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name)


def get_champion_version(client: Any, model_name: str) -> Optional[Any]:
    """
    Return the @champion model version object, or None if the alias does not exist.

    Ref: Design Doc §Step D
    """
    try:
        return client.get_model_version_by_alias(model_name, "champion")
    except Exception:
        return None

