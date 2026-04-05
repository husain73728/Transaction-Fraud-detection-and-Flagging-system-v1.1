"""Lightweight wrapper around the provided fraud models for web inference.

This module loads the two pickled models from `backend/` once and exposes
helpers to score an uploaded CSV plus produce a compact JSON summary that
the frontend can render.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import uuid

import joblib
import numpy as np
import pandas as pd
from pandas.api import types as ptypes

BASE_DIR = Path(__file__).resolve().parent
PAYSIM_MODEL_PATH = BASE_DIR / "paysim_model.pkl"
KARTIK_MODEL_PATH = BASE_DIR / "kartik_model.pkl"

# Lazily load to avoid repeated disk hits if the module is re-imported.
_paysim_model = joblib.load(PAYSIM_MODEL_PATH)
_kartik_model = joblib.load(KARTIK_MODEL_PATH)

# =========================================================
# FEATURE LISTS USED DURING TRAINING
# =========================================================
PAYSIM_FEATURES = [
    "step",
    "type",
    "amount",
    "oldbalanceorg",
    "newbalanceorig",
    "oldbalancedest",
    "newbalancedest",
    "hour",
    "is_night",
    "amount_log",
    "orig_balance_diff",
    "dest_balance_diff",
    "orig_zero",
    "dest_zero",
]

KARTIK_FEATURES = [
    "merchant",
    "category",
    "amt",
    "gender",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "unix_time",
    "merch_lat",
    "merch_long",
    "hour",
    "day",
    "month",
    "is_weekend",
    "is_night",
    "age",
    "amt_log",
    "distance_km",
]


# =========================================================
# HELPERS
# =========================================================
def haversine_vectorized(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def ensure_numeric(df: pd.DataFrame, cols: List[str]):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def prepare_paysim(df: pd.DataFrame) -> pd.DataFrame | None:
    temp = df.copy()
    temp.columns = temp.columns.str.strip().str.lower()

    needed_raw = [
        "step",
        "type",
        "amount",
        "oldbalanceorg",
        "newbalanceorig",
        "oldbalancedest",
        "newbalancedest",
    ]
    if not all(col in temp.columns for col in needed_raw):
        return None

    temp = ensure_numeric(
        temp,
        [
            "step",
            "amount",
            "oldbalanceorg",
            "newbalanceorig",
            "oldbalancedest",
            "newbalancedest",
        ],
    )

    temp = temp.dropna(subset=needed_raw)
    temp = temp[temp["amount"] >= 0]

    # encode type if it is not already numeric (handles object + pandas string dtypes)
    if not ptypes.is_numeric_dtype(temp["type"]):
        temp["type"] = temp["type"].astype("category").cat.codes

    temp["hour"] = temp["step"] % 24
    temp["is_night"] = ((temp["hour"] >= 23) | (temp["hour"] <= 5)).astype(int)
    temp["amount_log"] = np.log1p(temp["amount"])
    temp["orig_balance_diff"] = temp["newbalanceorig"] - temp["oldbalanceorg"]
    temp["dest_balance_diff"] = temp["newbalancedest"] - temp["oldbalancedest"]
    temp["orig_zero"] = (temp["oldbalanceorg"] == 0).astype(int)
    temp["dest_zero"] = (temp["oldbalancedest"] == 0).astype(int)

    for col in PAYSIM_FEATURES:
        if col not in temp.columns:
            temp[col] = 0

    return temp[PAYSIM_FEATURES].copy()


def prepare_kartik(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp.columns = temp.columns.str.strip().str.lower()

    # if timestamp exists, derive time features
    if "trans_date_trans_time" in temp.columns:
        temp["trans_date_trans_time"] = pd.to_datetime(
            temp["trans_date_trans_time"], errors="coerce"
        )
        temp["hour"] = temp["trans_date_trans_time"].dt.hour
        temp["day"] = temp["trans_date_trans_time"].dt.weekday
        temp["month"] = temp["trans_date_trans_time"].dt.month
        temp["is_weekend"] = (temp["day"] >= 5).astype(int)
        temp["is_night"] = ((temp["hour"] >= 23) | (temp["hour"] <= 5)).astype(int)

    # age from dob if possible
    if "dob" in temp.columns and "trans_date_trans_time" in temp.columns:
        temp["dob"] = pd.to_datetime(temp["dob"], errors="coerce")
        age = ((temp["trans_date_trans_time"] - temp["dob"]).dt.days / 365.25)
        temp["age"] = age.clip(lower=18, upper=100)

    # amount log
    if "amt" in temp.columns:
        temp["amt"] = pd.to_numeric(temp["amt"], errors="coerce")
        temp["amt_log"] = np.log1p(temp["amt"])

    # distance
    if all(col in temp.columns for col in ["lat", "long", "merch_lat", "merch_long"]):
        temp = ensure_numeric(temp, ["lat", "long", "merch_lat", "merch_long"])
        temp["distance_km"] = haversine_vectorized(
            temp["lat"], temp["long"], temp["merch_lat"], temp["merch_long"]
        )

    # crude encoding for object columns
    # note: for a serious deployment, save training encoders and reuse them
    for col in temp.columns:
        if not ptypes.is_numeric_dtype(temp[col]):
            temp[col] = temp[col].fillna("unknown").astype("category").cat.codes

    temp = ensure_numeric(temp, KARTIK_FEATURES)

    # fill missing feature columns
    for col in KARTIK_FEATURES:
        if col not in temp.columns:
            temp[col] = 0

    # fill numeric missing values
    for col in KARTIK_FEATURES:
        temp[col] = temp[col].fillna(0)

    return temp[KARTIK_FEATURES].copy()


def score_dataframe(
    df: pd.DataFrame,
    paysim_weight: float = 0.3,
    kartik_weight: float = 0.7,
    final_threshold: float = 0.5,
) -> pd.DataFrame:
    """Return scored dataframe with probability columns and labels."""
    result = df.copy()
    paysim_prob = None
    kartik_prob = None

    # -------------------------
    # PAYSIM SCORE
    # -------------------------
    X_paysim = prepare_paysim(df)
    if X_paysim is not None and len(X_paysim) > 0:
        paysim_prob = _paysim_model.predict_proba(X_paysim)[:, 1]
        result.loc[X_paysim.index, "paysim_prob"] = paysim_prob
    else:
        result["paysim_prob"] = np.nan

    # -------------------------
    # KARTIK SCORE
    # -------------------------
    X_kartik = prepare_kartik(df)
    if X_kartik is not None and len(X_kartik) > 0:
        kartik_prob = _kartik_model.predict_proba(X_kartik)[:, 1]
        result.loc[X_kartik.index, "kartik_prob"] = kartik_prob
    else:
        result["kartik_prob"] = np.nan

    # -------------------------
    # FINAL COMBINED SCORE
    # if one model is unavailable, use the other
    # -------------------------
    def combine_scores(row):
        p = row["paysim_prob"]
        k = row["kartik_prob"]

        if pd.notna(p) and pd.notna(k):
            return paysim_weight * p + kartik_weight * k
        elif pd.notna(p):
            return p
        elif pd.notna(k):
            return k
        else:
            return np.nan

    result["final_score"] = result.apply(combine_scores, axis=1)
    result["risk_percent"] = result["final_score"] * 100
    result["fraud_flag"] = (result["final_score"] >= final_threshold).astype("Int64")

    def label_risk(score):
        if pd.isna(score):
            return "UNKNOWN"
        if score >= 0.8:
            return "HIGH RISK"
        elif score >= 0.5:
            return "MEDIUM RISK"
        else:
            return "LOW RISK"

    result["risk_label"] = result["final_score"].apply(label_risk)
    return result


@dataclass
class RunSummary:
    job_id: str
    file_name: str
    processed_at: str
    rows: int
    high_risk: int
    medium_risk: int
    low_risk: int
    unknown: int
    flagged: int
    avg_risk_percent: float
    max_risk_percent: float
    download_path: str
    top_cases: List[Dict[str, Any]]

    def to_json(self) -> Dict[str, Any]:
        payload = asdict(self)
        # ensure native types for JSON serialization
        for key, value in payload.items():
            if isinstance(value, (np.generic, pd.Timestamp)):
                payload[key] = value.item()
        return payload


def _pick_sample_columns(df: pd.DataFrame) -> List[str]:
    'Pick a small set of columns to surface in the UI table.'
    preferred = [
        "amount",
        "amt",
        "type",
        "category",
        "merchant",
        "hour",
        "distance_km",
    ]

    cols = [c for c in preferred if c in df.columns]

    # If preferred fields are missing (e.g., generic customer CSV),
    # surface the first few non-risk source columns so the table isn't empty.
    skip = {"paysim_prob", "kartik_prob", "final_score", "risk_percent", "fraud_flag", "risk_label"}
    for col in df.columns:
        if col in cols or col.lower() in skip:
            continue
        cols.append(col)
        if len(cols) >= 5:
            break

    for tail in ["risk_percent", "fraud_flag", "risk_label"]:
        if tail in df.columns and tail not in cols:
            cols.append(tail)

    return cols


def summarize_results(
    scored_df: pd.DataFrame, job_id: str, file_name: str, download_path: str
) -> RunSummary:
    counts = scored_df["risk_label"].value_counts(dropna=False)
    high = int(counts.get("HIGH RISK", 0))
    med = int(counts.get("MEDIUM RISK", 0))
    low = int(counts.get("LOW RISK", 0))
    unknown = int(counts.get("UNKNOWN", 0))
    flagged = int(scored_df["fraud_flag"].fillna(0).sum())
    avg_risk = float(scored_df["risk_percent"].fillna(0).mean())
    max_risk = float(scored_df["risk_percent"].fillna(0).max())

    sample_cols = _pick_sample_columns(scored_df)
    top_cases_df = (
        scored_df.sort_values("final_score", ascending=False)[sample_cols]
        .head(8)
        .reset_index(drop=True)
    )
    top_cases = top_cases_df.to_dict(orient="records")

    return RunSummary(
        job_id=job_id,
        file_name=file_name,
        processed_at=datetime.utcnow().isoformat() + "Z",
        rows=int(len(scored_df)),
        high_risk=high,
        medium_risk=med,
        low_risk=low,
        unknown=unknown,
        flagged=flagged,
        avg_risk_percent=round(avg_risk, 2),
        max_risk_percent=round(max_risk, 2),
        download_path=download_path,
        top_cases=top_cases,
    )


def score_csv_file(
    csv_path: Path,
    output_path: Path,
    paysim_weight: float = 0.3,
    kartik_weight: float = 0.7,
    final_threshold: float = 0.5,
) -> Tuple[pd.DataFrame, RunSummary]:
    """Score a CSV on disk and persist the predictions CSV and summary."""
    df = pd.read_csv(csv_path)
    scored = score_dataframe(
        df,
        paysim_weight=paysim_weight,
        kartik_weight=kartik_weight,
        final_threshold=final_threshold,
    )

    # if both models failed to produce any scores, abort early
    if scored["final_score"].isna().all():
        raise ValueError(
            "CSV did not contain any columns the models recognize. "
            "Include PaySim-style or Kartik-style transaction columns."
        )

    scored.to_csv(output_path, index=False)

    job_id = output_path.stem
    summary = summarize_results(
        scored, job_id=job_id, file_name=csv_path.name, download_path=str(output_path)
    )
    return scored, summary


def generate_job_id() -> str:
    return uuid.uuid4().hex[:12]
