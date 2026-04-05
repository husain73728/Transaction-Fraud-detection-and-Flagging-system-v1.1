"""Pure-Python runtime scoring helpers for Vercel deployment.

This module avoids heavy ML runtime dependencies by loading exported random
forest structures from compressed JSON and evaluating them directly.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parent
PAYSIM_MODEL_PATH = BASE_DIR / "paysim_model.json.gz"
KARTIK_MODEL_PATH = BASE_DIR / "kartik_model.json.gz"

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

PAYSIM_TYPE_MAP = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4,
}

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

KARTIK_RAW_COLUMNS = {
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
    "trans_date_trans_time",
    "dob",
}


def _load_forest(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


_paysim_model = _load_forest(PAYSIM_MODEL_PATH)
_kartik_model = _load_forest(KARTIK_MODEL_PATH)


def _read_csv_rows(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _write_csv_rows(output_path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _safe_int(value: Any) -> Optional[int]:
    number = _safe_float(value)
    if number is None:
        return None
    return int(number)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator in (0, 0.0):
        return 0.0
    return numerator / denominator


def _normalize_row(raw_row: Dict[str, Any]) -> Dict[str, Any]:
    return {str(key).strip().lower(): value for key, value in raw_row.items()}


def _normalize_type_label(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def safe_display_name(file_name: str, job_id: str) -> str:
    suffix = Path(file_name).suffix.lower() or ".csv"
    return f"upload_{job_id}{suffix}"


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    candidates = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y",
    ]
    for fmt in candidates:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    )
    return 6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _tree_probability(tree: Dict[str, Any], feature_row: List[float]) -> float:
    node = 0
    children_left = tree["children_left"]
    children_right = tree["children_right"]
    features = tree["feature"]
    thresholds = tree["threshold"]
    values = tree["values"]

    while children_left[node] != -1:
        feature_index = features[node]
        threshold = thresholds[node]
        feature_value = feature_row[feature_index]
        if feature_value <= threshold:
            node = children_left[node]
        else:
            node = children_right[node]

    leaf_values = values[node]
    total = sum(leaf_values)
    if total <= 0:
        return 0.0
    return float(leaf_values[1]) / float(total)


def predict_forest_proba(model: Dict[str, Any], feature_rows: List[List[float]]) -> List[float]:
    trees = model["trees"]
    if not trees:
        return [0.0 for _ in feature_rows]

    predictions: List[float] = []
    for feature_row in feature_rows:
        probability_sum = 0.0
        for tree in trees:
            probability_sum += _tree_probability(tree, feature_row)
        predictions.append(probability_sum / len(trees))
    return predictions


def generate_pay_sim_reasons(raw_row: Dict[str, Any], derived: Dict[str, float]) -> List[str]:
    tx_type = _normalize_type_label(raw_row.get("type"))
    amount = float(raw_row.get("amount", 0) or 0)
    old_org = float(raw_row.get("oldbalanceOrg", raw_row.get("oldbalanceorg", 0)) or 0)
    new_org = float(raw_row.get("newbalanceOrig", raw_row.get("newbalanceorig", 0)) or 0)
    old_dest = float(raw_row.get("oldbalanceDest", raw_row.get("oldbalancedest", 0)) or 0)
    new_dest = float(raw_row.get("newbalanceDest", raw_row.get("newbalancedest", 0)) or 0)
    hour = derived.get("hour")

    reasons: List[str] = []
    drain_ratio = 1 - _safe_ratio(new_org, old_org) if old_org > 0 else 0.0
    sender_drop = old_org - new_org
    dest_gain = new_dest - old_dest
    exact_sender_match = abs(sender_drop - amount) <= max(1.0, amount * 0.01)
    exact_dest_match = abs(dest_gain - amount) <= max(1.0, amount * 0.01)

    if tx_type in {"TRANSFER", "CASH_OUT"}:
        reasons.append("high-risk transaction type")
    if amount >= 500000:
        reasons.append("unusually high transaction amount")
    elif amount >= 250000:
        reasons.append("elevated transaction amount")
    if drain_ratio >= 0.98:
        reasons.append("account nearly drained")
    elif drain_ratio >= 0.9:
        reasons.append("sender balance heavily reduced")
    if tx_type == "TRANSFER" and old_dest == 0 and new_dest == 0:
        reasons.append("suspicious destination balance pattern")
    elif tx_type == "CASH_OUT" and exact_dest_match:
        reasons.append("destination balance rose almost exactly with the withdrawal")
    if exact_sender_match and amount >= 250000:
        reasons.append("sender balance drop matches the transaction amount")
    if hour is not None and int(hour) in {23, 0, 1, 2, 3, 4, 5}:
        reasons.append("night-time transaction")

    return list(dict.fromkeys(reasons))


def paysim_reason_score(raw_row: Dict[str, Any], reasons: List[str]) -> float:
    tx_type = _normalize_type_label(raw_row.get("type"))
    amount = float(raw_row.get("amount", 0) or 0)
    old_org = float(raw_row.get("oldbalanceOrg", raw_row.get("oldbalanceorg", 0)) or 0)
    new_org = float(raw_row.get("newbalanceOrig", raw_row.get("newbalanceorig", 0)) or 0)
    drain_ratio = 1 - _safe_ratio(new_org, old_org) if old_org > 0 else 0.0

    score = 0.0
    if "high-risk transaction type" in reasons:
        score += 0.08
    if "unusually high transaction amount" in reasons:
        score += 0.22
    elif "elevated transaction amount" in reasons:
        score += 0.12
    if "account nearly drained" in reasons:
        score += 0.22
    elif "sender balance heavily reduced" in reasons:
        score += 0.12
    if "suspicious destination balance pattern" in reasons:
        score += 0.28
    if "destination balance rose almost exactly with the withdrawal" in reasons:
        score += 0.18
    if "sender balance drop matches the transaction amount" in reasons:
        score += 0.12
    if "night-time transaction" in reasons:
        score += 0.05

    if tx_type == "TRANSFER" and amount >= 750000 and drain_ratio >= 0.98:
        score += 0.08
    if tx_type == "CASH_OUT" and amount >= 750000 and drain_ratio >= 0.98:
        score += 0.05

    return min(score, 0.95)


def assign_risk_label(
    *,
    score: Optional[float],
    effective_threshold: float,
    raw_row: Dict[str, Any],
    reasons: List[str],
    reason_score: float,
    paysim_available: bool,
    kartik_available: bool,
) -> str:
    if score is None:
        return "UNKNOWN"

    strong_reasons = len(reasons)
    tx_type = _normalize_type_label(raw_row.get("type"))
    amount = float(raw_row.get("amount", 0) or 0)

    if score >= 0.75:
        return "HIGH RISK"
    if paysim_available and not kartik_available:
        if score >= 0.35 and reason_score >= 0.55 and strong_reasons >= 3:
            return "HIGH RISK"
        if tx_type == "TRANSFER" and score >= 0.25 and reason_score >= 0.7:
            return "HIGH RISK"
        if score >= effective_threshold:
            return "MEDIUM RISK"
        if reason_score >= 0.45 and strong_reasons >= 3 and amount >= 500000:
            return "MEDIUM RISK"
        return "LOW RISK"

    if score >= max(0.5, effective_threshold):
        return "MEDIUM RISK"
    return "LOW RISK"


def prepare_paysim_rows(raw_rows: List[Dict[str, Any]]) -> Tuple[List[int], List[List[float]], Dict[int, Dict[str, float]]]:
    row_indexes: List[int] = []
    feature_rows: List[List[float]] = []
    derived_by_index: Dict[int, Dict[str, float]] = {}

    for index, raw_row in enumerate(raw_rows):
        row = _normalize_row(raw_row)
        needed = [
            row.get("step"),
            row.get("type"),
            row.get("amount"),
            row.get("oldbalanceorg"),
            row.get("newbalanceorig"),
            row.get("oldbalancedest"),
            row.get("newbalancedest"),
        ]
        if any(value is None or str(value).strip() == "" for value in needed):
            continue

        step = _safe_int(row.get("step"))
        amount = _safe_float(row.get("amount"))
        oldbalanceorg = _safe_float(row.get("oldbalanceorg"))
        newbalanceorig = _safe_float(row.get("newbalanceorig"))
        oldbalancedest = _safe_float(row.get("oldbalancedest"))
        newbalancedest = _safe_float(row.get("newbalancedest"))
        tx_type = PAYSIM_TYPE_MAP.get(_normalize_type_label(row.get("type")), -1)

        if None in [step, amount, oldbalanceorg, newbalanceorig, oldbalancedest, newbalancedest]:
            continue
        if amount < 0:
            continue

        hour = step % 24
        is_night = 1 if (hour >= 23 or hour <= 5) else 0
        amount_log = math.log1p(amount)
        orig_balance_diff = newbalanceorig - oldbalanceorg
        dest_balance_diff = newbalancedest - oldbalancedest
        orig_zero = 1 if oldbalanceorg == 0 else 0
        dest_zero = 1 if oldbalancedest == 0 else 0

        feature_rows.append(
            [
                float(step),
                float(tx_type),
                float(amount),
                float(oldbalanceorg),
                float(newbalanceorig),
                float(oldbalancedest),
                float(newbalancedest),
                float(hour),
                float(is_night),
                float(amount_log),
                float(orig_balance_diff),
                float(dest_balance_diff),
                float(orig_zero),
                float(dest_zero),
            ]
        )
        row_indexes.append(index)
        derived_by_index[index] = {
            "hour": float(hour),
            "is_night": float(is_night),
        }

    return row_indexes, feature_rows, derived_by_index


def prepare_kartik_rows(raw_rows: List[Dict[str, Any]]) -> Tuple[List[int], List[List[float]]]:
    normalized_rows = [_normalize_row(row) for row in raw_rows]
    present_cols = set()
    for row in normalized_rows:
        present_cols.update({key for key, value in row.items() if str(value).strip()})
    if "amt" not in present_cols or len(KARTIK_RAW_COLUMNS.intersection(present_cols)) < 4:
        return [], []

    categorical_fields = [
        "merchant",
        "category",
        "gender",
        "city",
        "state",
        "job",
    ]
    categorical_maps: Dict[str, Dict[str, int]] = {}
    for field in categorical_fields:
        values = sorted(
            {
                str(row.get(field, "unknown")).strip() or "unknown"
                for row in normalized_rows
            }
        )
        categorical_maps[field] = {value: idx for idx, value in enumerate(values)}

    row_indexes: List[int] = []
    feature_rows: List[List[float]] = []
    for index, row in enumerate(normalized_rows):
        amt = _safe_float(row.get("amt"))
        if amt is None or amt < 0:
            continue

        trans_dt = _parse_datetime(row.get("trans_date_trans_time"))
        dob = _parse_datetime(row.get("dob"))
        hour = trans_dt.hour if trans_dt else 0
        day = trans_dt.weekday() if trans_dt else 0
        month = trans_dt.month if trans_dt else 0
        is_weekend = 1 if day >= 5 else 0
        is_night = 1 if (hour >= 23 or hour <= 5) else 0
        age = 0
        if trans_dt and dob:
            age = max(18, min(100, int((trans_dt - dob).days / 365.25)))

        lat = _safe_float(row.get("lat")) or 0.0
        lon = _safe_float(row.get("long")) or 0.0
        merch_lat = _safe_float(row.get("merch_lat")) or 0.0
        merch_lon = _safe_float(row.get("merch_long")) or 0.0
        distance_km = (
            haversine_distance_km(lat, lon, merch_lat, merch_lon)
            if any([lat, lon, merch_lat, merch_lon])
            else 0.0
        )

        feature_rows.append(
            [
                float(categorical_maps["merchant"].get(str(row.get("merchant", "unknown")).strip() or "unknown", 0)),
                float(categorical_maps["category"].get(str(row.get("category", "unknown")).strip() or "unknown", 0)),
                float(amt),
                float(categorical_maps["gender"].get(str(row.get("gender", "unknown")).strip() or "unknown", 0)),
                float(categorical_maps["city"].get(str(row.get("city", "unknown")).strip() or "unknown", 0)),
                float(categorical_maps["state"].get(str(row.get("state", "unknown")).strip() or "unknown", 0)),
                float(_safe_float(row.get("zip")) or 0.0),
                float(lat),
                float(lon),
                float(_safe_float(row.get("city_pop")) or 0.0),
                float(categorical_maps["job"].get(str(row.get("job", "unknown")).strip() or "unknown", 0)),
                float(_safe_float(row.get("unix_time")) or 0.0),
                float(merch_lat),
                float(merch_lon),
                float(hour),
                float(day),
                float(month),
                float(is_weekend),
                float(is_night),
                float(age),
                float(math.log1p(amt)),
                float(distance_km),
            ]
        )
        row_indexes.append(index)

    return row_indexes, feature_rows


def score_rows(
    raw_rows: List[Dict[str, Any]],
    paysim_weight: float = 0.3,
    kartik_weight: float = 0.7,
    final_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    result_rows = [dict(row) for row in raw_rows]

    paysim_indexes, paysim_features, paysim_derived = prepare_paysim_rows(raw_rows)
    kartik_indexes, kartik_features = prepare_kartik_rows(raw_rows)

    paysim_probs = predict_forest_proba(_paysim_model, paysim_features) if paysim_features else []
    kartik_probs = predict_forest_proba(_kartik_model, kartik_features) if kartik_features else []

    for row in result_rows:
        row["paysim_prob"] = None
        row["kartik_prob"] = None

    for index, probability in zip(paysim_indexes, paysim_probs):
        result_rows[index]["paysim_prob"] = probability
        result_rows[index]["hour"] = paysim_derived.get(index, {}).get("hour")

    for index, probability in zip(kartik_indexes, kartik_probs):
        result_rows[index]["kartik_prob"] = probability

    paysim_available = any(row["paysim_prob"] is not None for row in result_rows)
    kartik_available = any(row["kartik_prob"] is not None for row in result_rows)
    effective_threshold = min(final_threshold, 0.25) if paysim_available ^ kartik_available else final_threshold

    for index, row in enumerate(result_rows):
        p = row["paysim_prob"]
        k = row["kartik_prob"]
        if p is not None and k is not None:
            final_score = paysim_weight * p + kartik_weight * k
        elif p is not None:
            final_score = p
        elif k is not None:
            final_score = k
        else:
            final_score = None

        row["final_score"] = final_score
        row["model_score"] = final_score
        reasons = generate_pay_sim_reasons(raw_rows[index], paysim_derived.get(index, {})) if p is not None else []
        reason_score = paysim_reason_score(raw_rows[index], reasons) if reasons else 0.0
        row["reason_score"] = reason_score
        row["risk_reasons"] = "; ".join(reasons)
        row["risk_percent"] = None if final_score is None else final_score * 100
        row["risk_label"] = assign_risk_label(
            score=final_score,
            effective_threshold=effective_threshold,
            raw_row=raw_rows[index],
            reasons=reasons,
            reason_score=reason_score,
            paysim_available=p is not None,
            kartik_available=k is not None,
        )
        row["fraud_flag"] = 1 if row["risk_label"] in {"MEDIUM RISK", "HIGH RISK"} else 0

    return result_rows


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
        return asdict(self)


def _pick_sample_columns(rows: List[Dict[str, Any]]) -> List[str]:
    preferred = [
        "risk_reasons",
        "amount",
        "amt",
        "type",
        "category",
        "merchant",
        "hour",
        "distance_km",
    ]

    if not rows:
        return preferred[:]

    cols = [column for column in preferred if column in rows[0]]
    skip = {"paysim_prob", "kartik_prob", "final_score", "risk_percent", "fraud_flag", "risk_label", "model_score", "reason_score"}
    for column in rows[0].keys():
        if column in cols or column.lower() in skip:
            continue
        cols.append(column)
        if len(cols) >= 5:
            break

    for tail in ["risk_percent", "fraud_flag", "risk_label"]:
        if rows and tail in rows[0] and tail not in cols:
            cols.append(tail)

    return cols


def summarize_results(
    scored_rows: List[Dict[str, Any]], job_id: str, file_name: str, download_path: str
) -> RunSummary:
    counts = {"HIGH RISK": 0, "MEDIUM RISK": 0, "LOW RISK": 0, "UNKNOWN": 0}
    for row in scored_rows:
        counts[row.get("risk_label", "UNKNOWN")] = counts.get(row.get("risk_label", "UNKNOWN"), 0) + 1

    flagged = sum(int(row.get("fraud_flag") or 0) for row in scored_rows)
    risk_values = [float(row["risk_percent"]) for row in scored_rows if row.get("risk_percent") is not None]
    avg_risk = round(sum(risk_values) / len(risk_values), 2) if risk_values else 0.0
    max_risk = round(max(risk_values), 2) if risk_values else 0.0

    sample_cols = _pick_sample_columns(scored_rows)
    top_rows = sorted(
        scored_rows,
        key=lambda row: row.get("final_score") if row.get("final_score") is not None else -1,
        reverse=True,
    )[:8]
    top_cases = [{column: row.get(column) for column in sample_cols if column in row} for row in top_rows]

    return RunSummary(
        job_id=job_id,
        file_name=safe_display_name(file_name, job_id),
        processed_at=datetime.utcnow().isoformat() + "Z",
        rows=len(scored_rows),
        high_risk=counts.get("HIGH RISK", 0),
        medium_risk=counts.get("MEDIUM RISK", 0),
        low_risk=counts.get("LOW RISK", 0),
        unknown=counts.get("UNKNOWN", 0),
        flagged=flagged,
        avg_risk_percent=avg_risk,
        max_risk_percent=max_risk,
        download_path=download_path,
        top_cases=top_cases,
    )


def score_csv_file(
    csv_path: Path,
    output_path: Path,
    paysim_weight: float = 0.3,
    kartik_weight: float = 0.7,
    final_threshold: float = 0.5,
) -> Tuple[List[Dict[str, Any]], RunSummary]:
    raw_rows = _read_csv_rows(csv_path)
    scored_rows = score_rows(
        raw_rows,
        paysim_weight=paysim_weight,
        kartik_weight=kartik_weight,
        final_threshold=final_threshold,
    )

    if all(row.get("final_score") is None for row in scored_rows):
        raise ValueError(
            "CSV did not contain any columns the models recognize. "
            "Include PaySim-style or Kartik-style transaction columns."
        )

    _write_csv_rows(output_path, scored_rows)
    job_id = output_path.stem
    summary = summarize_results(
        scored_rows, job_id=job_id, file_name=csv_path.name, download_path=str(output_path)
    )
    return scored_rows, summary


def generate_job_id() -> str:
    return uuid.uuid4().hex[:12]

