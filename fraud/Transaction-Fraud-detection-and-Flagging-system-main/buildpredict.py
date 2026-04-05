import pandas as pd
import numpy as np
import joblib


# =========================================================
# LOAD TRAINED MODELS
# =========================================================
# make sure these files exist
PAYSIM_MODEL_PATH = "paysim_model.pkl"
KARTIK_MODEL_PATH = "kartik_model.pkl"

paysim_model = joblib.load(PAYSIM_MODEL_PATH)
kartik_model = joblib.load(KARTIK_MODEL_PATH)


# =========================================================
# FEATURE LISTS USED DURING TRAINING
# =========================================================
PAYSIM_FEATURES = [
    "step", "type", "amount", "oldbalanceorg", "newbalanceorig",
    "oldbalancedest", "newbalancedest", "hour", "is_night",
    "amount_log", "orig_balance_diff", "dest_balance_diff",
    "orig_zero", "dest_zero"
]

KARTIK_FEATURES = [
    "merchant", "category", "amt", "gender", "city", "state", "zip",
    "lat", "long", "city_pop", "job", "unix_time", "merch_lat",
    "merch_long", "hour", "day", "month", "is_weekend", "is_night",
    "age", "amt_log", "distance_km"
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


def ensure_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def prepare_paysim(df):
    temp = df.copy()
    temp.columns = temp.columns.str.strip().str.lower()

    needed_raw = [
        "step", "type", "amount", "oldbalanceorg",
        "newbalanceorig", "oldbalancedest", "newbalancedest"
    ]
    if not all(col in temp.columns for col in needed_raw):
        return None

    temp = ensure_numeric(temp, [
        "step", "amount", "oldbalanceorg",
        "newbalanceorig", "oldbalancedest", "newbalancedest"
    ])

    temp = temp.dropna(subset=needed_raw)
    temp = temp[temp["amount"] >= 0]

    # encode type if string
    if temp["type"].dtype == object:
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


def prepare_kartik(df):
    temp = df.copy()
    temp.columns = temp.columns.str.strip().str.lower()

    # if timestamp exists, derive time features
    if "trans_date_trans_time" in temp.columns:
        temp["trans_date_trans_time"] = pd.to_datetime(temp["trans_date_trans_time"], errors="coerce")
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
        if temp[col].dtype == object:
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


def score_transactions(input_csv, output_csv="predictions.csv",
                       paysim_weight=0.3, kartik_weight=0.7,
                       final_threshold=0.5):
    df = pd.read_csv(input_csv)

    result = df.copy()
    paysim_prob = None
    kartik_prob = None

    # -------------------------
    # PAYSIM SCORE
    # -------------------------
    X_paysim = prepare_paysim(df)
    if X_paysim is not None and len(X_paysim) > 0:
        paysim_prob = paysim_model.predict_proba(X_paysim)[:, 1]
        result.loc[X_paysim.index, "paysim_prob"] = paysim_prob
    else:
        result["paysim_prob"] = np.nan

    # -------------------------
    # KARTIK SCORE
    # -------------------------
    X_kartik = prepare_kartik(df)
    if X_kartik is not None and len(X_kartik) > 0:
        kartik_prob = kartik_model.predict_proba(X_kartik)[:, 1]
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

    result.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")
    print(result[["paysim_prob", "kartik_prob", "final_score", "risk_percent", "fraud_flag", "risk_label"]].head())

    return result


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    # change this filename to your input file
    score_transactions(
        input_csv="input_transactions.csv",
        output_csv="predictions.csv",
        paysim_weight=0.3,
        kartik_weight=0.7,
        final_threshold=0.5
    )