import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# =========================================================
# PAYSIM CLEANING
# =========================================================

df1 = pd.read_csv("paysim1.csv")   # change if filename differs

df1.columns = df1.columns.str.strip().str.lower()
df1 = df1.drop_duplicates()

# numeric conversion
numeric_cols_1 = [
    "step", "amount", "oldbalanceorg", "newbalanceorig",
    "oldbalancedest", "newbalancedest", "isfraud", "isflaggedfraud"
]
for col in numeric_cols_1:
    if col in df1.columns:
        df1[col] = pd.to_numeric(df1[col], errors="coerce")

# remove rows only if important columns are broken
critical_cols_1 = [
    "step", "type", "amount", "oldbalanceorg", "newbalanceorig",
    "oldbalancedest", "newbalancedest", "isfraud"
]
df1 = df1.dropna(subset=[c for c in critical_cols_1 if c in df1.columns])

# valid amount
df1 = df1[df1["amount"] >= 0]

# feature engineering
df1["hour"] = df1["step"] % 24
df1["is_night"] = ((df1["hour"] >= 23) | (df1["hour"] <= 5)).astype(int)
df1["amount_log"] = np.log1p(df1["amount"])

df1["orig_balance_diff"] = df1["newbalanceorig"] - df1["oldbalanceorg"]
df1["dest_balance_diff"] = df1["newbalancedest"] - df1["oldbalancedest"]

df1["orig_zero"] = (df1["oldbalanceorg"] == 0).astype(int)
df1["dest_zero"] = (df1["oldbalancedest"] == 0).astype(int)

# optional but useful consistency features
df1["orig_expected_diff"] = df1["oldbalanceorg"] - df1["newbalanceorig"] - df1["amount"]
df1["dest_expected_diff"] = df1["newbalancedest"] - df1["oldbalancedest"] - df1["amount"]

# drop ID-like columns
df1 = df1.drop(columns=["nameorig", "namedest"], errors="ignore")

# encode type
if "type" in df1.columns:
    df1["type"] = df1["type"].astype("category").cat.codes

print("PaySim cleaned shape:", df1.shape)
print(df1.isnull().sum())

df1.to_csv("paysim_cleaned.csv", index=False)


# =========================================================
# KARTIK CLEANING
# =========================================================

df2 = pd.read_csv("fraudTrain.csv")   # change path if needed

df2.columns = df2.columns.str.strip().str.lower()
df2 = df2.drop_duplicates()
df2 = df2.drop(columns=["unnamed:_0"], errors="ignore")

# datetime conversion
if "trans_date_trans_time" in df2.columns:
    df2["trans_date_trans_time"] = pd.to_datetime(df2["trans_date_trans_time"], errors="coerce")
if "dob" in df2.columns:
    df2["dob"] = pd.to_datetime(df2["dob"], errors="coerce")

# numeric conversion
numeric_cols_2 = [
    "amt", "lat", "long", "city_pop", "merch_lat", "merch_long", "is_fraud"
]
for col in numeric_cols_2:
    if col in df2.columns:
        df2[col] = pd.to_numeric(df2[col], errors="coerce")

# only drop rows missing important fields
critical_cols_2 = ["trans_date_trans_time", "amt", "is_fraud"]
df2 = df2.dropna(subset=[c for c in critical_cols_2 if c in df2.columns])

# valid amount
df2 = df2[df2["amt"] >= 0]

# fill categorical nulls instead of dropping all rows
for col in df2.select_dtypes(include=["object"]).columns:
    df2[col] = df2[col].fillna("unknown")

# fill numeric nulls with median
for col in df2.select_dtypes(include=["number"]).columns:
    df2[col] = df2[col].fillna(df2[col].median())

# time features
df2["hour"] = df2["trans_date_trans_time"].dt.hour
df2["day"] = df2["trans_date_trans_time"].dt.weekday
df2["month"] = df2["trans_date_trans_time"].dt.month
df2["is_weekend"] = (df2["day"] >= 5).astype(int)
df2["is_night"] = ((df2["hour"] >= 23) | (df2["hour"] <= 5)).astype(int)

# age
if "dob" in df2.columns:
    df2["age"] = ((df2["trans_date_trans_time"] - df2["dob"]).dt.days / 365.25)
    df2["age"] = df2["age"].clip(lower=18, upper=100).fillna(df2["age"].median()).astype(int)

# log amount
df2["amt_log"] = np.log1p(df2["amt"])

# vectorized distance
if all(col in df2.columns for col in ["lat", "long", "merch_lat", "merch_long"]):
    lat1 = np.radians(df2["lat"])
    lon1 = np.radians(df2["long"])
    lat2 = np.radians(df2["merch_lat"])
    lon2 = np.radians(df2["merch_long"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    df2["distance_km"] = 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# drop raw datetime columns
df2 = df2.drop(columns=["trans_date_trans_time", "dob"], errors="ignore")

# label encode categoricals
label_encoders = {}
for col in df2.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df2[col] = le.fit_transform(df2[col].astype(str))
    label_encoders[col] = le

print("Kartik cleaned shape:", df2.shape)
print(df2.isnull().sum())

df2.to_csv("kartik_cleaned.csv", index=False)

