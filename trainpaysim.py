import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

print("Loading cleaned dataset...")
df = pd.read_csv("paysim_cleaned.csv", engine="python", on_bad_lines="skip", nrows=300000)

print(df.head())
print(df.shape)
# make dtypes smaller to save memory
for col in df.columns:
    if df[col].dtype == "float64":
        df[col] = df[col].astype("float32")
    elif df[col].dtype == "int64":
        df[col] = df[col].astype("int32")

X = df.drop(columns=[
    "isfraud",
    "isflaggedfraud",
    "orig_expected_diff",
    "dest_expected_diff"
], errors="ignore")

print("Columns used for training:")
print(list(X.columns))

y = df["isfraud"].astype("int8")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=30,
    max_depth=10,
    random_state=42,
    class_weight="balanced",
    n_jobs=1
)

print("Training model...")
model.fit(X_train, y_train)


y_prob = model.predict_proba(X_test)[:, 1]
threshold = 0.5
y_pred = (y_prob > threshold).astype(int)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nTop 15 Important Features:")
print(feature_importance.head(15))

joblib.dump(model, "paysim_model.pkl")