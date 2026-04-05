from __future__ import annotations

import gzip
import json
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"


def export_forest(src_name: str, dst_name: str) -> None:
    model = joblib.load(BACKEND / src_name)
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        values = []
        for node_values in tree.value:
            values.append([float(v) for v in node_values[0].tolist()])
        trees.append(
            {
                "children_left": tree.children_left.tolist(),
                "children_right": tree.children_right.tolist(),
                "feature": tree.feature.tolist(),
                "threshold": [float(v) for v in tree.threshold.tolist()],
                "values": values,
            }
        )

    payload = {
        "n_classes": int(model.n_classes_),
        "n_features_in": int(model.n_features_in_),
        "trees": trees,
    }

    with gzip.open(BACKEND / dst_name, "wt", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))


if __name__ == "__main__":
    export_forest("paysim_model.pkl", "paysim_model.json.gz")
    export_forest("kartik_model.pkl", "kartik_model.json.gz")

