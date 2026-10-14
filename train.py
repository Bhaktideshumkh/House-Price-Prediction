"""Train and compare regression models, save the best one + metrics."""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["city", "locality", "furnishing"]
NUMERIC = ["area_sqft", "bhk", "bathrooms", "age_years", "floor",
           "total_floors", "parking", "lift", "metro_km"]
TARGET = "price_lakh"


def build_pipeline(model):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def main():
    df = pd.read_csv("data/housing.csv")
    X, y = df[CATEGORICAL + NUMERIC], df[TARGET]
    # train on log price so errors are relative, not dominated by Mumbai
    y_log = np.log1p(y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_log, test_size=0.2, random_state=42)

    candidates = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=250, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=400, learning_rate=0.06,
                                                       max_depth=4, random_state=42),
    }

    results, fitted = {}, {}
    for name, model in candidates.items():
        pipe = build_pipeline(model).fit(X_tr, y_tr)
        pred = np.expm1(pipe.predict(X_te))
        true = np.expm1(y_te)
        results[name] = {
            "r2": round(float(r2_score(true, pred)), 4),
            "mae_lakh": round(float(mean_absolute_error(true, pred)), 2),
            "rmse_lakh": round(float(np.sqrt(mean_squared_error(true, pred))), 2),
            "mape_pct": round(float(np.mean(np.abs((true - pred) / true)) * 100), 2),
        }
        fitted[name] = pipe
        print(f"{name:20s} {results[name]}")

    best = max(results, key=lambda k: results[k]["r2"])
    print("Best model:", best)

    # permutation-free importance: use model's own importances when available
    best_pipe = fitted[best]
    feat_names = best_pipe.named_steps["pre"].get_feature_names_out()
    importances = getattr(best_pipe.named_steps["model"], "feature_importances_", None)
    imp = {}
    if importances is not None:
        for fname, v in zip(feat_names, importances):
            base = fname.split("__", 1)[1]
            for c in CATEGORICAL:
                if base.startswith(c + "_"):
                    base = c
            imp[base] = imp.get(base, 0.0) + float(v)
        imp = dict(sorted(imp.items(), key=lambda kv: -kv[1]))

    # actual vs predicted sample for scatter chart
    pred_all = np.expm1(best_pipe.predict(X_te))
    idx = np.random.default_rng(0).choice(len(X_te), 200, replace=False)
    scatter = [[round(float(np.expm1(y_te.iloc[i])), 1), round(float(pred_all[i]), 1)] for i in idx]

    os.makedirs("model", exist_ok=True)
    joblib.dump(best_pipe, "model/model.joblib")
    with open("model/metrics.json", "w") as f:
        json.dump({"best": best, "results": results,
                   "importance": imp, "scatter": scatter,
                   "n_train": len(X_tr), "n_test": len(X_te)}, f, indent=2)
    print("Saved model/model.joblib and model/metrics.json")


if __name__ == "__main__":
    main()
