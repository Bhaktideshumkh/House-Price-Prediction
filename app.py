"""Flask web app: serves the UI and a prediction API."""
import json

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from generate_data import CITIES, FURNISHING, LOCALITY
from train import CATEGORICAL, NUMERIC

app = Flask(__name__)
model = joblib.load("model/model.joblib")
with open("model/metrics.json") as f:
    METRICS = json.load(f)

COLUMNS = CATEGORICAL + NUMERIC


def parse(payload: dict) -> pd.DataFrame:
    total_floors = max(int(payload["total_floors"]), 1)
    row = {
        "city": payload["city"],
        "locality": payload["locality"],
        "furnishing": payload["furnishing"],
        "area_sqft": float(payload["area_sqft"]),
        "bhk": int(payload["bhk"]),
        "bathrooms": int(payload["bathrooms"]),
        "age_years": int(payload["age_years"]),
        "floor": min(int(payload["floor"]), total_floors),
        "total_floors": total_floors,
        "parking": int(payload["parking"]),
        "lift": int(bool(payload["lift"])),
        "metro_km": float(payload["metro_km"]),
    }
    return pd.DataFrame([row])[COLUMNS]


@app.route("/")
def index():
    return render_template(
        "index.html",
        cities=list(CITIES), localities=list(LOCALITY),
        furnishings=list(FURNISHING), best=METRICS["best"],
    )


@app.post("/api/predict")
def predict():
    try:
        X = parse(request.get_json(force=True))
        import numpy as np
        price = float(np.expm1(model.predict(X)[0]))
        mape = METRICS["results"][METRICS["best"]]["mape_pct"] / 100
        return jsonify({
            "price_lakh": round(price, 2),
            "low_lakh": round(price * (1 - mape), 2),
            "high_lakh": round(price * (1 + mape), 2),
            "per_sqft": round(price * 1e5 / float(X["area_sqft"][0])),
        })
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400


@app.get("/api/metrics")
def metrics():
    return jsonify(METRICS)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
