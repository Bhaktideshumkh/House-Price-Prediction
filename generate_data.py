"""Generate a realistic synthetic housing dataset (prices in Indian Rupees, Lakhs).

Replace data/housing.csv with a real dataset (e.g. from Kaggle) that has the
same columns if you want to train on real data.
"""
import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# base price per sq.ft (Rs) and city tier multiplier
CITIES = {
    "Mumbai": 21000,
    "Pune": 8200,
    "Nashik": 5200,
    "Nagpur": 4600,
    "Chh. Sambhajinagar": 4300,
    "Thane": 11500,
}
LOCALITY = {"Premium": 1.35, "Standard": 1.0, "Budget": 0.78}
FURNISHING = {"Unfurnished": 1.0, "Semi-Furnished": 1.05, "Furnished": 1.11}


def make_dataset(n: int = 6000) -> pd.DataFrame:
    city = RNG.choice(list(CITIES), n, p=[0.14, 0.24, 0.17, 0.15, 0.15, 0.15])
    locality = RNG.choice(list(LOCALITY), n, p=[0.22, 0.52, 0.26])
    bhk = RNG.choice([1, 2, 3, 4, 5], n, p=[0.14, 0.38, 0.30, 0.14, 0.04])
    area = np.clip(RNG.normal(330 + bhk * 280, 90, n), 280, 4500).round()
    bathrooms = np.clip(bhk + RNG.integers(-1, 2, n), 1, 6)
    age = RNG.integers(0, 31, n)
    total_floors = RNG.integers(1, 31, n)
    floor = np.array([RNG.integers(0, f + 1) for f in total_floors])
    parking = RNG.choice([0, 1, 2], n, p=[0.25, 0.55, 0.20])
    furnishing = RNG.choice(list(FURNISHING), n, p=[0.4, 0.38, 0.22])
    lift = (total_floors > 4).astype(int) & RNG.choice([0, 1], n, p=[0.1, 0.9])
    metro_km = np.round(RNG.exponential(3.0, n).clip(0.1, 20), 1)

    base = np.array([CITIES[c] for c in city], dtype=float)
    price = area * base
    price *= np.array([LOCALITY[l] for l in locality])
    price *= np.array([FURNISHING[f] for f in furnishing])
    price *= 1 - 0.011 * age                       # depreciation
    price *= 1 + 0.004 * np.minimum(floor, 15)     # higher floors cost more
    price *= 1 + 0.025 * parking
    price *= 1 + 0.03 * lift
    price *= 1 + 0.02 * (bathrooms - bhk).clip(-1, 1)
    price *= np.exp(-0.012 * metro_km)             # closer to metro = costlier
    price *= RNG.lognormal(0, 0.06, n)             # market noise
    price_lakh = (price / 1e5).round(2)

    return pd.DataFrame({
        "city": city, "locality": locality, "area_sqft": area.astype(int),
        "bhk": bhk, "bathrooms": bathrooms, "age_years": age,
        "floor": floor, "total_floors": total_floors, "parking": parking,
        "furnishing": furnishing, "lift": lift, "metro_km": metro_km,
        "price_lakh": price_lakh,
    })


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = make_dataset()
    df.to_csv("data/housing.csv", index=False)
    print(f"Saved data/housing.csv  shape={df.shape}")
    print(df.head())
