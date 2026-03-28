"""Save trained model and metadata for serving predictions."""

import joblib

from src.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_feature_matrix,
    select_features,
)
from src.load_data import load_raw_data
from src.preprocess import preprocess
from src.train import split_data, train_baseline

MODELS_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent / "models"


def save():
    df = load_raw_data()
    df = preprocess(df)

    # Extract dropdown values before one-hot encoding
    selected = select_features(df)
    category_values = {
        col: sorted(selected[col].dropna().unique().tolist())
        for col in CATEGORICAL_FEATURES
    }

    # Global numeric defaults (fallback)
    numeric_defaults = {
        col: float(selected[col].median()) for col in NUMERIC_FEATURES
    }

    # Per-route numeric medians keyed by (airport, country)
    route_stats = {}
    for (airport, country), group in df.groupby(
        ["reporting_airport", "origin_destination_country"]
    ):
        route_stats[(airport, country)] = {
            col: float(group[col].median()) for col in NUMERIC_FEATURES
        }

    # Which airlines fly each airport-country pair
    route_airlines = {}
    for (airport, country), group in df.groupby(
        ["reporting_airport", "origin_destination_country"]
    ):
        route_airlines[(airport, country)] = sorted(
            group["airline_name"].unique().tolist()
        )

    # Flight-level detail: destinations and flight counts per
    # (airport, country, airline, direction) for generating schedules
    flights = {}
    group_cols = [
        "reporting_airport",
        "origin_destination_country",
        "airline_name",
        "arrival_departure",
    ]
    for keys, group in df.groupby(group_cols):
        route_key = "||".join(keys)
        dests = []
        for _, row in group.iterrows():
            dests.append(
                {
                    "destination": row["origin_destination"],
                    "flights_per_month": int(row["number_flights_matched"]),
                    "avg_delay": round(row["average_delay_mins"], 1)
                    if not __import__("math").isnan(row["average_delay_mins"])
                    else None,
                }
            )
        # Deduplicate destinations (multiple reporting periods), keep latest
        seen = {}
        for d in dests:
            seen[d["destination"]] = d
        flights[route_key] = sorted(seen.values(), key=lambda x: x["destination"])

    # Build features and train
    df_features = build_feature_matrix(df)
    X_train, X_test, y_train, y_test = split_data(df_features)
    model = train_baseline(X_train, y_train)

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "model.joblib")
    joblib.dump(
        {
            "columns": X_train.columns.tolist(),
            "category_values": category_values,
            "numeric_defaults": numeric_defaults,
            "route_stats": route_stats,
            "route_airlines": route_airlines,
            "flights": flights,
        },
        MODELS_DIR / "model_metadata.joblib",
    )
    print(f"Saved model and metadata to {MODELS_DIR}")


if __name__ == "__main__":
    save()
