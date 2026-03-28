"""Feature engineering for delay prediction."""

import pandas as pd


# Columns used as features for the model
CATEGORICAL_FEATURES = [
    "reporting_airport",
    "origin_destination_country",
    "airline_name",
    "arrival_departure",
]

NUMERIC_FEATURES = [
    "number_flights_matched",
    "number_flights_cancelled",
    "flights_cancelled_percent",
    "previous_year_month_average_delay",
]

TARGET = "is_delayed"


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and return only the feature columns plus target."""
    cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET]
    return df[cols].copy()


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering: fill nulls, one-hot encode categoricals."""
    df = select_features(df)

    # Fill numeric nulls with median
    for col in NUMERIC_FEATURES:
        df[col] = df[col].fillna(df[col].median())

    # One-hot encode categoricals (drop_first to avoid multicollinearity)
    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)

    return df


if __name__ == "__main__":
    from src.load_data import load_raw_data
    from src.preprocess import preprocess

    df = load_raw_data()
    df = preprocess(df)
    features = build_feature_matrix(df)
    print(f"Feature matrix shape: {features.shape}")
    print(f"\nFeature columns ({features.shape[1] - 1} features):")
    print([c for c in features.columns if c != TARGET])
