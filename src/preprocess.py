"""Clean raw CAA data and define the binary delay target."""

import pandas as pd


# Percentage of flights >15 min late above which a route is "delayed"
DELAY_THRESHOLD = 50.0

# Minimum flights for a row to be meaningful
MIN_FLIGHTS = 10


def compute_late_percent(df: pd.DataFrame) -> pd.Series:
    """Sum all delay-bucket columns that represent >15 minutes late."""
    late_cols = [
        "flights_between_16_and_30_minutes_late_percent",
        "flights_between_31_and_60_minutes_late_percent",
        "flights_between_61_and_120_minutes_late_percent",
        "flights_between_121_and_180_minutes_late_percent",
        "flights_between_181_and_360_minutes_late_percent",
        "flights_more_than_360_minutes_late_percent",
    ]
    return df[late_cols].sum(axis=1)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning: drop low-volume rows, filter to scheduled flights."""
    df = df.copy()

    # Keep only scheduled flights (charter routes are noisy and sparse)
    df = df[df["scheduled_charter"] == "S"]

    # Drop rows with very few flights (unreliable percentages)
    df = df[df["number_flights_matched"] >= MIN_FLIGHTS]

    # Drop rows where all delay-bucket columns are null
    delay_cols = [c for c in df.columns if "percent" in c and "flights" in c]
    df = df.dropna(subset=delay_cols, how="all")

    df = df.reset_index(drop=True)
    return df


def add_target(df: pd.DataFrame, threshold: float = DELAY_THRESHOLD) -> pd.DataFrame:
    """Add binary target: 1 if >threshold% of flights were >15 min late."""
    df = df.copy()
    df["late_percent"] = compute_late_percent(df)
    df["is_delayed"] = (df["late_percent"] >= threshold).astype(int)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Run full preprocessing: clean then add target."""
    df = clean(df)
    df = add_target(df)
    return df


if __name__ == "__main__":
    from src.load_data import load_raw_data

    df = load_raw_data()
    df = preprocess(df)
    print(f"Rows after cleaning: {len(df):,}")
    print(f"\nTarget distribution:\n{df['is_delayed'].value_counts()}")
    print(f"\nlate_percent stats:\n{df['late_percent'].describe()}")
