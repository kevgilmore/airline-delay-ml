"""Load and combine raw UK CAA punctuality CSV files."""

import glob
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_raw_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load all CSV files from the raw data directory into a single DataFrame."""
    files = sorted(glob.glob(str(data_dir / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = [pd.read_csv(f, encoding="utf-8-sig") for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def summarise_data(df: pd.DataFrame) -> None:
    """Print a quick summary of the loaded dataset."""
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]}")
    print(f"\nColumn names:\n{list(df.columns)}\n")
    print(f"dtypes:\n{df.dtypes}\n")
    print(f"Null counts:\n{df.isnull().sum()[df.isnull().sum() > 0]}\n")
    print(f"Sample rows:")
    print(df.head(3).to_string())


if __name__ == "__main__":
    df = load_raw_data()
    summarise_data(df)
