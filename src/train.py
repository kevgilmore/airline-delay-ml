"""Train baseline classification model."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.features import TARGET, build_feature_matrix
from src.load_data import load_raw_data
from src.preprocess import preprocess


def split_data(df, test_size=0.2, random_state=42):
    """Split feature matrix into train and test sets."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_baseline(X_train, y_train):
    """Train a Random Forest classifier with class-weight balancing."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def run_training():
    """End-to-end: load, preprocess, featurise, split, train. Returns model and test data."""
    df = load_raw_data()
    df = preprocess(df)
    df = build_feature_matrix(df)

    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    print(f"Train delay rate: {y_train.mean():.2%} | Test delay rate: {y_test.mean():.2%}")

    model = train_baseline(X_train, y_train)
    return model, X_test, y_test


if __name__ == "__main__":
    model, X_test, y_test = run_training()
    print(f"\nModel trained: {model.__class__.__name__}")
    print(f"Number of features: {model.n_features_in_}")
