"""Evaluate model performance and produce reports."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
)

PLOTS_DIR = Path(__file__).resolve().parent.parent / "models"


def print_report(model, X_test, y_test):
    """Print classification report and ROC-AUC."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["On Time", "Delayed"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")


def plot_confusion_matrix(model, X_test, y_test, save=True):
    """Display and optionally save confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=["On Time", "Delayed"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Confusion Matrix — Baseline Random Forest")
    plt.tight_layout()
    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150)
        print(f"Saved to {PLOTS_DIR / 'confusion_matrix.png'}")
    plt.close(fig)


def plot_feature_importance(model, feature_names, top_n=20, save=True):
    """Plot top N feature importances."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    top.sort_values().plot.barh(ax=ax, color="steelblue")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / "feature_importance.png", dpi=150)
        print(f"Saved to {PLOTS_DIR / 'feature_importance.png'}")
    plt.close(fig)


if __name__ == "__main__":
    from src.train import run_training

    model, X_test, y_test = run_training()
    print()
    print_report(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test)
    plot_feature_importance(model, X_test.columns)
