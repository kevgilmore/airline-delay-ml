"""Run the full ML pipeline end-to-end."""

from src.evaluate import plot_confusion_matrix, plot_feature_importance, print_report
from src.train import run_training


def main():
    print("=" * 60)
    print("Flight Delay Prediction — Baseline Pipeline")
    print("=" * 60)

    print("\n[1/2] Training model...")
    model, X_test, y_test = run_training()

    print("\n[2/2] Evaluating model...")
    print_report(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test)
    plot_feature_importance(model, X_test.columns)

    print("\nDone.")


if __name__ == "__main__":
    main()
