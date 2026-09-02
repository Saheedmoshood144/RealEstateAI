from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")


def load_data():
    """Load processed training and testing data."""

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")

    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()

    return X_train, X_test, y_train, y_test


def evaluate_model(y_test, predictions):
    """Calculate regression evaluation metrics."""

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\n=== MODEL EVALUATION ===")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    return mae, rmse, r2


def plot_feature_importance(model, feature_names):
    """Create and save a feature-importance chart."""

    importance = pd.Series(
        model.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    importance.plot(kind="barh")

    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()

    output_path = REPORTS_DIR / "random_forest_feature_importance.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nFeature importance plot saved to: {output_path}")

    print("\n=== FEATURE IMPORTANCE ===")

    for feature, value in importance.sort_values(ascending=False).items():
        print(f"{feature:15s}: {value:.4f}")

    return output_path, importance


def train_model():
    """Train, evaluate, analyze, and track the Random Forest model."""

    X_train, X_test, y_train, y_test = load_data()

    # MLflow experiment
    mlflow.set_experiment("California")
    with mlflow.start_run(run_name="random_forest_baseline"):

        # Model configuration
        n_estimators = 100
        max_depth = 15
        random_state = 42

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

        print("Training Random Forest...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        print("\n=== RANDOM FOREST MODEL ===")
        print(f"Trees: {n_estimators}")
        print(f"Max depth: {max_depth}")

        # Evaluation
        mae, rmse, r2 = evaluate_model(
            y_test,
            predictions,
        )

        # Feature importance
        plot_path, importance = plot_feature_importance(
            model,
            X_train.columns,
        )

        # Log parameters
        mlflow.log_params(
            {
                "model_type": "RandomForestRegressor",
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": random_state,
            }
        )

        # Log metrics
        mlflow.log_metrics(
            {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
        )

        # Log feature importance values
        for feature, value in importance.items():
            mlflow.log_metric(
                f"importance_{feature}",
                value,
            )

        # Log feature-importance plot
        mlflow.log_artifact(str(plot_path))

        # Log trained model
        mlflow.sklearn.log_model(
            model,
            name="random_forest_model",
        )

        print("\n=== MLFLOW TRACKING ===")
        print("Experiment: California")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print("Parameters, metrics, plot, and model logged successfully.")

    return model


if __name__ == "__main__":
    train_model()