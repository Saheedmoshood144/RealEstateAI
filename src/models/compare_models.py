from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROCESSED_DIR = Path("data/processed")


# ============================================================
# MLflow CONFIGURATION
# ============================================================

EXPERIMENT_NAME = "California"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load the processed California housing train/test datasets."""

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")

    y_train = pd.read_csv(
        PROCESSED_DIR / "y_train.csv"
    ).squeeze()

    y_test = pd.read_csv(
        PROCESSED_DIR / "y_test.csv"
    ).squeeze()

    return X_train, X_test, y_train, y_test


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(y_test, predictions):
    """Calculate regression evaluation metrics."""

    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test,
        predictions,
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions,
    )

    return mae, rmse, r2


# ============================================================
# LOG MODEL TO MLFLOW
# ============================================================

def log_model_run(
    model,
    model_name,
    X_train,
    X_test,
    y_train,
    y_test,
    parameters,
):
    """Train, evaluate, and log one model to MLflow."""

    with mlflow.start_run(run_name=model_name):

        print(f"\nTraining {model_name}...")

        # Train
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Evaluate
        mae, rmse, r2 = evaluate_model(
            y_test,
            predictions,
        )

        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        mlflow.log_params(parameters)

        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # ----------------------------------------------------
        # Add useful metadata
        # ----------------------------------------------------

        mlflow.set_tag(
            "dataset",
            "California Housing",
        )

        mlflow.set_tag(
            "model_type",
            model_name,
        )

        # ----------------------------------------------------
        # Log model
        # ----------------------------------------------------

        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
        )

        print(f"Logged Model URI: {model_info.model_uri}")
        print(f"Logged Model ID:  {model_info.model_id}")

        # ----------------------------------------------------
        # Terminal output
        # ----------------------------------------------------

        print(f"\n=== {model_name.upper()} ===")

        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")

        print("\nMLflow:")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

        return {
            "model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        }


# ============================================================
# MODEL COMPARISON
# ============================================================

def compare_models():
    """Train and compare Linear Regression and Random Forest."""

    print("Loading California housing data...")

    X_train, X_test, y_train, y_test = load_data()

    print(f"Training features: {X_train.shape}")
    print(f"Testing features:  {X_test.shape}")

    # --------------------------------------------------------
    # Set MLflow experiment
    # --------------------------------------------------------

    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"\nMLflow experiment: {EXPERIMENT_NAME}")

    # ========================================================
    # 1. LINEAR REGRESSION
    # ========================================================

    linear_model = LinearRegression()

    linear_results = log_model_run(
        model=linear_model,
        model_name="linear_regression_baseline",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        parameters={
            "model_type": "LinearRegression",
        },
    )

    # ========================================================
    # 2. RANDOM FOREST
    # ========================================================

    random_forest_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
    )

    random_forest_results = log_model_run(
        model=random_forest_model,
        model_name="random_forest_baseline",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        parameters={
            "model_type": "RandomForestRegressor",
            "n_estimators": 100,
            "max_depth": 15,
            "random_state": 42,
            "n_jobs": -1,
        },
    )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    results = pd.DataFrame(
        [
            linear_results,
            random_forest_results,
        ]
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results[
            [
                "model",
                "MAE",
                "RMSE",
                "R2",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # Identify best model by R²
    # --------------------------------------------------------

    best_model = results.loc[
        results["R2"].idxmax()
    ]

    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(f"Model: {best_model['model']}")
    print(f"MAE:  {best_model['MAE']:.4f}")
    print(f"RMSE: {best_model['RMSE']:.4f}")
    print(f"R²:   {best_model['R2']:.4f}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    compare_models()