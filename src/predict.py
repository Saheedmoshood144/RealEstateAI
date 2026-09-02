from pathlib import Path

import mlflow
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_URI = "models:/CaliforniaHousingRandomForest/1"

PROCESSED_DIR = Path("data/processed")


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """Load the registered Random Forest model from MLflow."""

    print("Loading registered model...")
    print(f"Model URI: {MODEL_URI}")

    model = mlflow.pyfunc.load_model(MODEL_URI)

    print("Model loaded successfully.")

    return model


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_sample_data():
    """Load a sample from the California test dataset."""

    X_test = pd.read_csv(
        PROCESSED_DIR / "X_test.csv"
    )

    y_test = pd.read_csv(
        PROCESSED_DIR / "y_test.csv"
    ).squeeze()

    return X_test, y_test


# ============================================================
# MAKE PREDICTION
# ============================================================

def predict():
    """Load the registered model and make a sample prediction."""

    model = load_model()

    X_test, y_test = load_sample_data()

    # Take the first property from the test set
    sample = X_test.iloc[[0]]

    actual_value = y_test.iloc[0]

    prediction = model.predict(sample)[0]

    print("\n=== REAL ESTATE PREDICTION ===")

    print("\nInput features:")
    print(sample.to_string(index=False))

    print(f"\nPredicted house value: {prediction:.4f}")
    print(f"Actual house value:    {actual_value:.4f}")

    print("\nPrediction completed successfully.")

    return prediction


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    predict()