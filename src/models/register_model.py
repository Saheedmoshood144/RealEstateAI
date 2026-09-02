import mlflow


MODEL_NAME = "CaliforniaHousingRandomForest"

# Paste the Random Forest model URI printed by compare_models.py
MODEL_URI = "models:/m-0ca3d62cb47c4f5bb77e43ff20897070"


def register_model():
    """Register the selected Random Forest model in MLflow."""

    print("Registering selected Random Forest model...")
    print(f"Model URI: {MODEL_URI}")
    print(f"Registered model name: {MODEL_NAME}")

    registered_model = mlflow.register_model(
        model_uri=MODEL_URI,
        name=MODEL_NAME,
    )

    print("\n=== MODEL REGISTERED ===")
    print(f"Name:    {registered_model.name}")
    print(f"Version: {registered_model.version}")
    print(f"Status:  {registered_model.status}")


if __name__ == "__main__":
    register_model()