from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_URI = "models:/CaliforniaHousingRandomForest/1"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="RealEstateAI",
    description="California housing price prediction API",
    version="1.0.0",
)


# ============================================================
# INPUT SCHEMA
# ============================================================

class HouseInput(BaseModel):
    """
    Features required by the California housing model.
    """

    MedInc: float = Field(
        ...,
        gt=0,
        description="Median income in the block group.",
    )

    HouseAge: float = Field(
        ...,
        ge=0,
        le=100,
        description="Median house age in years.",
    )

    AveRooms: float = Field(
        ...,
        gt=0,
        description="Average number of rooms.",
    )

    AveBedrms: float = Field(
        ...,
        gt=0,
        description="Average number of bedrooms.",
    )

    Population: float = Field(
        ...,
        ge=0,
        description="Block-group population.",
    )

    AveOccup: float = Field(
        ...,
        gt=0,
        description="Average household occupancy.",
    )

    Latitude: float = Field(
        ...,
        ge=32,
        le=42,
        description="California latitude.",
    )

    Longitude: float = Field(
        ...,
        ge=-125,
        le=-114,
        description="California longitude.",
    )


# ============================================================
# LOAD REGISTERED MODEL
# ============================================================

print("Loading registered model...")

model = mlflow.pyfunc.load_model(MODEL_URI)

print("Registered model loaded successfully.")


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():
    """API health check."""

    return {
        "application": "RealEstateAI",
        "status": "running",
        "model": "CaliforniaHousingRandomForest",
        "version": 1,
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict_house_price(house: HouseInput):
    """
    Predict California median house value.
    """

    input_data = pd.DataFrame(
        [
            {
                "MedInc": house.MedInc,
                "HouseAge": house.HouseAge,
                "AveRooms": house.AveRooms,
                "AveBedrms": house.AveBedrms,
                "Population": house.Population,
                "AveOccup": house.AveOccup,
                "Latitude": house.Latitude,
                "Longitude": house.Longitude,
            }
        ]
    )

    prediction = model.predict(input_data)[0]

    return {
        "predicted_house_value": round(float(prediction), 4),
        "model": "CaliforniaHousingRandomForest",
        "model_version": 1,
    }