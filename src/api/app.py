import json
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.recommendation_engine import RecommendationEngine


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_URI = "models:/CaliforniaHousingRandomForest/1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LISTINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "housing_market.csv"
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="RealEstateAI",
    description=(
        "Real estate intelligence API providing California "
        "housing price prediction and property recommendations."
    ),
    version="2.0.0",
)


# ============================================================
# PREDICTION INPUT SCHEMA
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
# RECOMMENDATION INPUT SCHEMA
# ============================================================

class RecommendationInput(BaseModel):
    """
    User preferences for property recommendations.
    """

    city: str | None = Field(
        default=None,
        description="Preferred city.",
    )

    neighborhood: str | None = Field(
        default=None,
        description="Preferred neighborhood.",
    )

    property_type: str | None = Field(
        default=None,
        description=(
            "Preferred property type, such as Condo, "
            "Single Family, Townhouse, or Multi-Family."
        ),
    )

    condition: str | None = Field(
        default=None,
        description=(
            "Preferred property condition, such as "
            "Excellent, Good, Fair, or Poor."
        ),
    )

    min_bedrooms: int | None = Field(
        default=None,
        ge=0,
        le=20,
        description="Minimum number of bedrooms.",
    )

    min_bathrooms: float | None = Field(
        default=None,
        ge=0,
        le=20,
        description="Minimum number of bathrooms.",
    )

    max_price: float | None = Field(
        default=None,
        gt=0,
        description="Maximum listing price in dollars.",
    )

    min_sqft: int | None = Field(
        default=None,
        ge=0,
        description="Minimum property size in square feet.",
    )

    max_sqft: int | None = Field(
        default=None,
        ge=0,
        description="Maximum property size in square feet.",
    )

    top_n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of recommendations to return.",
    )


# ============================================================
# LOAD REGISTERED PREDICTION MODEL
# ============================================================

print("Loading registered prediction model...")

model = mlflow.pyfunc.load_model(MODEL_URI)

print("Registered prediction model loaded successfully.")


# ============================================================
# LOAD RECOMMENDATION ENGINE
# ============================================================

print("Loading recommendation engine...")

recommendation_engine = RecommendationEngine(
    LISTINGS_PATH
)

print(
    "Recommendation engine loaded successfully "
    f"with {len(recommendation_engine.data)} listings."
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():
    """
    API health check and service information.
    """

    return {
        "application": "RealEstateAI",
        "status": "running",
        "api_version": "2.0.0",
        "services": {
            "prediction": {
                "endpoint": "/predict",
                "model": "CaliforniaHousingRandomForest",
                "model_version": 1,
            },
            "recommendation": {
                "endpoint": "/recommend",
                "listings": len(recommendation_engine.data),
            },
        },
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
        "predicted_house_value": round(
            float(prediction),
            4,
        ),
        "model": "CaliforniaHousingRandomForest",
        "model_version": 1,
    }


# ============================================================
# RECOMMENDATION ENDPOINT
# ============================================================

@app.post("/recommend")
def recommend_properties(
    preferences: RecommendationInput,
):
    """
    Return ranked property recommendations based on
    user preferences.
    """

    if (
        preferences.min_sqft is not None
        and preferences.max_sqft is not None
        and preferences.min_sqft > preferences.max_sqft
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "min_sqft cannot be greater than max_sqft."
            ),
        )

    recommendations = recommendation_engine.recommend(
        city=preferences.city,
        neighborhood=preferences.neighborhood,
        property_type=preferences.property_type,
        condition=preferences.condition,
        min_bedrooms=preferences.min_bedrooms,
        min_bathrooms=preferences.min_bathrooms,
        max_price=preferences.max_price,
        min_sqft=preferences.min_sqft,
        max_sqft=preferences.max_sqft,
        top_n=preferences.top_n,
    )

    if recommendations.empty:
        return {
            "count": 0,
            "message": (
                "No properties matched the requested criteria."
            ),
            "recommendations": [],
        }

    output_columns = [
        "ListingID",
        "City",
        "Neighborhood",
        "PropertyType",
        "Condition",
        "Bedrooms",
        "Bathrooms",
        "SqFt",
        "LotSize_sqft",
        "Age_years",
        "GarageSpaces",
        "ListPrice",
        "PricePerSqFt",
        "DaysOnMarket",
        "ListingDate",
        "RecommendationScore",
    ]

    results = recommendations[
        output_columns
    ].copy()

    results["RecommendationScore"] = (
        results["RecommendationScore"].round(2)
    )

    # Convert pandas/numpy values into standard JSON values.
    recommendation_records = json.loads(
        results.to_json(
            orient="records",
        )
    )

    return {
        "count": len(recommendation_records),
        "recommendations": recommendation_records,
    }