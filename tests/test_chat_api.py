from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_chat_recommendation_request():
    response = client.post(
        "/chat",
        json={
            "message": (
                "I want a townhouse in Riverside "
                "with 3 bedrooms and 2 bathrooms"
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["intent"] == "recommend_properties"
    assert "properties matching your request" in data["message"]
    assert isinstance(data["data"], list)
    assert len(data["data"]) <= 5


def test_chat_recommendation_returns_expected_fields():
    response = client.post(
        "/chat",
        json={
            "message": (
                "Show me houses in Riverside "
                "with at least 3 bedrooms"
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["intent"] == "recommend_properties"

    if data["data"]:
        property_data = data["data"][0]

        expected_fields = {
            "ListingID",
            "City",
            "Neighborhood",
            "PropertyType",
            "Condition",
            "Bedrooms",
            "Bathrooms",
            "SqFt",
            "ListPrice",
            "RecommendationScore",
        }

        assert expected_fields.issubset(
            property_data.keys()
        )


def test_chat_prediction_request():
    response = client.post(
        "/chat",
        json={
            "message": (
                "I want to estimate the value "
                "of a California property"
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["intent"] == "predict_price"

    assert "need the following information" in data["message"]

    assert "missing_features" in data["data"]

    expected_features = {
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    }

    assert set(data["data"]["missing_features"]) == expected_features


def test_chat_help_request():
    response = client.post(
        "/chat",
        json={
            "message": "What can you do?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["intent"] == "general_help"
    assert data["data"] is None


def test_chat_unknown_request():
    response = client.post(
        "/chat",
        json={
            "message": "Tell me something unrelated"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["intent"] == "unknown"
    assert data["data"] is None


def test_chat_empty_message():
    response = client.post(
        "/chat",
        json={
            "message": ""
        },
    )

    assert response.status_code == 422


def test_chat_missing_message():
    response = client.post(
        "/chat",
        json={}
    )

    assert response.status_code == 422