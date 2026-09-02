from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


# ---------------------------------------------------------
# Test 1: Valid prediction request
# ---------------------------------------------------------
def test_predict_valid_request():
    payload = {
        "MedInc": 1.6812,
        "HouseAge": 25.0,
        "AveRooms": 4.192201,
        "AveBedrms": 1.022284,
        "Population": 1392.0,
        "AveOccup": 3.877437,
        "Latitude": 36.06,
        "Longitude": -119.01,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "predicted_house_value" in data
    assert "model" in data
    assert "model_version" in data

    assert isinstance(data["predicted_house_value"], (int, float))
    assert data["model"] == "CaliforniaHousingRandomForest"
    assert data["model_version"] == 1


# ---------------------------------------------------------
# Test 2: Missing required feature
# ---------------------------------------------------------
def test_predict_missing_feature():
    payload = {
        "MedInc": 1.6812,
        "HouseAge": 25.0,
        "AveRooms": 4.192201,
        "AveBedrms": 1.022284,
        "Population": 1392.0,
        "AveOccup": 3.877437,
        "Latitude": 36.06,
        # Longitude intentionally missing
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


# ---------------------------------------------------------
# Test 3: Invalid feature type
# ---------------------------------------------------------
def test_predict_invalid_feature_type():
    payload = {
        "MedInc": "not-a-number",
        "HouseAge": 25.0,
        "AveRooms": 4.192201,
        "AveBedrms": 1.022284,
        "Population": 1392.0,
        "AveOccup": 3.877437,
        "Latitude": 36.06,
        "Longitude": -119.01,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


# ---------------------------------------------------------
# Test 4: Missing multiple required features
# ---------------------------------------------------------
def test_predict_missing_multiple_features():
    payload = {
        "MedInc": 1.6812,
        "HouseAge": 25.0,
        "AveRooms": 4.192201,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


# ---------------------------------------------------------
# Test 5: Empty request body
# ---------------------------------------------------------
def test_predict_empty_request():
    response = client.post("/predict", json={})

    assert response.status_code == 422


# ---------------------------------------------------------
# Test 6: Response structure
# ---------------------------------------------------------
def test_predict_response_structure():
    payload = {
        "MedInc": 1.6812,
        "HouseAge": 25.0,
        "AveRooms": 4.192201,
        "AveBedrms": 1.022284,
        "Population": 1392.0,
        "AveOccup": 3.877437,
        "Latitude": 36.06,
        "Longitude": -119.01,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    expected_keys = {
        "predicted_house_value",
        "model",
        "model_version",
    }

    assert set(data.keys()) == expected_keys