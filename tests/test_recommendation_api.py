from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


# ============================================================
# Test 1: Valid recommendation request
# ============================================================

def test_recommend_valid_request():
    """Test a valid property recommendation request."""

    payload = {
        "city": "Riverside",
        "min_bedrooms": 3,
        "min_bathrooms": 2,
        "max_price": 500000,
        "top_n": 5,
    }

    response = client.post(
        "/recommend",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "count" in data
    assert "recommendations" in data

    assert data["count"] <= 5
    assert len(data["recommendations"]) <= 5


# ============================================================
# Test 2: City filtering
# ============================================================

def test_recommend_city_filter():
    """Test that the API respects the requested city."""

    payload = {
        "city": "Riverside",
        "top_n": 5,
    }

    response = client.post(
        "/recommend",
        json=payload,
    )

    assert response.status_code == 200

    recommendations = response.json()["recommendations"]

    assert len(recommendations) <= 5

    for property_data in recommendations:
        assert property_data["City"].lower() == "riverside"


# ============================================================
# Test 3: Price constraint
# ============================================================

def test_recommend_max_price():
    """Test that returned properties respect the maximum price."""

    max_price = 500000

    payload = {
        "max_price": max_price,
        "top_n": 5,
    }

    response = client.post(
        "/recommend",
        json=payload,
    )

    assert response.status_code == 200

    recommendations = response.json()["recommendations"]

    for property_data in recommendations:
        assert property_data["ListPrice"] <= max_price


# ============================================================
# Test 4: Bedroom constraint
# ============================================================

def test_recommend_min_bedrooms():
    """Test minimum bedroom filtering."""

    payload = {
        "min_bedrooms": 3,
        "top_n": 5,
    }

    response = client.post(
        "/recommend",
        json=payload,
    )

    assert response.status_code == 200

    recommendations = response.json()["recommendations"]

    for property_data in recommendations:
        assert property_data["Bedrooms"] >= 3


# ============================================================
# Test 5: Bathroom constraint
# ============================================================

def test_recommend_min_bathrooms():
    """Test minimum bathroom filtering."""

    payload = {
        "min_bathrooms": 2,
        "top_n": 5,
    }

    response = client.post(
        "/recommend",
        json=payload,
    )

    assert response.status_code == 200

    recommendations = response.json()["recommendations"]

    for property_data in recommendations:
        assert property_data["Bathrooms"] >= 2


# ============================================================
# Test 6: Top-N constraint
# ============================================================

def test_recommend_top_n():
    """Test that the API respects the requested top_n."""

    payload = {
        "city": "Riverside",
        "top_n": 3,
    }

    response = client.post(
        "/recommend",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["count"] <= 3
    assert len(data["recommendations"]) <= 3


# ============================================================
# Test 7: Invalid top_n
# ============================================================

def test_recommend_invalid_top_n():
    """Test that top_n=0 fails validation."""

    payload = {
        "city": "Riverside",
        "top_n": 0,
    }

    response = client.post(
        "/recommend",
        json=payload,
    )

    assert response.status_code == 422


# ============================================================
# Test 8: Invalid square-footage range
# ============================================================

def test_recommend_invalid_sqft_range():
    """Test that min_sqft cannot exceed max_sqft."""

    payload = {
        "min_sqft": 2500,
        "max_sqft": 1000,
        "top_n": 5,
    }

    response = client.post(
        "/recommend",
        json=payload,
    )

    assert response.status_code == 400

    data = response.json()

    assert (
        data["detail"]
        == "min_sqft cannot be greater than max_sqft."
    )


# ============================================================
# Test 9: No matching properties
# ============================================================

def test_recommend_no_matches():
    """Test the API response when no valid properties match."""

    payload = {
        "city": "Riverside",
        "min_bedrooms": 20,
        "top_n": 5,
    }

    response = client.post(
        "/recommend",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["count"] == 0
    assert data["recommendations"] == []


# ============================================================
# Test 10: Recommendation score
# ============================================================

def test_recommendation_score_present():
    """Test that every recommendation contains a score."""

    payload = {
        "city": "Riverside",
        "top_n": 5,
    }

    response = client.post(
        "/recommend",
        json=payload,
    )

    assert response.status_code == 200

    recommendations = response.json()["recommendations"]

    for property_data in recommendations:
        assert "RecommendationScore" in property_data
        assert property_data["RecommendationScore"] >= 0


# ============================================================
# Test 11: Recommendation ranking
# ============================================================

def test_recommendations_are_ranked():
    """Test that recommendations are ordered by score."""

    payload = {
        "city": "Riverside",
        "top_n": 10,
    }

    response = client.post(
        "/recommend",
        json=payload,
    )

    assert response.status_code == 200

    recommendations = response.json()["recommendations"]

    scores = [
        property_data["RecommendationScore"]
        for property_data in recommendations
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


# ============================================================
# Test 12: Case-insensitive city
# ============================================================

def test_recommend_city_case_insensitive():
    """Test that city matching is case-insensitive."""

    payload = {
        "city": "rIvErSiDe",
        "top_n": 5,
    }

    response = client.post(
        "/recommend",
        json=payload,
    )

    assert response.status_code == 200

    recommendations = response.json()["recommendations"]

    for property_data in recommendations:
        assert property_data["City"].lower() == "riverside"


# ============================================================
# Test 13: Property type filter
# ============================================================

def test_recommend_property_type():
    """Test property type filtering."""

    payload = {
        "property_type": "Condo",
        "top_n": 5,
    }

    response = client.post(
        "/recommend",
        json=payload,
    )

    assert response.status_code == 200

    recommendations = response.json()["recommendations"]

    for property_data in recommendations:
        assert (
            property_data["PropertyType"].lower()
            == "condo"
        )


# ============================================================
# Test 14: Combined preferences
# ============================================================

def test_recommend_combined_preferences():
    """Test several recommendation constraints together."""

    payload = {
        "city": "Riverside",
        "property_type": "Single Family",
        "min_bedrooms": 3,
        "min_bathrooms": 2,
        "max_price": 500000,
        "min_sqft": 1000,
        "top_n": 5,
    }

    response = client.post(
        "/recommend",
        json=payload,
    )

    assert response.status_code == 200

    recommendations = response.json()["recommendations"]

    for property_data in recommendations:
        assert property_data["City"].lower() == "riverside"
        assert (
            property_data["PropertyType"].lower()
            == "single family"
        )
        assert property_data["Bedrooms"] >= 3
        assert property_data["Bathrooms"] >= 2
        assert property_data["ListPrice"] <= 500000
        assert property_data["SqFt"] >= 1000