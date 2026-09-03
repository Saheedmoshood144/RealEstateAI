import pytest

from src.models.recommendation_engine import RecommendationEngine


DATA_PATH = "data/raw/housing_market.csv"


@pytest.fixture
def engine():
    """Create a recommendation engine using the housing dataset."""
    return RecommendationEngine(DATA_PATH)


def test_engine_loads_dataset(engine):
    """Test that the housing dataset loads successfully."""
    assert not engine.data.empty
    assert len(engine.data) == 1500


def test_required_columns_exist(engine):
    """Test that all required recommendation columns exist."""

    required_columns = {
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
    }

    assert required_columns.issubset(engine.data.columns)


def test_city_filter(engine):
    """Test that recommendations can be filtered by city."""

    recommendations = engine.recommend(
        city="Riverside",
        top_n=5,
    )

    assert len(recommendations) <= 5

    if not recommendations.empty:
        assert all(
            recommendations["City"].str.lower() == "riverside"
        )


def test_bedroom_filter(engine):
    """Test that the minimum bedroom requirement is respected."""

    recommendations = engine.recommend(
        min_bedrooms=3,
        top_n=5,
    )

    assert len(recommendations) <= 5

    if not recommendations.empty:
        assert all(recommendations["Bedrooms"] >= 3)


def test_bathroom_filter(engine):
    """Test that the minimum bathroom requirement is respected."""

    recommendations = engine.recommend(
        min_bathrooms=2,
        top_n=5,
    )

    assert len(recommendations) <= 5

    if not recommendations.empty:
        assert all(recommendations["Bathrooms"] >= 2)


def test_price_filter(engine):
    """Test that the maximum price requirement is respected."""

    max_price = 500000

    recommendations = engine.recommend(
        max_price=max_price,
        top_n=5,
    )

    assert len(recommendations) <= 5

    if not recommendations.empty:
        assert all(recommendations["ListPrice"] <= max_price)


def test_min_sqft_filter(engine):
    """Test that the minimum square-foot requirement is respected."""

    min_sqft = 1500

    recommendations = engine.recommend(
        min_sqft=min_sqft,
        top_n=5,
    )

    assert len(recommendations) <= 5

    if not recommendations.empty:
        assert all(recommendations["SqFt"] >= min_sqft)


def test_max_sqft_filter(engine):
    """Test that the maximum square-foot requirement is respected."""

    max_sqft = 2000

    recommendations = engine.recommend(
        max_sqft=max_sqft,
        top_n=5,
    )

    assert len(recommendations) <= 5

    if not recommendations.empty:
        assert all(recommendations["SqFt"] <= max_sqft)


def test_combined_filters(engine):
    """Test multiple user requirements at the same time."""

    recommendations = engine.recommend(
        city="Riverside",
        min_bedrooms=3,
        min_bathrooms=2,
        max_price=500000,
        top_n=5,
    )

    assert len(recommendations) <= 5

    if not recommendations.empty:
        assert all(
            recommendations["City"].str.lower() == "riverside"
        )
        assert all(recommendations["Bedrooms"] >= 3)
        assert all(recommendations["Bathrooms"] >= 2)
        assert all(recommendations["ListPrice"] <= 500000)


def test_top_n_limit(engine):
    """Test that the engine returns no more than top_n properties."""

    recommendations = engine.recommend(
        top_n=3,
    )

    assert len(recommendations) == 3


def test_invalid_top_n(engine):
    """Test that invalid top_n values raise an error."""

    with pytest.raises(ValueError):
        engine.recommend(top_n=0)


def test_get_listing(engine):
    """Test retrieving a property by ListingID."""

    listing = engine.get_listing("L00001")

    assert listing is not None
    assert str(listing["ListingID"]) == "L00001"


def test_get_nonexistent_listing(engine):
    """Test that an unknown ListingID returns None."""

    listing = engine.get_listing("DOES_NOT_EXIST")

    assert listing is None


def test_neighborhood_filter(engine):
    """Test that neighborhood filtering works."""

    recommendations = engine.recommend(
        neighborhood="Oakwood",
        top_n=5,
    )

    assert len(recommendations) <= 5

    if not recommendations.empty:
        assert all(
            recommendations["Neighborhood"].str.lower()
            == "oakwood"
        )


def test_property_type_filter(engine):
    """Test that property type filtering works."""

    recommendations = engine.recommend(
        property_type="Condo",
        top_n=5,
    )

    assert len(recommendations) <= 5

    if not recommendations.empty:
        assert all(
            recommendations["PropertyType"].str.lower()
            == "condo"
        )


def test_condition_preference_affects_score(engine):
    """Test that preferred condition adds the expected score bonus."""

    baseline = engine.recommend(
        top_n=len(engine.data),
    )

    preferred = engine.recommend(
        condition="Excellent",
        top_n=len(engine.data),
    )

    baseline_scores = baseline.set_index("ListingID")[
        "RecommendationScore"
    ]

    preferred_scores = preferred.set_index("ListingID")[
        "RecommendationScore"
    ]

    excellent_ids = engine.data.loc[
        engine.data["Condition"].str.lower() == "excellent",
        "ListingID",
    ]

    assert not excellent_ids.empty

    for listing_id in excellent_ids:
        score_difference = (
            preferred_scores.loc[listing_id]
            - baseline_scores.loc[listing_id]
        )

        assert score_difference == 10.0


def test_recommendation_score_exists(engine):
    """Test that recommendations contain a valid score."""

    recommendations = engine.recommend(
        city="Riverside",
        top_n=5,
    )

    assert "RecommendationScore" in recommendations.columns

    if not recommendations.empty:
        assert recommendations["RecommendationScore"].notna().all()
        assert (recommendations["RecommendationScore"] >= 0).all()


def test_recommendations_are_ranked(engine):
    """Test that recommendations are sorted by score."""

    recommendations = engine.recommend(
        city="Riverside",
        top_n=10,
    )

    if len(recommendations) > 1:
        scores = recommendations["RecommendationScore"].tolist()

        assert scores == sorted(scores, reverse=True)


def test_city_filter_is_case_insensitive(engine):
    """Test that city searches are case-insensitive."""

    lowercase_results = engine.recommend(
        city="riverside",
        top_n=5,
    )

    uppercase_results = engine.recommend(
        city="RIVERSIDE",
        top_n=5,
    )

    assert set(lowercase_results["ListingID"]) == set(
        uppercase_results["ListingID"]
    )


def test_no_matching_properties(engine):
    """Test that impossible requirements return no recommendations."""

    recommendations = engine.recommend(
        city="Riverside",
        min_bedrooms=100,
        top_n=5,
    )

    assert recommendations.empty