from pathlib import Path

from src.conversation.schemas import ConversationIntent
from src.conversation.service import ConversationService


LISTINGS_PATH = Path("data/raw/housing_market.csv")


def create_service():
    return ConversationService(LISTINGS_PATH)


def test_recommendation_request():
    service = create_service()

    result = service.process_message(
        "Find me a 3-bedroom home in Riverside under 500000"
    )

    assert result["intent"] == ConversationIntent.RECOMMEND_PROPERTIES.value
    assert "I found" in result["message"]
    assert isinstance(result["data"], list)
    assert len(result["data"]) <= 5


def test_recommendation_extracts_city():
    service = create_service()

    result = service.process_message(
        "Find homes in Riverside"
    )

    assert result["intent"] == ConversationIntent.RECOMMEND_PROPERTIES.value
    assert isinstance(result["data"], list)

    for property_data in result["data"]:
        assert property_data["City"].lower() == "riverside"


def test_recommendation_extracts_bedrooms():
    service = create_service()

    result = service.process_message(
        "Show me homes with at least 3 bedrooms"
    )

    assert result["intent"] == ConversationIntent.RECOMMEND_PROPERTIES.value

    for property_data in result["data"]:
        assert property_data["Bedrooms"] >= 3


def test_recommendation_extracts_property_type():
    service = create_service()

    result = service.process_message(
        "Show me condos"
    )

    assert result["intent"] == ConversationIntent.RECOMMEND_PROPERTIES.value

    for property_data in result["data"]:
        assert property_data["PropertyType"] == "Condo"


def test_recommendation_extracts_price():
    service = create_service()

    result = service.process_message(
        "Find homes under 400000"
    )

    assert result["intent"] == ConversationIntent.RECOMMEND_PROPERTIES.value

    for property_data in result["data"]:
        assert property_data["ListPrice"] <= 400000


def test_recommendation_extracts_combined_filters():
    service = create_service()

    result = service.process_message(
        "Find a townhouse in Riverside with 3 bedrooms "
        "and 2 bathrooms"
    )

    assert result["intent"] == ConversationIntent.RECOMMEND_PROPERTIES.value

    for property_data in result["data"]:
        assert property_data["City"].lower() == "riverside"
        assert property_data["PropertyType"] == "Townhouse"
        assert property_data["Bedrooms"] >= 3
        assert property_data["Bathrooms"] >= 2


def test_prediction_request_requires_features():
    service = create_service()

    result = service.process_message(
        "I want to estimate the value of a California property"
    )

    assert result["intent"] == ConversationIntent.PREDICT_PRICE.value

    assert "missing_features" in result["data"]

    assert set(result["data"]["missing_features"]) == {
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    }


def test_general_help():
    service = create_service()

    result = service.process_message(
        "What can you do?"
    )

    assert result["intent"] == ConversationIntent.GENERAL_HELP.value
    assert result["data"] is None
    assert "properties" in result["message"].lower()


def test_unknown_request():
    service = create_service()

    result = service.process_message(
        "Tell me a joke"
    )

    assert result["intent"] == ConversationIntent.UNKNOWN.value
    assert result["data"] is None


def test_empty_message():
    service = create_service()

    try:
        service.process_message("")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_follow_up_changes_city(
    monkeypatch,
):
    from src.conversation.service import (
        ConversationService,
    )

    service = ConversationService(
        listings_path="data/raw/housing_market.csv"
    )

    first = service.process_message(
        "Find me a house in Los Angeles under $800k.",
        session_id="city-follow-up",
    )

    assert first["intent"] == "recommend_properties"

    second = service.process_message(
        "What about San Diego?",
        session_id="city-follow-up",
    )

    assert second["intent"] == "recommend_properties"
    assert isinstance(
        second["data"],
        list,
    )


def test_follow_up_changes_property_type():
    from src.conversation.service import (
        ConversationService,
    )

    service = ConversationService(
        listings_path="data/raw/housing_market.csv"
    )

    service.process_message(
        "Find houses in Los Angeles under $800k.",
        session_id="type-follow-up",
    )

    result = service.process_message(
        "Only show condos.",
        session_id="type-follow-up",
    )

    assert result["intent"] == "recommend_properties"
    assert isinstance(
        result["data"],
        list,
    )

def test_conversation_follow_up_changes_city():
    from src.conversation.service import ConversationService

    service = ConversationService(
        listings_path="data/raw/housing_market.csv"
    )

    first = service.process_message(
        "Find me houses in Riverside under $800k.",
        session_id="e2e-city",
    )

    assert first["intent"] == "recommend_properties"
    assert isinstance(first["data"], list)

    second = service.process_message(
        "What about San Diego?",
        session_id="e2e-city",
    )

    assert second["intent"] == "recommend_properties"
    assert isinstance(second["data"], list)

def test_conversation_follow_up_changes_property_type():
    from src.conversation.service import ConversationService

    service = ConversationService(
        listings_path="data/raw/housing_market.csv"
    )

    first = service.process_message(
        "Find houses in Los Angeles under $800k.",
        session_id="e2e-property-type",
    )

    assert first["intent"] == "recommend_properties"

    second = service.process_message(
        "Only show condos.",
        session_id="e2e-property-type",
    )

    assert second["intent"] == "recommend_properties"
    assert isinstance(second["data"], list)

    for property_data in second["data"]:
        assert property_data["PropertyType"].lower() == "condo"

def test_conversation_follow_up_requests_cheaper_properties():
    from src.conversation.service import ConversationService

    service = ConversationService(
        listings_path="data/raw/housing_market.csv"
    )

    first = service.process_message(
        "Find houses in Los Angeles under $800k.",
        session_id="e2e-cheaper",
    )

    assert first["intent"] == "recommend_properties"

    second = service.process_message(
        "Show me cheaper ones.",
        session_id="e2e-cheaper",
    )

    assert second["intent"] == "recommend_properties"
    assert isinstance(second["data"], list)

    for property_data in second["data"]:
        assert property_data["ListPrice"] <= 720000

def test_conversation_selects_second_property():
    from src.conversation.service import ConversationService

    service = ConversationService(
        listings_path="data/raw/housing_market.csv"
    )

    first = service.process_message(
        "Find me houses in Riverside.",
        session_id="e2e-selection",
    )

    assert first["intent"] == "recommend_properties"
    assert len(first["data"]) >= 2

    expected = first["data"][1]

    second = service.process_message(
        "Show me the second property.",
        session_id="e2e-selection",
    )

    assert second["intent"] == "recommend_properties"
    assert second["data"] == expected
    assert expected["ListingID"] in second["message"]

def test_conversation_sessions_are_isolated():
    from src.conversation.service import ConversationService

    service = ConversationService(
        listings_path="data/raw/housing_market.csv"
    )

    service.process_message(
        "Find houses in Los Angeles.",
        session_id="user-a",
    )

    service.process_message(
        "Find houses in San Diego.",
        session_id="user-b",
    )

    state_a = service.conversation_states["user-a"]
    state_b = service.conversation_states["user-b"]

    assert state_a.last_parameters["city"] == "Los Angeles"
    assert state_b.last_parameters["city"] == "San Diego"

