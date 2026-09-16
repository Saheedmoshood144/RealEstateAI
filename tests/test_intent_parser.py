from src.conversation.intent_parser import IntentParser
from src.conversation.schemas import ConversationIntent


parser = IntentParser()


def test_townhouse_synonym():
    result = parser.parse(
        "I want a townhome in Riverside"
    )

    assert result.intent == ConversationIntent.RECOMMEND_PROPERTIES
    assert result.parameters["property_type"] == "Townhouse"
    assert result.parameters["city"] == "Riverside"


def test_condo_synonym():
    result = parser.parse(
        "Show me condominiums in Riverside"
    )

    assert result.intent == ConversationIntent.RECOMMEND_PROPERTIES
    assert result.parameters["property_type"] == "Condo"


def test_single_family_synonym():
    result = parser.parse(
        "Find me a family home in Riverside"
    )

    assert result.intent == ConversationIntent.RECOMMEND_PROPERTIES
    assert result.parameters["property_type"] == "Single Family"


def test_bed_synonym():
    result = parser.parse(
        "Find a townhouse in Riverside with 3 beds"
    )

    assert result.parameters["min_bedrooms"] == 3


def test_bath_synonym():
    result = parser.parse(
        "Find a townhouse in Riverside with 2 baths"
    )

    assert result.parameters["min_bathrooms"] == 2.0


def test_thousand_price():
    result = parser.parse(
        "Find homes in Riverside under 500 thousand"
    )

    assert result.parameters["max_price"] == 500000


def test_k_price():
    result = parser.parse(
        "Find homes in Riverside under $500k"
    )

    assert result.parameters["max_price"] == 500000


def test_million_price():
    result = parser.parse(
        "Find homes in Riverside below $1.2 million"
    )

    assert result.parameters["max_price"] == 1200000


def test_budget_price():
    result = parser.parse(
        "I need a house in Riverside with a budget of $450k"
    )

    assert result.parameters["max_price"] == 450000


def test_sqft_synonym():
    result = parser.parse(
        "Find homes in Riverside with at least 1500 sqft"
    )

    assert result.parameters["min_sqft"] == 1500


def test_square_feet():
    result = parser.parse(
        "Find homes with at least 1800 square feet"
    )

    assert result.parameters["min_sqft"] == 1800


def test_sqft_range():
    result = parser.parse(
        "Find homes between 1500 and 2000 sqft"
    )

    assert result.parameters["min_sqft"] == 1500
    assert result.parameters["max_sqft"] == 2000


def test_prediction_intent():
    result = parser.parse(
        "Can you estimate the value of this property?"
    )

    assert result.intent == ConversationIntent.PREDICT_PRICE


def test_help_intent():
    result = parser.parse(
        "What can you do?"
    )

    assert result.intent == ConversationIntent.GENERAL_HELP


def test_unknown_intent():
    result = parser.parse(
        "The weather is nice today"
    )

    assert result.intent == ConversationIntent.UNKNOWN


def test_combined_natural_language_request():
    result = parser.parse(
        "I'm looking for a 3 bed townhouse around Riverside "
        "with 2 baths under $400k and at least 1500 sqft"
    )

    assert result.intent == ConversationIntent.RECOMMEND_PROPERTIES

    assert result.parameters["city"] == "Riverside"
    assert result.parameters["property_type"] == "Townhouse"
    assert result.parameters["min_bedrooms"] == 3
    assert result.parameters["min_bathrooms"] == 2.0
    assert result.parameters["max_price"] == 400000
    assert result.parameters["min_sqft"] == 1500