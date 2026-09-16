from src.conversation.follow_up import (
    FollowUpResolver,
)
from src.conversation.schemas import (
    ConversationIntent,
    ParsedMessage,
)


def test_follow_up_detection():
    assert FollowUpResolver.looks_like_follow_up(
        "What about 4 bedrooms?"
    )


def test_non_follow_up_detection():
    assert not FollowUpResolver.looks_like_follow_up(
        "Find houses in Los Angeles"
    )


def test_follow_up_merges_previous_parameters():
    previous = {
        "city": "Los Angeles",
        "max_price": 800000,
        "min_bedrooms": 3,
    }

    parsed = ParsedMessage(
        intent=ConversationIntent.RECOMMEND_PROPERTIES,
        parameters={
            "min_bedrooms": 4,
        },
        confidence=0.95,
        original_message="What about 4 bedrooms?",
    )

    result = FollowUpResolver.merge(
        previous,
        parsed,
    )

    assert result.parameters["city"] == "Los Angeles"
    assert result.parameters["max_price"] == 800000
    assert result.parameters["min_bedrooms"] == 4


def test_follow_up_preserves_new_intent():
    previous = {
        "city": "Los Angeles",
    }

    parsed = ParsedMessage(
        intent=ConversationIntent.PREDICT_PRICE,
        parameters={},
        confidence=0.90,
        original_message="Estimate the value",
    )

    result = FollowUpResolver.merge(
        previous,
        parsed,
    )

    assert result.intent == ConversationIntent.PREDICT_PRICE

def test_cheaper_follow_up_reduces_previous_max_price():
    previous = {
        "city": "Los Angeles",
        "max_price": 800000,
    }

    parsed = ParsedMessage(
        intent=ConversationIntent.RECOMMEND_PROPERTIES,
        parameters={},
        confidence=0.90,
        original_message="Show me cheaper ones.",
    )

    result = FollowUpResolver.merge(
        previous,
        parsed,
        [],
    )

    assert result.parameters["city"] == "Los Angeles"
    assert result.parameters["max_price"] == 720000


def test_cheaper_follow_up_uses_average_listing_price():
    previous = {
        "city": "Los Angeles",
    }

    recommendations = [
        {"ListPrice": 600000},
        {"ListPrice": 700000},
        {"ListPrice": 800000},
    ]

    parsed = ParsedMessage(
        intent=ConversationIntent.RECOMMEND_PROPERTIES,
        parameters={},
        confidence=0.90,
        original_message="Show me cheaper ones.",
    )

    result = FollowUpResolver.merge(
        previous,
        parsed,
        recommendations,
    )

    assert result.parameters["max_price"] == 700000


def test_city_follow_up_overrides_previous_city():
    previous = {
        "city": "Los Angeles",
        "max_price": 800000,
        "min_bedrooms": 3,
    }

    parsed = ParsedMessage(
        intent=ConversationIntent.RECOMMEND_PROPERTIES,
        parameters={
            "city": "San Diego",
        },
        confidence=0.95,
        original_message="What about San Diego?",
    )

    result = FollowUpResolver.merge(
        previous,
        parsed,
        [],
    )

    assert result.parameters["city"] == "San Diego"
    assert result.parameters["max_price"] == 800000
    assert result.parameters["min_bedrooms"] == 3


def test_property_type_follow_up_overrides_previous_type():
    previous = {
        "city": "Los Angeles",
        "property_type": "House",
        "max_price": 800000,
    }

    parsed = ParsedMessage(
        intent=ConversationIntent.RECOMMEND_PROPERTIES,
        parameters={
            "property_type": "Condo",
        },
        confidence=0.95,
        original_message="Only show condos.",
    )

    result = FollowUpResolver.merge(
        previous,
        parsed,
        [],
    )

    assert result.parameters["property_type"] == "Condo"
    assert result.parameters["city"] == "Los Angeles"
    assert result.parameters["max_price"] == 800000


def test_second_listing_is_selected():
    recommendations = [
        {"ListingID": "L001"},
        {"ListingID": "L002"},
        {"ListingID": "L003"},
    ]

    selected = FollowUpResolver.select_listing(
        "Show me the second property.",
        recommendations,
    )

    assert selected == {
        "ListingID": "L002"
    }


def test_out_of_range_listing_returns_none():
    recommendations = [
        {"ListingID": "L001"},
        {"ListingID": "L002"},
    ]

    selected = FollowUpResolver.select_listing(
        "Show me the fifth property.",
        recommendations,
    )

    assert selected is None