from src.conversation.response_formatter import (
    ConversationResponseFormatter,
)


def test_recommendation_formatter_includes_count():
    formatter = ConversationResponseFormatter()

    records = [
        {
            "ListingID": "L001",
            "City": "Los Angeles",
            "ListPrice": 700000,
        },
        {
            "ListingID": "L002",
            "City": "Los Angeles",
            "ListPrice": 750000,
        },
    ]

    message = formatter.format_recommendations(
        {
            "city": "Los Angeles",
            "max_price": 800000,
        },
        records,
    )

    assert "I found 2 properties" in message


def test_recommendation_formatter_includes_filters():
    formatter = ConversationResponseFormatter()

    message = formatter.format_recommendations(
        {
            "city": "Los Angeles",
            "min_bedrooms": 3,
            "max_price": 800000,
        },
        [
            {
                "ListingID": "L001",
                "City": "Los Angeles",
            }
        ],
    )

    assert "City: Los Angeles" in message
    assert "Bedrooms: 3+" in message
    assert "Maximum price: $800,000" in message


def test_recommendation_formatter_handles_no_results():
    formatter = ConversationResponseFormatter()

    message = formatter.format_recommendations(
        {
            "city": "Los Angeles",
        },
        [],
    )

    assert "couldn't find any properties" in message


def test_help_formatter():
    formatter = ConversationResponseFormatter()

    message = formatter.format_help()

    assert "real-estate" in message.lower()
    assert "properties" in message.lower()


def test_unknown_formatter():
    formatter = ConversationResponseFormatter()

    message = formatter.format_unknown()

    assert "real-estate assistant" in message.lower()


def test_prediction_missing_features_formatter():
    formatter = ConversationResponseFormatter()

    message = (
        formatter.format_prediction_missing_features(
            [
                "median income",
                "house age",
                "latitude",
            ]
        )
    )

    assert "median income" in message
    assert "house age" in message
    assert "latitude" in message

def test_recommendation_formatter_includes_property_details():
    formatter = ConversationResponseFormatter()

    message = formatter.format_recommendations(
        {
            "city": "Los Angeles",
        },
        [
            {
                "ListingID": "L001",
                "City": "Los Angeles",
                "Neighborhood": "Downtown",
                "PropertyType": "House",
                "Bedrooms": 3,
                "Bathrooms": 2,
                "SqFt": 1800,
                "ListPrice": 750000,
                "RecommendationScore": 0.91,
            }
        ],
    )

    assert "Top recommendations:" in message
    assert "House in Downtown, Los Angeles" in message
    assert "3 bd" in message
    assert "2 ba" in message
    assert "1,800 sq ft" in message
    assert "$750,000" in message
    assert "0.91" in message
    assert "L001" in message