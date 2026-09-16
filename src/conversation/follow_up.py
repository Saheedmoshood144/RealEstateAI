from __future__ import annotations

import re
from typing import Any

from src.conversation.schemas import (
    ConversationIntent,
    ParsedMessage,
)


class FollowUpResolver:
    """
    Resolves conversational follow-up requests using
    the previous conversation state.

    This component only interprets conversational context.
    It does not access listings or ML models.
    """

    FOLLOW_UP_PATTERNS = (
        r"\bwhat about\b",
        r"\bhow about\b",
        r"\bshow me\b",
        r"\bfind me\b",
        r"\bmake it\b",
        r"\bchange it\b",
        r"\bwith\b",
        r"\bunder\b",
        r"\bover\b",
        r"\binstead\b",
        r"\bonly\b",
        r"\bcheaper\b",
        r"\bcheap\b",
        r"\bmore expensive\b",
        r"\bmore affordable\b",
        r"\bmore bedrooms?\b",
        r"\bfewer bedrooms?\b",
        r"\bsecond\b",
        r"\bthird\b",
        r"\bfirst\b",
    )

    ORDINALS = {
        "first": 1,
        "1st": 1,
        "second": 2,
        "2nd": 2,
        "third": 3,
        "3rd": 3,
        "fourth": 4,
        "4th": 4,
        "fifth": 5,
        "5th": 5,
    }

    @classmethod
    def looks_like_follow_up(cls, message: str) -> bool:
        normalized = message.lower().strip()

        return any(
            re.search(pattern, normalized)
            for pattern in cls.FOLLOW_UP_PATTERNS
        )

    @classmethod
    def get_ordinal(cls, message: str) -> int | None:
        normalized = message.lower().strip()

        for word, number in cls.ORDINALS.items():
            pattern = rf"\b{re.escape(word)}\b"

            if re.search(pattern, normalized):
                return number

        return None

    @classmethod
    def is_listing_selection(cls, message: str) -> bool:
        return cls.get_ordinal(message) is not None

    @staticmethod
    def _price_from_recommendations(
        recommendations: list[dict[str, Any]],
    ) -> float | None:
        prices = []

        for recommendation in recommendations:
            price = recommendation.get("ListPrice")

            if isinstance(price, (int, float)):
                prices.append(float(price))

        if not prices:
            return None

        return sum(prices) / len(prices)

    @classmethod
    def resolve_relative_price(
        cls,
        message: str,
        previous_parameters: dict[str, Any],
        previous_recommendations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Converts relative price language such as
        'cheaper' into a deterministic price constraint.
        """

        normalized = message.lower()

        if not any(
            phrase in normalized
            for phrase in (
                "cheaper",
                "cheap",
                "more affordable",
            )
        ):
            return {}

        previous_max_price = previous_parameters.get(
            "max_price"
        )

        if previous_max_price is not None:
            return {
                "max_price": round(
                    float(previous_max_price) * 0.90
                )
            }

        average_price = cls._price_from_recommendations(
            previous_recommendations
        )

        if average_price is not None:
            return {
                "max_price": round(
                    average_price
                )
            }

        return {}

    @classmethod
    def merge(
        cls,
        previous_parameters: dict[str, Any],
        new_parsed: ParsedMessage,
        previous_recommendations: list[dict[str, Any]]
        | None = None,
    ) -> ParsedMessage:
        """
        Combines previous parameters with newly extracted
        parameters.

        New explicit parameters always override previous
        parameters.
        """

        previous_recommendations = (
            previous_recommendations or []
        )

        merged_parameters = previous_parameters.copy()

        merged_parameters.update(
            new_parsed.parameters
        )

        relative_price = cls.resolve_relative_price(
            new_parsed.original_message,
            previous_parameters,
            previous_recommendations,
        )

        merged_parameters.update(
            relative_price
        )

        return ParsedMessage(
            intent=(
                new_parsed.intent
                if new_parsed.intent
                != ConversationIntent.UNKNOWN
                else ConversationIntent.RECOMMEND_PROPERTIES
            ),
            parameters=merged_parameters,
            confidence=new_parsed.confidence,
            original_message=new_parsed.original_message,
        )

    @classmethod
    def select_listing(
        cls,
        message: str,
        recommendations: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        Selects a recommendation using ordinal language.

        Example:
            'show me the second property'
        """

        ordinal = cls.get_ordinal(message)

        if ordinal is None:
            return None

        index = ordinal - 1

        if index < 0 or index >= len(recommendations):
            return None

        return recommendations[index]