from __future__ import annotations

from typing import Any

from src.conversation.schemas import ConversationIntent


class ConversationResponseFormatter:
    """
    Converts structured conversation results into
    natural, user-friendly responses.

    This class does not make ML predictions or property
    recommendations. It only formats results produced
    by the application services.
    """

    @staticmethod
    def format_recommendations(
        parameters: dict[str, Any],
        recommendations: list[dict[str, Any]],
    ) -> str:
        """
        Create a conversational response for property
        recommendations.
        """

        if not recommendations:
            return (
                "I couldn't find any properties matching "
                "your current criteria. Try relaxing one "
                "or more filters such as price, bedrooms, "
                "property type, or location."
            )

        count = len(recommendations)

        message = (
            f"I found {count} "
            f"{'property' if count == 1 else 'properties'} "
            "matching your request."
        )

        filter_lines = []

        if parameters.get("city"):
            filter_lines.append(
                f"City: {parameters['city']}"
            )

        if parameters.get("neighborhood"):
            filter_lines.append(
                f"Neighborhood: {parameters['neighborhood']}"
            )

        if parameters.get("property_type"):
            filter_lines.append(
                f"Property type: {parameters['property_type']}"
            )

        if parameters.get("condition"):
            filter_lines.append(
                f"Condition: {parameters['condition']}"
            )

        if parameters.get("min_bedrooms") is not None:
            filter_lines.append(
                f"Bedrooms: {parameters['min_bedrooms']}+"
            )

        if parameters.get("min_bathrooms") is not None:
            filter_lines.append(
                f"Bathrooms: {parameters['min_bathrooms']}+"
            )

        if parameters.get("max_price") is not None:
            filter_lines.append(
                "Maximum price: "
                f"${parameters['max_price']:,.0f}"
            )

        if parameters.get("min_sqft") is not None:
            filter_lines.append(
                f"Minimum size: "
                f"{parameters['min_sqft']:,} sq ft"
            )

        if parameters.get("max_sqft") is not None:
            filter_lines.append(
                f"Maximum size: "
                f"{parameters['max_sqft']:,} sq ft"
            )

        if filter_lines:
            message += "\n\nFilters applied:\n"
            message += "\n".join(
                f"• {line}" for line in filter_lines
            )

        message += "\n\nTop recommendations:\n"

        for index, property_data in enumerate(
            recommendations,
            start=1,
        ):
            listing_id = property_data.get(
                "ListingID",
                "N/A",
            )

            city = property_data.get(
                "City",
                "Unknown location",
            )

            neighborhood = property_data.get(
                "Neighborhood"
            )

            property_type = property_data.get(
                "PropertyType",
                "Property",
            )

            condition = property_data.get(
                "Condition"
            )

            bedrooms = property_data.get(
                "Bedrooms"
            )

            bathrooms = property_data.get(
                "Bathrooms"
            )

            sqft = property_data.get(
                "SqFt"
            )

            price = property_data.get(
                "ListPrice"
            )

            score = property_data.get(
                "RecommendationScore"
            )

            location = city

            if neighborhood:
                location = (
                    f"{neighborhood}, {city}"
                )

            details = []

            if bedrooms is not None:
                details.append(
                    f"{bedrooms} bd"
                )

            if bathrooms is not None:
                details.append(
                    f"{bathrooms} ba"
                )

            if sqft is not None:
                details.append(
                    f"{sqft:,} sq ft"
                )

            detail_text = " • ".join(details)

            message += (
                f"\n{index}. {property_type} "
                f"in {location}\n"
            )

            if detail_text:
                message += f"   {detail_text}\n"

            if price is not None:
                message += (
                    f"   Price: "
                    f"${price:,.0f}\n"
                )

            if condition:
                message += (
                    f"   Condition: {condition}\n"
                )

            if score is not None:
                message += (
                    f"   Recommendation score: "
                    f"{score:.2f}\n"
                )

            message += (
                f"   Listing ID: {listing_id}\n"
            )

        message += (
            "\nI've ranked these properties using "
            "the recommendation engine."
        )

        return message

    @staticmethod
    def format_help() -> str:
        """
        Return the assistant capability description.
        """

        return (
            "I can help you with California real-estate "
            "search and analysis.\n\n"
            "You can ask me to:\n"
            "• Find properties based on location, price, "
            "bedrooms, bathrooms, size, type, or condition.\n"
            "• Get ranked property recommendations.\n"
            "• Estimate a property's value when the required "
            "California housing model features are available.\n\n"
            "For example:\n"
            "\"Find me a 3-bedroom house in Los Angeles "
            "under $800k.\""
        )

    @staticmethod
    def format_unknown() -> str:
        """
        Return a helpful response for unsupported requests.
        """

        return (
            "I'm a real-estate assistant focused on "
            "California property search and valuation. "
            "Try asking me to find properties, recommend "
            "homes, or estimate a property's value."
        )

    @staticmethod
    def format_prediction_missing_features(
        missing_features: list[str],
    ) -> str:
        """
        Explain which model inputs are still required
        for a property-value prediction.
        """

        if not missing_features:
            return (
                "I have the information needed to estimate "
                "the property's value."
            )

        formatted = "\n".join(
            f"• {feature}"
            for feature in missing_features
        )

        return (
            "I can estimate the property's value, but I "
            "still need the following information:\n\n"
            f"{formatted}"
        )