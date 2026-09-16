from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from src.conversation.follow_up import FollowUpResolver
from src.conversation.hybrid_parser import HybridParser
from src.conversation.provider import create_llm_fallback
from src.conversation.response_formatter import (
    ConversationResponseFormatter,
)
from src.conversation.schemas import (
    ConversationIntent,
    ParsedMessage,
)
from src.conversation.state import ConversationState
from src.models.recommendation_engine import RecommendationEngine


class ConversationService:
    """
    Orchestrates the RealEstateAI conversational layer.

    The service:
    1. Parses the user's natural-language message.
    2. Determines the user's intent.
    3. Routes recommendation requests to the recommendation engine.
    4. Routes prediction requests to the registered MLflow model.
    5. Formats results into conversational responses.
    6. Handles general help and unknown requests.
    7. Maintains conversation state for contextual follow-ups.
    8. Handles selection of properties from previous recommendations.

    The LLM is used only for understanding ambiguous requests.
    It does not directly access listings or make predictions.
    """

    PREDICTION_FEATURES = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]

    MODEL_URI = "models:/CaliforniaHousingRandomForest/1"

    def __init__(
        self,
        listings_path: str | Path,
    ):
        """
        Initialize the conversational service.

        Parameters
        ----------
        listings_path:
            Path to the housing listings CSV.
        """

        self.parser = HybridParser(
            llm_fallback=create_llm_fallback()
        )

        self.recommendation_engine = RecommendationEngine(
            data_path=listings_path
        )

        self.model = mlflow.pyfunc.load_model(
            self.MODEL_URI
        )

        self.response_formatter = (
            ConversationResponseFormatter()
        )

        self.conversation_states: dict[
            str,
            ConversationState,
        ] = {}

    def process_message(
        self,
        message: str,
        session_id: str = "default",
    ) -> dict[str, Any]:
        """
        Process a user's conversational message.

        Handles:
        - New property searches.
        - Contextual follow-up requests.
        - Listing selection from previous recommendations.
        - Price prediction requests.
        - Help requests.
        - Unknown requests.
        """

        state = self.conversation_states.setdefault(
            session_id,
            ConversationState(),
        )

        parsed = self.parser.parse(message)

        # Handle requests such as:
        # "show me the second property"
        if (
            state.last_recommendations
            and FollowUpResolver.is_listing_selection(
                message
            )
        ):
            return self._handle_listing_selection(
                message,
                state,
            )

        # Resolve contextual follow-up requests.
        if (
            state.last_parameters
            and (
                FollowUpResolver.looks_like_follow_up(
                    message
                )
                or bool(parsed.parameters)
            )
            and parsed.intent
            in (
                ConversationIntent.RECOMMEND_PROPERTIES,
                ConversationIntent.UNKNOWN,
            )
        ):
            parsed = FollowUpResolver.merge(
                state.last_parameters,
                parsed,
                state.last_recommendations,
            )

        if parsed.intent == ConversationIntent.RECOMMEND_PROPERTIES:
            response = self._handle_recommendation(
                parsed
            )

        elif parsed.intent == ConversationIntent.PREDICT_PRICE:
            response = self._handle_prediction(
                parsed
            )

        elif parsed.intent == ConversationIntent.GENERAL_HELP:
            response = self._handle_help(
                parsed
            )

        else:
            response = self._handle_unknown(
                parsed
            )

        self._update_state(
            state,
            parsed,
            response,
        )

        return response

    def _handle_listing_selection(
        self,
        message: str,
        state: ConversationState,
    ) -> dict[str, Any]:
        """
        Return details for a property selected from the
        previous recommendation results.
        """

        selected = FollowUpResolver.select_listing(
            message,
            state.last_recommendations,
        )

        if selected is None:
            return {
                "intent": (
                    ConversationIntent.RECOMMEND_PROPERTIES.value
                ),
                "message": (
                    "I couldn't find that property in the "
                    "current recommendations. Please choose "
                    "a valid property number."
                ),
                "data": None,
            }

        listing_id = selected.get(
            "ListingID",
            "N/A",
        )

        city = selected.get(
            "City",
            "Unknown location",
        )

        property_type = selected.get(
            "PropertyType",
            "Property",
        )

        price = selected.get(
            "ListPrice"
        )

        bedrooms = selected.get(
            "Bedrooms"
        )

        bathrooms = selected.get(
            "Bathrooms"
        )

        sqft = selected.get(
            "SqFt"
        )

        details = []

        if bedrooms is not None:
            details.append(
                f"{bedrooms} bedrooms"
            )

        if bathrooms is not None:
            details.append(
                f"{bathrooms} bathrooms"
            )

        if sqft is not None:
            details.append(
                f"{sqft:,} sq ft"
            )

        message_lines = [
            f"Here are the details for listing {listing_id}:",
            "",
            f"{property_type} in {city}",
        ]

        if details:
            message_lines.append(
                " • ".join(details)
            )

        if price is not None:
            message_lines.append(
                f"Price: ${price:,.0f}"
            )

        return {
            "intent": ConversationIntent.RECOMMEND_PROPERTIES.value,
            "message": "\n".join(message_lines),
            "data": selected,
        }

    def _update_state(
        self,
        state: ConversationState,
        parsed: ParsedMessage,
        response: dict[str, Any],
    ) -> None:
        """
        Update the conversation state after processing a message.
        """

        recommendations = response.get("data")

        if not isinstance(recommendations, list):
            recommendations = None

        state.update(
            intent=parsed.intent.value,
            parameters=parsed.parameters,
            recommendations=recommendations,
        )

    def _handle_recommendation(
        self,
        parsed: ParsedMessage,
    ) -> dict[str, Any]:
        """
        Handle property recommendation requests.

        The recommendation engine remains the source of truth
        for property matching and ranking.
        """

        results = self.recommendation_engine.recommend(
            city=parsed.parameters.get("city"),
            neighborhood=parsed.parameters.get("neighborhood"),
            property_type=parsed.parameters.get("property_type"),
            condition=parsed.parameters.get("condition"),
            min_bedrooms=parsed.parameters.get("min_bedrooms"),
            min_bathrooms=parsed.parameters.get("min_bathrooms"),
            max_price=parsed.parameters.get("max_price"),
            min_sqft=parsed.parameters.get("min_sqft"),
            max_sqft=parsed.parameters.get("max_sqft"),
            top_n=5,
        )

        if results.empty:
            return {
                "intent": parsed.intent.value,
                "message": (
                    self.response_formatter.format_recommendations(
                        parsed.parameters,
                        [],
                    )
                ),
                "data": [],
            }

        output_columns = [
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
        ]

        available_columns = [
            column
            for column in output_columns
            if column in results.columns
        ]

        recommendations = results[
            available_columns
        ].copy()

        if "RecommendationScore" in recommendations.columns:
            recommendations["RecommendationScore"] = (
                recommendations["RecommendationScore"].round(2)
            )

        records = recommendations.to_dict(
            orient="records"
        )

        message = (
            self.response_formatter.format_recommendations(
                parsed.parameters,
                records,
            )
        )

        return {
            "intent": parsed.intent.value,
            "message": message,
            "data": records,
        }

    def _handle_prediction(
        self,
        parsed: ParsedMessage,
    ) -> dict[str, Any]:
        """
        Handle property-price prediction requests.

        The prediction model requires eight numerical features.
        If they are not available, ask the user for them instead
        of attempting an invalid prediction.
        """

        provided = parsed.parameters

        missing_features = [
            feature
            for feature in self.PREDICTION_FEATURES
            if feature not in provided
        ]

        if missing_features:
            feature_descriptions = {
                "MedInc": "median income",
                "HouseAge": "house age",
                "AveRooms": "average number of rooms",
                "AveBedrms": "average number of bedrooms",
                "Population": "population",
                "AveOccup": "average household occupancy",
                "Latitude": "latitude",
                "Longitude": "longitude",
            }

            missing_descriptions = [
                feature_descriptions[feature]
                for feature in missing_features
            ]

            message = (
                self.response_formatter
                .format_prediction_missing_features(
                    missing_descriptions
                )
            )

            return {
                "intent": parsed.intent.value,
                "message": message,
                "data": {
                    "missing_features": missing_features,
                },
            }

        input_data = pd.DataFrame(
            [
                {
                    feature: provided[feature]
                    for feature in self.PREDICTION_FEATURES
                }
            ]
        )

        prediction = self.model.predict(input_data)[0]

        prediction_value = float(prediction)

        return {
            "intent": parsed.intent.value,
            "message": (
                "Based on the supplied property data, "
                "the estimated median house value is "
                f"${prediction_value * 100000:,.2f}."
            ),
            "data": {
                "predicted_house_value": round(
                    prediction_value,
                    4,
                ),
            },
        }

    def _handle_help(
        self,
        parsed: ParsedMessage,
    ) -> dict[str, Any]:
        """
        Return information about what the assistant can do.
        """

        return {
            "intent": parsed.intent.value,
            "message": self.response_formatter.format_help(),
            "data": None,
        }

    def _handle_unknown(
        self,
        parsed: ParsedMessage,
    ) -> dict[str, Any]:
        """
        Handle messages whose intent cannot be determined.
        """

        return {
            "intent": parsed.intent.value,
            "message": self.response_formatter.format_unknown(),
            "data": None,
        }