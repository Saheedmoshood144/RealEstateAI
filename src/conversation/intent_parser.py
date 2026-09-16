import re

from src.conversation.schemas import ConversationIntent, ParsedMessage


class IntentParser:
    """
    Lightweight natural-language intent parser for RealEstateAI.

    The parser identifies the user's main real-estate intent and
    extracts common property-search parameters.
    """

    RECOMMENDATION_KEYWORDS = {
        "find",
        "show",
        "search",
        "recommend",
        "recommendation",
        "recommendations",
        "looking for",
        "look for",
        "help me find",
        "i need",
        "i want",
        "want a",
        "want an",
        "homes",
        "houses",
        "properties",
        "property",
        "listings",
        "listing",
        "available",
        "buy",
        "buying",
        "purchase",
        "purchasing",
    }

    PREDICTION_KEYWORDS = {
        "predict",
        "prediction",
        "estimate",
        "estimated",
        "estimation",
        "valuation",
        "value",
        "worth",
        "price prediction",
        "house price",
        "property value",
        "property price",
        "home value",
        "home price",
    }

    HELP_KEYWORDS = {
        "help",
        "what can you do",
        "how does this work",
        "how can you help",
        "what do you offer",
    }

    # Longer and more specific phrases are handled first by
    # _extract_property_type().
    PROPERTY_TYPES = {
        "single family home": "Single Family",
        "single-family home": "Single Family",
        "family home": "Single Family",
        "family house": "Single Family",
        "single family": "Single Family",
        "single-family": "Single Family",
        "townhouse": "Townhouse",
        "townhome": "Townhouse",
        "condominiums": "Condo",
        "condominium": "Condo",
        "condos": "Condo",
        "condo": "Condo",
        "house": "Single Family",
        "home": "Single Family",
    }

    CONDITIONS = {
        "excellent": "Excellent",
        "good": "Good",
        "fair": "Fair",
        "poor": "Poor",
    }

    def parse(self, message: str) -> ParsedMessage:
        """
        Parse a natural-language user message.

        Parameters
        ----------
        message : str
            User's natural-language request.

        Returns
        -------
        ParsedMessage
            Structured interpretation of the message.
        """

        if not isinstance(message, str):
            raise TypeError("message must be a string.")

        message = message.strip()

        if not message:
            raise ValueError("message cannot be empty.")

        normalized = message.lower()

        intent, confidence = self._detect_intent(normalized)

        parameters = {}

        if intent == ConversationIntent.RECOMMEND_PROPERTIES:
            parameters = self._extract_recommendation_parameters(
                normalized
            )

        elif intent == ConversationIntent.PREDICT_PRICE:
            parameters = self._extract_prediction_parameters(
                normalized
            )

        return ParsedMessage(
            intent=intent,
            parameters=parameters,
            confidence=confidence,
            original_message=message,
        )

    def _detect_intent(
        self,
        message: str,
    ) -> tuple[ConversationIntent, float]:
        """
        Detect the user's intent using keyword matching.
        """

        recommendation_matches = sum(
            1
            for keyword in self.RECOMMENDATION_KEYWORDS
            if keyword in message
        )

        prediction_matches = sum(
            1
            for keyword in self.PREDICTION_KEYWORDS
            if keyword in message
        )

        help_matches = sum(
            1
            for keyword in self.HELP_KEYWORDS
            if keyword in message
        )

        # Help intent
        if help_matches > 0:
            confidence = min(
                0.80 + (help_matches - 1) * 0.05,
                0.95,
            )
            return ConversationIntent.GENERAL_HELP, confidence

        # Prediction takes priority when prediction keywords
        # are equal to or greater than recommendation keywords.
        if prediction_matches > 0:
            confidence = min(
                0.70 + (prediction_matches - 1) * 0.10,
                0.95,
            )

            if prediction_matches >= recommendation_matches:
                return ConversationIntent.PREDICT_PRICE, confidence

        # Recommendation intent
        if recommendation_matches > 0:
            confidence = min(
                0.70 + (recommendation_matches - 1) * 0.05,
                0.95,
            )
            return ConversationIntent.RECOMMEND_PROPERTIES, confidence

        return ConversationIntent.UNKNOWN, 0.0

    def _extract_recommendation_parameters(
        self,
        message: str,
    ) -> dict:
        """
        Extract property-search parameters from a recommendation request.
        """

        parameters = {}

        city = self._extract_city(message)
        if city:
            parameters["city"] = city

        neighborhood = self._extract_neighborhood(message)
        if neighborhood:
            parameters["neighborhood"] = neighborhood

        property_type = self._extract_property_type(message)
        if property_type:
            parameters["property_type"] = property_type

        condition = self._extract_condition(message)
        if condition:
            parameters["condition"] = condition

        bedrooms = self._extract_bedrooms(message)
        if bedrooms is not None:
            parameters["min_bedrooms"] = bedrooms

        bathrooms = self._extract_bathrooms(message)
        if bathrooms is not None:
            parameters["min_bathrooms"] = bathrooms

        max_price = self._extract_max_price(message)
        if max_price is not None:
            parameters["max_price"] = max_price

        min_sqft, max_sqft = self._extract_sqft(message)

        if min_sqft is not None:
            parameters["min_sqft"] = min_sqft

        if max_sqft is not None:
            parameters["max_sqft"] = max_sqft

        return parameters

    def _extract_prediction_parameters(
        self,
        message: str,
    ) -> dict:
        """
        Extract prediction-related parameters.

        The prediction model currently requires eight numerical
        California Housing features. Natural-language extraction
        for these features will be implemented later.
        """

        parameters = {}

        # Placeholder for the prediction feature extraction layer.
        #
        # The eight model features are:
        #
        # MedInc
        # HouseAge
        # AveRooms
        # AveBedrms
        # Population
        # AveOccup
        # Latitude
        # Longitude

        return parameters

    def _extract_city(
        self,
        message: str,
    ) -> str | None:
        """
        Extract a city from common search phrasing.

        Example:
            "Find homes in Riverside"
            -> Riverside
        """

        match = re.search(
            r"\b(?:in|near|around)\s+([A-Za-z][A-Za-z\s-]{1,40})",
            message,
        )

        if not match:
            return None

        city = match.group(1).strip()

        # Remove common trailing search phrases.
        city = re.split(
            r"\b(?:under|below|with|for|and|having)\b",
            city,
            maxsplit=1,
        )[0].strip()

        if not city:
            return None

        return city.title()

    def _extract_neighborhood(
        self,
        message: str,
    ) -> str | None:
        """
        Extract a neighborhood when explicitly requested.

        Example:
            "Find homes in the Eastside neighborhood"
            -> Eastside
        """

        match = re.search(
            r"\b(?:neighborhood|area)\s+([A-Za-z][A-Za-z\s-]{1,40})",
            message,
        )

        if not match:
            return None

        neighborhood = match.group(1).strip()

        neighborhood = re.split(
            r"\b(?:under|below|with|for|and|having)\b",
            neighborhood,
            maxsplit=1,
        )[0].strip()

        return neighborhood.title() if neighborhood else None

    def _extract_property_type(
        self,
        message: str,
    ) -> str | None:
        """
        Extract the requested property type.

        Matching is case-insensitive and uses word boundaries so
        that an alias cannot accidentally match inside another word.

        Examples:
            "townhome" -> "Townhouse"
            "townhouse" -> "Townhouse"
            "family home" -> "Single Family"
            "single-family home" -> "Single Family"
            "condo" -> "Condo"
        """

        # Check longer phrases first. This prevents a generic alias
        # such as "home" from winning over "family home".
        property_types = sorted(
            self.PROPERTY_TYPES.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        )

        for phrase, property_type in property_types:
            pattern = rf"\b{re.escape(phrase)}\b"

            if re.search(pattern, message):
                return property_type

        return None

    def _extract_condition(
        self,
        message: str,
    ) -> str | None:
        """
        Extract requested property condition.
        """

        for phrase, condition in self.CONDITIONS.items():
            pattern = rf"\b{re.escape(phrase)}\b"

            if re.search(pattern, message):
                return condition

        return None

    def _extract_bedrooms(
        self,
        message: str,
    ) -> int | None:
        """
        Extract minimum bedroom requirement.

        Supported examples:
            "3 bedroom"
            "3 bedrooms"
            "3 bed"
            "3 beds"
            "at least 3 bedrooms"
            "minimum 3 beds"
        """

        patterns = [
            r"\b(?:at least|minimum|min)\s+(\d+)\s+"
            r"(?:bedrooms?|beds?)\b",
            r"\b(\d+)\s*[- ]?\s*"
            r"(?:bedrooms?|beds?)\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, message)

            if match:
                return int(match.group(1))

        return None

    def _extract_bathrooms(
        self,
        message: str,
    ) -> float | None:
        """
        Extract minimum bathroom requirement.

        Supported examples:
            "2 bathrooms"
            "2 bath"
            "2 baths"
            "2.5 bathrooms"
            "2.5 bath"
            "at least 2 bathrooms"
        """

        patterns = [
            r"\b(?:at least|minimum|min)\s+"
            r"(\d+(?:\.\d+)?)\s+"
            r"(?:bathrooms?|baths?)\b",
            r"\b(\d+(?:\.\d+)?)\s*[- ]?\s*"
            r"(?:bathrooms?|baths?)\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, message)

            if match:
                return float(match.group(1))

        return None

    def _extract_max_price(
        self,
        message: str,
    ) -> float | None:
        """
        Extract maximum property price.

        Supported examples:
            "under $500,000"
            "below $500k"
            "less than 500k"
            "up to $500,000"
            "max $500000"
            "budget of $500k"
            "500k or less"
        """

        patterns = [
            (
                r"\b(?:under|below|less than|up to|max(?:imum)?)\s*"
                r"\$?\s*([\d,.]+)\s*"
                r"(k|m|million|thousand)?\b"
            ),
            (
                r"\b(?:budget|budget of)\s*"
                r"\$?\s*([\d,.]+)\s*"
                r"(k|m|million|thousand)?\b"
            ),
            (
                r"\$?\s*([\d,.]+)\s*"
                r"(k|m|million|thousand)\s+"
                r"(?:or less|maximum|max)\b"
            ),
        ]

        for pattern in patterns:
            match = re.search(pattern, message)

            if not match:
                continue

            number = float(
                match.group(1).replace(",", "")
            )

            suffix = match.group(2)

            if suffix:
                suffix = suffix.lower()

                if suffix == "k":
                    number *= 1_000

                elif suffix in {"m", "million"}:
                    number *= 1_000_000

                elif suffix == "thousand":
                    number *= 1_000

            return number

        return None

    def _extract_sqft(
        self,
        message: str,
    ) -> tuple[int | None, int | None]:
        """
        Extract square-foot requirements.

        Supported examples:
            "1500 sq ft"
            "1500 sqft"
            "1500 square feet"
            "at least 1500 sqft"
            "over 1500 square feet"
            "between 1500 and 2500 sqft"
        """

        min_sqft = None
        max_sqft = None

        # Range:
        # "between 1500 and 2500 sqft"
        range_match = re.search(
            r"\bbetween\s+([\d,]+)\s+and\s+([\d,]+)\s*"
            r"(?:sq\.?\s*ft\.?|sqft|square\s+feet)\b",
            message,
        )

        if range_match:
            min_sqft = int(
                range_match.group(1).replace(",", "")
            )

            max_sqft = int(
                range_match.group(2).replace(",", "")
            )

            return min_sqft, max_sqft

        # Minimum:
        # "at least 1500 sqft"
        # "over 1500 square feet"
        min_match = re.search(
            r"\b(?:at least|minimum|min|over|above)\s+"
            r"([\d,]+)\s*"
            r"(?:sq\.?\s*ft\.?|sqft|square\s+feet)\b",
            message,
        )

        if min_match:
            min_sqft = int(
                min_match.group(1).replace(",", "")
            )

        # Maximum:
        # "under 2500 sqft"
        # "below 2500 square feet"
        max_match = re.search(
            r"\b(?:under|below|less than|up to)\s+"
            r"([\d,]+)\s*"
            r"(?:sq\.?\s*ft\.?|sqft|square\s+feet)\b",
            message,
        )

        if max_match:
            max_sqft = int(
                max_match.group(1).replace(",", "")
            )

        return min_sqft, max_sqft
