import os
from abc import ABC, abstractmethod

from openai import OpenAI

from src.conversation.schemas import (
    ConversationIntent,
    LLMParseOutput,
    ParsedMessage,
)


class LLMFallback(ABC):
    """
    Interface for LLM-based fallback parsers.
    """

    @abstractmethod
    def parse(self, message: str) -> ParsedMessage:
        raise NotImplementedError


class UnavailableLLMFallback(LLMFallback):
    """
    Safe fallback used when no LLM provider is configured.
    """

    def parse(self, message: str) -> ParsedMessage:
        return ParsedMessage(
            intent=ConversationIntent.UNKNOWN,
            parameters={},
            confidence=0.0,
            original_message=message,
        )


class OpenAILLMFallback(LLMFallback):
    """
    LLM fallback using the OpenAI Responses API.

    The LLM only interprets the user's message. It does not access
    listings, execute code, or make property predictions.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        self.model = model or os.getenv(
            "OPENAI_MODEL",
            "gpt-5.6-luna",
        )

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not configured."
            )

        self.client = OpenAI(api_key=self.api_key)

    def parse(self, message: str) -> ParsedMessage:
        """
        Parse an ambiguous natural-language request using an LLM.

        The LLM returns a validated LLMParseOutput object, which is
        converted into the application's ParsedMessage structure.
        """

        if not isinstance(message, str):
            raise TypeError("message must be a string.")

        message = message.strip()

        if not message:
            raise ValueError("message cannot be empty.")

        system_prompt = """
You are the natural-language understanding component of a
real-estate AI application.

Your ONLY task is to interpret the user's request and convert it
into structured information.

Allowed intents:

- recommend_properties
- predict_price
- general_help
- unknown

For recommend_properties, extract only parameters explicitly
stated or strongly implied by the user.

Supported parameters:

- city
- neighborhood
- property_type
- condition
- min_bedrooms
- min_bathrooms
- max_price
- min_sqft
- max_sqft

Normalize property types to:

- Single Family
- Townhouse
- Condo
- Multi-Family

Normalize common price expressions:

- "$500k" -> 500000
- "$1.2 million" -> 1200000
- "half a million" -> 500000
- "500 thousand" -> 500000

For phrases such as "three bedrooms", convert the number to 3.

Do NOT invent values.

Only return parameters that can reasonably be extracted from
the user's message.

Return a confidence score from 0.0 to 1.0.
"""

        try:
            response = self.client.responses.parse(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": message,
                    },
                ],
                text_format=LLMParseOutput,
            )

            parsed_output = response.output_parsed

            if parsed_output is None:
                raise ValueError(
                    "LLM did not return a structured response."
                )

            parameters = parsed_output.parameters.model_dump(
                exclude_none=True
            )

            return ParsedMessage(
                intent=parsed_output.intent,
                parameters=parameters,
                confidence=parsed_output.confidence,
                original_message=message,
            )

        except Exception:
            # The LLM is an optional enhancement.
            # If it is unavailable, return UNKNOWN so that the
            # deterministic/hybrid parser can remain the source
            # of truth.
            return ParsedMessage(
                intent=ConversationIntent.UNKNOWN,
                parameters={},
                confidence=0.0,
                original_message=message,
            )