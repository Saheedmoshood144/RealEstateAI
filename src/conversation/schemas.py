from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ConversationIntent(str, Enum):
    RECOMMEND_PROPERTIES = "recommend_properties"
    PREDICT_PRICE = "predict_price"
    GENERAL_HELP = "general_help"
    UNKNOWN = "unknown"


class ParsedMessage(BaseModel):
    """
    Structured representation of a user's natural-language request.
    """

    intent: ConversationIntent

    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured parameters extracted from the user's message.",
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence that the detected intent and parameters are correct.",
    )

    original_message: str


class LLMParameters(BaseModel):
    """
    Strict set of parameters that the LLM is allowed to extract.

    All fields are optional because a user's message may contain
    only some of the supported property-search information.
    """

    city: str | None = Field(
        default=None,
        description="City explicitly stated or strongly implied by the user.",
    )

    neighborhood: str | None = Field(
        default=None,
        description="Neighborhood explicitly stated by the user.",
    )

    property_type: str | None = Field(
        default=None,
        description=(
            "Normalized property type: Single Family, Townhouse, "
            "Condo, or Multi-Family."
        ),
    )

    condition: str | None = Field(
        default=None,
        description="Property condition, such as Excellent, Good, Fair, or Poor.",
    )

    min_bedrooms: int | None = Field(
        default=None,
        ge=0,
        description="Minimum number of bedrooms requested.",
    )

    min_bathrooms: float | None = Field(
        default=None,
        ge=0,
        description="Minimum number of bathrooms requested.",
    )

    max_price: float | None = Field(
        default=None,
        ge=0,
        description="Maximum property price requested.",
    )

    min_sqft: int | None = Field(
        default=None,
        ge=0,
        description="Minimum property size in square feet.",
    )

    max_sqft: int | None = Field(
        default=None,
        ge=0,
        description="Maximum property size in square feet.",
    )


class LLMParseOutput(BaseModel):
    """
    Strict schema for structured output produced by an LLM.

    The LLM is responsible only for understanding the user's request.
    It does not directly access listings or make predictions.
    """

    intent: ConversationIntent

    parameters: LLMParameters = Field(
        default_factory=LLMParameters,
        description="Extracted real-estate parameters.",
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the interpretation.",
    )


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language real-estate request.",
    )
    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=100,
        description="Conversation session identifier.",
    )

class ChatResponse(BaseModel):
    intent: ConversationIntent
    message: str
    data: Any | None = None