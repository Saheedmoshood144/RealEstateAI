import pytest

from src.conversation.llm_fallback import UnavailableLLMFallback
from src.conversation.schemas import ConversationIntent

from src.conversation.schemas import (
    ConversationIntent,
    LLMParseOutput,
)

def test_unavailable_llm_fallback_returns_unknown():
    fallback = UnavailableLLMFallback()

    result = fallback.parse(
        "I'm looking for somewhere nice for my family."
    )

    assert result.intent == ConversationIntent.UNKNOWN
    assert result.parameters == {}
    assert result.confidence == 0.0


def test_unavailable_llm_preserves_original_message():
    fallback = UnavailableLLMFallback()

    message = "Help me find a home in Riverside."

    result = fallback.parse(message)

    assert result.original_message == message