from src.conversation.hybrid_parser import HybridParser
from src.conversation.intent_parser import IntentParser
from src.conversation.llm_fallback import LLMFallback
from src.conversation.schemas import (
    ConversationIntent,
    ParsedMessage,
)


class MockLLMFallback(LLMFallback):
    """
    Test double representing an LLM-based parser.
    """

    def parse(self, message: str) -> ParsedMessage:
        return ParsedMessage(
            intent=ConversationIntent.RECOMMEND_PROPERTIES,
            parameters={
                "city": "Riverside",
                "min_bedrooms": 3,
            },
            confidence=0.90,
            original_message=message,
        )


def test_deterministic_parser_is_used_when_confident():
    parser = HybridParser(
        deterministic_parser=IntentParser(),
        llm_fallback=MockLLMFallback(),
    )

    result = parser.parse(
        "Find a townhouse in Riverside with 3 bedrooms"
    )

    assert result.intent == ConversationIntent.RECOMMEND_PROPERTIES
    assert result.parameters["city"] == "Riverside"
    assert result.parameters["min_bedrooms"] == 3


def test_llm_fallback_is_used_for_unknown_request():
    parser = HybridParser(
        deterministic_parser=IntentParser(),
        llm_fallback=MockLLMFallback(),
    )

    result = parser.parse(
        "I'm relocating and need a spacious family home"
    )

    assert result.intent == ConversationIntent.RECOMMEND_PROPERTIES
    assert result.parameters["city"] == "Riverside"
    assert result.parameters["min_bedrooms"] == 3
    assert result.confidence == 0.95


def test_original_result_used_without_llm():
    parser = HybridParser(
        deterministic_parser=IntentParser(),
        llm_fallback=None,
    )

    result = parser.parse(
        "Tell me something unrelated"
    )

    assert result.intent == ConversationIntent.UNKNOWN
    assert result.confidence == 0.0

class MockLLMFallback:
    def __init__(self):
        self.called = False

    def parse(self, message: str) -> ParsedMessage:
        self.called = True

        return ParsedMessage(
            intent=ConversationIntent.RECOMMEND_PROPERTIES,
            parameters={
                "city": "Riverside",
                "min_bedrooms": 3,
            },
            confidence=0.95,
            original_message=message,
        )

def test_llm_fallback_receives_original_message():
    fallback = MockLLMFallback()

    parser = HybridParser(
        llm_fallback=fallback
    )

    message = "I'm moving soon and need a spacious place for my family."

    result = parser.parse(message)

    assert fallback.called is True
    assert result.original_message == message
    assert result.intent == ConversationIntent.RECOMMEND_PROPERTIES