from src.conversation.intent_parser import IntentParser
from src.conversation.llm_fallback import (
    LLMFallback,
    UnavailableLLMFallback,
)
from src.conversation.schemas import (
    ConversationIntent,
    ParsedMessage,
)


class HybridParser:
    """
    Combines deterministic NLP parsing with an optional LLM fallback.

    The deterministic parser is always tried first. The LLM is only
    consulted when the deterministic result is unknown or below the
    configured confidence threshold.
    """

    def __init__(
        self,
        deterministic_parser=None,
        llm_fallback: LLMFallback | None = None,
        confidence_threshold: float = 0.70,
    ):
        self.deterministic_parser = (
            deterministic_parser or IntentParser()
        )

        self.llm_fallback = (
            llm_fallback or UnavailableLLMFallback()
        )

        self.confidence_threshold = confidence_threshold

    def parse(self, message: str) -> ParsedMessage:
        """
        Parse a user message using the deterministic parser first.

        If the deterministic parser is confident enough, its result
        is returned immediately.

        Otherwise, the LLM fallback is consulted.
        """

        deterministic_result = self.deterministic_parser.parse(
            message
        )

        deterministic_is_confident = (
            deterministic_result.intent != ConversationIntent.UNKNOWN
            and deterministic_result.confidence
            >= self.confidence_threshold
        )

        if deterministic_is_confident:
            return deterministic_result

        llm_result = self.llm_fallback.parse(message)

        if llm_result.confidence > deterministic_result.confidence:
            return llm_result

        return deterministic_result