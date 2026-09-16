import os

from src.conversation.llm_fallback import (
    LLMFallback,
    OpenAILLMFallback,
    UnavailableLLMFallback,
)


def create_llm_fallback() -> LLMFallback:
    """
    Create the configured LLM fallback provider.

    If no API key is configured, return a safe unavailable provider.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return UnavailableLLMFallback()

    return OpenAILLMFallback(
        api_key=api_key,
        model=os.getenv("OPENAI_MODEL"),
    )