from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationState:
    """
    Stores conversational context for a single session.

    The state contains only information needed to interpret
    follow-up real-estate requests.
    """

    last_intent: str | None = None
    last_parameters: dict[str, Any] = field(default_factory=dict)
    last_recommendations: list[dict[str, Any]] = field(
        default_factory=list
    )

    def update(
        self,
        intent: str,
        parameters: dict[str, Any],
        recommendations: list[dict[str, Any]] | None = None,
    ) -> None:
        self.last_intent = intent
        self.last_parameters = parameters.copy()

        if recommendations is not None:
            self.last_recommendations = recommendations.copy()