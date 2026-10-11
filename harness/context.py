"""Context window management.

Strategy pattern: `ContextStrategy` decides what to do when the message log
approaches the model's context window. The default keeps the system prompt,
the original user goal, and the most recent turns, dropping the middle.

Token counting is a cheap char/4 heuristic by default (no tokenizer dep);
inject a real counter when exactness matters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Protocol

from harness.errors import ContextOverflow
from harness.models import Conversation, Message, SystemMessage, UserMessage


class TokenCounter(Protocol):
    def __call__(self, messages: Sequence[Message]) -> int: ...


def heuristic_counter(messages: Sequence[Message]) -> int:
    """~4 characters per token, plus fixed overhead per message."""
    total = 0
    for m in messages:
        total += 4
        total += len(getattr(m, "content", "") or "") // 4
        for tc in getattr(m, "tool_calls", None) or []:
            total += len(str(tc.arguments)) // 4 + 8
    return total


class ContextStrategy(ABC):
    """Decides the message list handed to the provider."""

    def __init__(self, counter: TokenCounter = heuristic_counter) -> None:
        self.count = counter

    @abstractmethod
    def fit(
        self, conversation: Conversation, *, budget: int
    ) -> tuple[list[Message], list[Message]]:
        """Return (kept, dropped) for the given token budget."""


class SlidingWindowStrategy(ContextStrategy):
    """Keep the system prompt + goal + latest messages; drop the middle.

    The first user message is pinned so the task statement survives even after
    many tool round-trips.
    """

    def __init__(
        self,
        counter: TokenCounter = heuristic_counter,
        *,
        min_recent: int = 6,
    ) -> None:
        super().__init__(counter)
        self._min_recent = min_recent

    def fit(
        self, conversation: Conversation, *, budget: int
    ) -> tuple[list[Message], list[Message]]:
        messages = conversation.messages
        if self.count(messages) <= budget:
            return list(messages), []

        system = [m for m in messages if isinstance(m, SystemMessage)]
        rest = [m for m in messages if not isinstance(m, SystemMessage)]

        pinned: list[Message] = []
        if rest and isinstance(rest[0], UserMessage):
            pinned = [rest[0]]
            rest = rest[1:]

        head_room = budget - self.count(system) - self.count(pinned)
        if head_room <= 0:
            raise ContextOverflow("system prompt and goal alone exceed the budget")

        # Walk backwards accumulating recent messages until head_room is spent.
        keep_count = 0
        used = 0
        for msg in reversed(rest):
            cost = self.count([msg])
            if used + cost > head_room and keep_count >= self._min_recent:
                break
            used += cost
            keep_count += 1

        split = len(rest) - keep_count
        kept = system + pinned + rest[split:]
        kept_ids = {id(m) for m in kept}
        dropped = [m for m in messages if id(m) not in kept_ids]
        return kept, dropped


def render_pruned_note(dropped: int) -> str:
    """A single line the model sees in place of the pruned span."""
    return f"[context pruned: {dropped} earlier messages elided to fit the window]"
