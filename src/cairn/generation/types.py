"""The types the generation code shares (Phase 7). Provided, not part of the exercise."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Usage:
    """Tokens a call used, as the provider counts them."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens
        )


@dataclass(frozen=True)
class Completion:
    """One finished answer from a language model.

    `stop_reason` uses three fixed words whatever the provider says: "end_turn" (finished),
    "max_tokens" (cut off by the limit) and "refusal" (the provider declined). Anything else is
    passed through as it came.
    """

    text: str
    usage: Usage = Usage()
    model: str = ""
    stop_reason: str = "end_turn"
    latency_ms: float = 0.0


class LLMClient(Protocol):
    """What the rest of the code needs from a language model, from any provider.

    `messages` is a list of {"role": "user" | "assistant", "content": str}; the system prompt is
    a separate argument. There is no temperature argument: the Anthropic API no longer accepts
    one. `last_usage` is the Usage of the last call; for `stream` it is set when the stream has
    been read to the end (None before that).
    """

    last_usage: Usage | None

    def complete(
        self, system: str, messages: list[dict], *, max_tokens: int = 1024
    ) -> Completion: ...

    def stream(
        self, system: str, messages: list[dict], *, max_tokens: int = 1024
    ) -> Iterator[str]: ...
