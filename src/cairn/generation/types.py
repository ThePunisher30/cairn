"""The types the generation code shares (Phases 7 and 8). Provided, not part of the exercise."""

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
class ToolSpec:
    """A tool as the model is told about it: a name, what it does, and a JSON schema of its
    arguments (`{"type": "object", "properties": {...}, "required": [...]}`)."""

    name: str
    description: str
    parameters: dict


@dataclass(frozen=True)
class ToolCall:
    """A request from the model to run a tool. The model runs nothing; our code does.

    `error` is set by a client when the arguments could not be read (for example a JSON string
    that does not parse). `arguments` is then {} and the loop reports the error to the model
    instead of running the tool.
    """

    id: str
    name: str
    arguments: dict
    error: str | None = None


@dataclass(frozen=True)
class Completion:
    """One finished answer from a language model.

    `stop_reason` uses three fixed words whatever the provider says: "end_turn" (finished),
    "max_tokens" (cut off by the limit) and "refusal" (the provider declined). Anything else is
    passed through as it came. When the model asks for tools, `tool_calls` is not empty and
    `stop_reason` is "tool_use" (the client sets it, even when the provider says "stop").
    """

    text: str
    usage: Usage = Usage()
    model: str = ""
    stop_reason: str = "end_turn"
    latency_ms: float = 0.0
    tool_calls: tuple[ToolCall, ...] = ()


class LLMClient(Protocol):
    """What the rest of the code needs from a language model, from any provider.

    `messages` is a list of {"role": "user" | "assistant", "content": str}; the system prompt is
    a separate argument. With tools (Phase 8) two more shapes appear, in one neutral format that
    each client translates to its provider's:

        {"role": "assistant", "content": "text, may be empty",
         "tool_calls": [{"id": "call_1", "name": "search", "arguments": {"query": "..."}}]}
        {"role": "tool", "tool_call_id": "call_1", "name": "search", "content": "the result"}

    Every tool call must be answered by a tool message with the same id. There is no temperature
    argument: the Anthropic API no longer accepts one. `last_usage` is the Usage of the last call;
    for `stream` it is set when the stream has been read to the end (None before that).
    """

    last_usage: Usage | None

    def complete(
        self,
        system: str,
        messages: list[dict],
        *,
        max_tokens: int = 1024,
        tools: list[ToolSpec] | None = None,
    ) -> Completion: ...

    def stream(
        self, system: str, messages: list[dict], *, max_tokens: int = 1024
    ) -> Iterator[str]: ...
