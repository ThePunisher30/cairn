"""A scripted language model for tests and offline runs (Phases 7 and 8).

Provided, not part of the exercise.

    FakeLLM(["first answer", Completion("second", Usage(5, 2)), SomeError()])

Each call takes the next item: a str becomes a Completion, a Completion is returned as it is, an
exception is raised. When the list is empty the `responder` (a function of system and messages) is
asked instead; with neither, the call fails loudly. `stream` answers the same way, in word-sized
pieces. A `Partial(text, error)` item streams `text` and then raises `error`. After `max_calls`
calls (default 500) it raises AssertionError, so a loop that never stops fails instead of hanging.

For the agent (Phase 8): `tool_reply(("search", {"query": "cors"}), ("read_chunk", {"id": 1}))` is
a Completion that asks for those tool calls (ids are made up: call_1, call_2, ...). `complete`
accepts `tools=` and records it in `calls[i]["tools"]`.
"""

import itertools
import re
from dataclasses import dataclass

from cairn.generation.types import Completion, ToolCall, Usage

DEFAULT_USAGE = Usage(10, 5)  # the tokens every scripted answer reports


_ids = itertools.count(1)


def tool_reply(*calls, text="", usage=DEFAULT_USAGE, model="fake-model"):
    """A scripted reply that asks for tools. Each call is (name, arguments) or a ToolCall."""
    made = tuple(
        c if isinstance(c, ToolCall) else ToolCall(f"call_{next(_ids)}", c[0], dict(c[1]))
        for c in calls
    )
    return Completion(text, usage, model, "tool_use", 0.0, made)


@dataclass(frozen=True)
class Partial:
    text: str
    error: BaseException


class FakeLLM:
    def __init__(
        self,
        replies=None,
        *,
        responder=None,
        model="fake-model",
        usage=DEFAULT_USAGE,
        max_calls=500,
    ):
        self.replies = list(replies or [])
        self.responder = responder
        self.model = model
        self.usage = usage
        self.max_calls = max_calls
        self.calls: list[dict] = []
        self.last_usage: Usage | None = None

    def _next(self, system, messages):
        # A responder that always answers never runs dry, so code that forgot to stop (a loop
        # without a budget check) would hang the test suite. This turns the hang into a failure.
        if len(self.calls) > self.max_calls:
            raise AssertionError(
                f"FakeLLM was called more than {self.max_calls} times: is the code looping?"
            )
        if self.replies:
            return self.replies.pop(0)
        if self.responder is not None:
            return self.responder(system, messages)
        raise AssertionError("FakeLLM has no reply left")

    def _record(self, system, messages, max_tokens, stream, tools=None):
        # the messages are copied: the agent loop keeps appending to its own list
        self.calls.append(
            {
                "system": system,
                "messages": list(messages),
                "max_tokens": max_tokens,
                "stream": stream,
                "tools": tools,
            }
        )

    def complete(self, system, messages, *, max_tokens=1024, tools=None):
        self._record(system, messages, max_tokens, False, tools)
        item = self._next(system, messages)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, Partial):
            raise item.error
        if isinstance(item, str):
            item = Completion(item, self.usage, self.model)
        self.last_usage = item.usage
        return item

    def stream(self, system, messages, *, max_tokens=1024):
        self._record(system, messages, max_tokens, True)
        item = self._next(system, messages)
        if isinstance(item, BaseException):
            raise item
        error = None
        if isinstance(item, Partial):
            item, error = item.text, item.error
        text = item.text if isinstance(item, Completion) else item
        for piece in re.findall(r"\S+\s*", text):
            yield piece
        if error is not None:
            raise error
        self.last_usage = item.usage if isinstance(item, Completion) else self.usage
