"""A scripted language model for tests and offline runs (Phase 7).

Provided, not part of the exercise.

    FakeLLM(["first answer", Completion("second", Usage(5, 2)), SomeError()])

Each call takes the next item: a str becomes a Completion, a Completion is returned as it is, an
exception is raised. When the list is empty the `responder` (a function of system and messages) is
asked instead; with neither, the call fails loudly. `stream` answers the same way, in word-sized
pieces. A `Partial(text, error)` item streams `text` and then raises `error`.
"""

import re
from dataclasses import dataclass

from cairn.generation.types import Completion, Usage

DEFAULT_USAGE = Usage(10, 5)  # the tokens every scripted answer reports


@dataclass(frozen=True)
class Partial:
    text: str
    error: BaseException


class FakeLLM:
    def __init__(self, replies=None, *, responder=None, model="fake-model", usage=DEFAULT_USAGE):
        self.replies = list(replies or [])
        self.responder = responder
        self.model = model
        self.usage = usage
        self.calls: list[dict] = []
        self.last_usage: Usage | None = None

    def _next(self, system, messages):
        if self.replies:
            return self.replies.pop(0)
        if self.responder is not None:
            return self.responder(system, messages)
        raise AssertionError("FakeLLM has no reply left")

    def _record(self, system, messages, max_tokens, stream):
        self.calls.append(
            {"system": system, "messages": messages, "max_tokens": max_tokens, "stream": stream}
        )

    def complete(self, system, messages, *, max_tokens=1024):
        self._record(system, messages, max_tokens, False)
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
