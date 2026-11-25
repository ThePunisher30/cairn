"""A record of what the agent did, step by step, written as JSON lines.

Without it you cannot explain a failure or count the calls an agent wasted.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from cairn.generation.types import Usage


@dataclass(frozen=True)
class ToolRecord:
    """One tool call: its arguments, the size of the text sent back, and how it went."""

    name: str
    arguments: dict
    result_chars: int
    error: str | None = None
    duplicate: bool = False  # a repeat of an earlier call, so it was not run again
    latency_ms: float = 0.0


@dataclass(frozen=True)
class Step:
    """One model call and the tool calls it asked for."""

    index: int
    text: str
    tool_calls: tuple[ToolRecord, ...] = ()
    usage: Usage = Usage()
    latency_ms: float = 0.0
    stop_reason: str = "end_turn"
    error: str | None = None  # set when the model call itself failed


@dataclass
class Trace:
    question: str
    steps: list[Step] = field(default_factory=list)

    def add(self, step):
        self.steps.append(step)

    def _records(self):
        return [record for step in self.steps for record in step.tool_calls]

    @property
    def n_tool_calls(self):
        return len(self._records())

    @property
    def n_errors(self):
        return sum(r.error is not None for r in self._records())

    @property
    def n_duplicates(self):
        return sum(r.duplicate for r in self._records())

    @property
    def usage(self):
        total = Usage()
        for step in self.steps:
            total = total + step.usage
        return total

    def to_dict(self):
        return {
            "question": self.question,
            "steps": [
                {
                    "index": s.index,
                    "text": s.text,
                    "tool_calls": [
                        {
                            "name": r.name,
                            "arguments": r.arguments,
                            "result_chars": r.result_chars,
                            "error": r.error,
                            "duplicate": r.duplicate,
                            "latency_ms": r.latency_ms,
                        }
                        for r in s.tool_calls
                    ],
                    "usage": {
                        "input_tokens": s.usage.input_tokens,
                        "output_tokens": s.usage.output_tokens,
                    },
                    "latency_ms": s.latency_ms,
                    "stop_reason": s.stop_reason,
                    "error": s.error,
                }
                for s in self.steps
            ],
        }

    @classmethod
    def from_dict(cls, d):
        steps = [
            Step(
                s["index"],
                s["text"],
                tuple(ToolRecord(**r) for r in s["tool_calls"]),
                Usage(s["usage"]["input_tokens"], s["usage"]["output_tokens"]),
                s["latency_ms"],
                s["stop_reason"],
                s["error"],
            )
            for s in d["steps"]
        ]
        return cls(d["question"], steps)


def write_traces(path, traces):
    """One trace per line. Returns how many were written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_traces(path):
    with Path(path).open(encoding="utf-8") as f:
        return [Trace.from_dict(json.loads(line)) for line in f if line.strip()]
