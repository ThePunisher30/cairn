"""Phase 8, step 3: the trace (`agent/trace.py`).

Without a record of what the agent did you cannot explain a failure, and you cannot count the calls
it wasted. Plain dataclasses, written as JSON lines.

    ToolRecord(name, arguments, result_chars, error=None, duplicate=False, latency_ms=0.0)
        frozen. One tool call: its arguments (a dict), the size of the text sent back to the
        model, the error message when it failed (else None), whether it was a repeat of an earlier
        call that was not run again, and how long it took.
    Step(index, text, tool_calls=(), usage=Usage(), latency_ms=0.0, stop_reason="end_turn",
         error=None)
        frozen. One model call: its number (1, 2, ...), the text it wrote, its ToolRecords (a
        tuple), the tokens it used, the call's latency, the completion's stop_reason, and an error
        message when the model call itself failed (else None).
    Trace(question, steps=[])        a dataclass (not frozen); `steps` is a list
        .add(step)       appends
        .n_tool_calls    all ToolRecords over all steps
        .n_errors        ToolRecords whose error is not None
        .n_duplicates    ToolRecords with duplicate True
        .usage           the Usage summed over the steps
        .to_dict()       {"question": ..., "steps": [{"index", "text", "tool_calls": [{"name",
                         "arguments", "result_chars", "error", "duplicate", "latency_ms"}],
                         "usage": {"input_tokens", "output_tokens"}, "latency_ms", "stop_reason",
                         "error"}]}  (only JSON types)
        Trace.from_dict(d)   the inverse: Trace.from_dict(t.to_dict()) == t
    write_traces(path, traces) -> int     one JSON object per line (a whole trace per line); returns
                                          the count; creates the parent directory
    read_traces(path) -> list[Trace]
"""

import json

from cairn.agent.trace import Step, ToolRecord, Trace, read_traces, write_traces
from cairn.generation.types import Usage


def sample():
    trace = Trace("How do I enable CORS?")
    trace.add(
        Step(
            1,
            "Let me search.",
            (
                ToolRecord("search", {"query": "cors", "k": 5}, 1200, latency_ms=12.5),
                ToolRecord("search", {"query": "cors", "k": 5}, 80, duplicate=True),
            ),
            Usage(100, 20),
            900.0,
            "tool_use",
        )
    )
    trace.add(
        Step(
            2,
            "",
            (ToolRecord("read_file", {"path": "../x"}, 40, error="path outside the corpus"),),
            Usage(150, 10),
            400.0,
            "tool_use",
        )
    )
    trace.add(Step(3, "Use CORSMiddleware [1].", (), Usage(200, 30), 700.0, "end_turn"))
    return trace


def test_an_empty_trace():
    t = Trace("q")
    assert t.steps == [] and t.n_tool_calls == 0 and t.usage == Usage(0, 0)


def test_two_traces_do_not_share_their_steps():
    a, b = Trace("a"), Trace("b")
    a.add(Step(1, "x"))
    assert b.steps == []


def test_the_counts():
    t = sample()
    assert t.n_tool_calls == 3
    assert t.n_errors == 1
    assert t.n_duplicates == 1
    assert t.usage == Usage(450, 60)
    assert [s.index for s in t.steps] == [1, 2, 3]


def test_a_step_defaults():
    s = Step(1, "hi")
    assert s.tool_calls == () and s.usage == Usage() and s.error is None
    assert s.stop_reason == "end_turn" and s.latency_ms == 0.0


def test_to_dict_holds_only_json_types_and_the_documented_keys():
    d = sample().to_dict()
    json.dumps(d)  # must not raise
    assert set(d) == {"question", "steps"}
    first = d["steps"][0]
    assert set(first) == {
        "index",
        "text",
        "tool_calls",
        "usage",
        "latency_ms",
        "stop_reason",
        "error",
    }
    assert first["usage"] == {"input_tokens": 100, "output_tokens": 20}
    assert first["tool_calls"][0] == {
        "name": "search",
        "arguments": {"query": "cors", "k": 5},
        "result_chars": 1200,
        "error": None,
        "duplicate": False,
        "latency_ms": 12.5,
    }
    assert d["steps"][1]["tool_calls"][0]["error"] == "path outside the corpus"


def test_the_dict_round_trips():
    t = sample()
    assert Trace.from_dict(t.to_dict()) == t
    assert Trace.from_dict(json.loads(json.dumps(t.to_dict()))) == t


def test_traces_are_written_and_read_as_json_lines(tmp_path):
    path = tmp_path / "deep" / "runs" / "t.jsonl"
    other = Trace("second")
    other.add(Step(1, "answer", error="boom"))
    assert write_traces(path, [sample(), other]) == 2
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2
    back = read_traces(path)
    assert back == [sample(), other]
