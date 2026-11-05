"""Phase 8, step 1: tools in the language-model clients (`generation/clients.py`, `retry.py`).

Provided: `ToolSpec`, `ToolCall`, `Completion.tool_calls` (`generation/types.py`) and the scripted
`FakeLLM` with `tool_reply(...)` (`generation/fake.py`). You change:

retry.py
    RetryingClient.complete(system, messages, *, max_tokens=1024, tools=None)
        passes `tools=tools` to the inner client, but ONLY when tools is not None (an inner client
        written before Phase 8 has no such argument and must keep working). Retries as before.

clients.py: both clients take `tools=None` in `complete`, and translate between our neutral message
format (see `generation/types.py`) and the provider's.

    OllamaClient
        tools -> body["tools"] = [{"type": "function", "function": {"name", "description",
        "parameters"}}, ...]. No "tools" key at all when tools is None.
        An assistant message with tool_calls goes out as
            {"role": "assistant", "content": text,
             "tool_calls": [{"function": {"name": ..., "arguments": {...}}}]}
        a tool message as {"role": "tool", "tool_name": name, "content": text}.
        Plain messages are sent unchanged.
        The reply's message.tool_calls become `Completion.tool_calls`:
            function.name, function.arguments, id = the reply's "id" when it has one, else
            "call_<position>" (call_0, call_1, ...).
        `arguments` may arrive as a dict, or as a JSON string (some models do that): a string is
        parsed. If it does not parse into an object, the ToolCall has arguments {} and `error`
        starts with "arguments are not a JSON object" (the loop reports it to the model).
        Ollama says done_reason "stop" for a reply that calls tools; the client reports
        stop_reason "tool_use" whenever tool_calls is not empty.

    AnthropicClient
        tools -> request["tools"] = [{"name", "description", "input_schema": parameters}, ...];
        no "tools" key when None. Everything else in the request is as in Phase 7.
        An assistant message with tool_calls goes out with content blocks: a {"type": "text"} block
        when the text is not empty, then one {"type": "tool_use", "id", "name", "input"} per call.
        Tool messages go out as a USER message of {"type": "tool_result", "tool_use_id", "content"}
        blocks; consecutive tool messages are merged into one user message (the API wants every
        result of one assistant turn together). Plain messages are unchanged.
        The reply's "tool_use" blocks become `Completion.tool_calls` (id, name, input). The text
        is still the text blocks joined, and stop_reason is passed through ("tool_use").
"""

import json
from types import SimpleNamespace

import pytest

from cairn.generation.clients import AnthropicClient, OllamaClient
from cairn.generation.errors import RateLimitError
from cairn.generation.fake import FakeLLM, tool_reply
from cairn.generation.retry import RetryingClient, RetryPolicy
from cairn.generation.types import Completion, ToolCall, ToolSpec, Usage

SEARCH = ToolSpec(
    "search",
    "Search the docs.",
    {
        "type": "object",
        "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
        "required": ["query"],
    },
)
QUESTION = [{"role": "user", "content": "How do I enable CORS?"}]
TURN = [
    *QUESTION,
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "c1", "name": "search", "arguments": {"query": "cors"}},
            {"id": "c2", "name": "search", "arguments": {"query": "middleware"}},
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "name": "search", "content": "result one"},
    {"role": "tool", "tool_call_id": "c2", "name": "search", "content": "result two"},
]


# ------------------------------------------------------------------ the provided pieces
def test_a_tool_reply_asks_for_tools_and_the_fake_records_them():
    fake = FakeLLM([tool_reply(("search", {"query": "x"}), ("read_chunk", {"id": 1}))])
    reply = fake.complete("sys", QUESTION, tools=[SEARCH])
    assert reply.stop_reason == "tool_use"
    assert [(c.name, c.arguments) for c in reply.tool_calls] == [
        ("search", {"query": "x"}),
        ("read_chunk", {"id": 1}),
    ]
    assert len({c.id for c in reply.tool_calls}) == 2
    assert fake.calls[0]["tools"] == [SEARCH]


def test_a_plain_reply_has_no_tool_calls():
    assert FakeLLM(["hi"]).complete("s", QUESTION).tool_calls == ()


# ------------------------------------------------------------------ RetryingClient
class OldClient:
    """A client from before Phase 8: no `tools` argument."""

    last_usage = None

    def complete(self, system, messages, *, max_tokens=1024):
        return Completion("ok")


def test_retrying_client_passes_the_tools_on():
    fake = FakeLLM([tool_reply(("search", {"query": "x"}))])
    reply = RetryingClient(fake).complete("s", QUESTION, tools=[SEARCH])
    assert reply.tool_calls[0].name == "search"
    assert fake.calls[0]["tools"] == [SEARCH]


def test_retrying_client_does_not_pass_tools_when_there_are_none():
    assert RetryingClient(OldClient()).complete("s", QUESTION).text == "ok"


def test_a_retry_sends_the_tools_again():
    fake = FakeLLM([RateLimitError("slow down"), tool_reply(("search", {"query": "x"}))])
    client = RetryingClient(fake, RetryPolicy(max_attempts=3), sleep=lambda s: None)
    reply = client.complete("s", QUESTION, tools=[SEARCH])
    assert reply.tool_calls and client.retries == 1
    assert [c["tools"] for c in fake.calls] == [[SEARCH], [SEARCH]]


# ------------------------------------------------------------------ OllamaClient
httpx = pytest.importorskip("httpx")


def ollama(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ollama.test")
    return OllamaClient("llama-test", client=client)


def chat_reply(message, **extra):
    data = {"model": "llama-test", "message": message, "done": True, "done_reason": "stop"}
    return httpx.Response(200, json={**data, **extra})


def test_ollama_sends_the_tools_in_function_format():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return chat_reply({"role": "assistant", "content": "done"})

    ollama(handler).complete("sys", QUESTION, tools=[SEARCH])
    assert seen[0]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the docs.",
                "parameters": SEARCH.parameters,
            },
        }
    ]
    assert seen[0]["messages"][0] == {"role": "system", "content": "sys"}


def test_ollama_sends_no_tools_key_without_tools():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return chat_reply({"role": "assistant", "content": "done"})

    ollama(handler).complete("sys", QUESTION)
    assert "tools" not in seen[0]


def test_ollama_translates_a_tool_turn_to_its_own_message_format():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return chat_reply({"role": "assistant", "content": "done"})

    ollama(handler).complete("sys", TURN, tools=[SEARCH])
    sent = seen[0]["messages"]
    assert sent[1] == QUESTION[0]
    assert sent[2] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "search", "arguments": {"query": "cors"}}},
            {"function": {"name": "search", "arguments": {"query": "middleware"}}},
        ],
    }
    assert sent[3] == {"role": "tool", "tool_name": "search", "content": "result one"}
    assert sent[4] == {"role": "tool", "tool_name": "search", "content": "result two"}


def test_ollama_reads_tool_calls_and_reports_tool_use():
    reply = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "search", "arguments": {"query": "cors", "k": 3}}},
            {"id": "abc", "function": {"name": "read_chunk", "arguments": {"id": 2}}},
        ],
    }
    result = ollama(lambda r: chat_reply(reply, prompt_eval_count=30, eval_count=9)).complete(
        "s", QUESTION, tools=[SEARCH]
    )
    assert result.stop_reason == "tool_use"  # Ollama itself says "stop"
    assert result.text == "" and result.usage == Usage(30, 9)
    assert result.tool_calls == (
        ToolCall("call_0", "search", {"query": "cors", "k": 3}),
        ToolCall("abc", "read_chunk", {"id": 2}),
    )


def test_ollama_parses_arguments_that_come_as_a_json_string():
    reply = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "search", "arguments": '{"query": "cors"}'}}],
    }
    result = ollama(lambda r: chat_reply(reply)).complete("s", QUESTION, tools=[SEARCH])
    assert result.tool_calls[0].arguments == {"query": "cors"}
    assert result.tool_calls[0].error is None


@pytest.mark.parametrize("bad", ["{not json", "[1, 2]", '"just a string"', 7])
def test_ollama_marks_unreadable_arguments_instead_of_raising(bad):
    reply = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "search", "arguments": bad}}],
    }
    result = ollama(lambda r: chat_reply(reply)).complete("s", QUESTION, tools=[SEARCH])
    call = result.tool_calls[0]
    assert call.name == "search" and call.arguments == {}
    assert call.error.startswith("arguments are not a JSON object")


def test_ollama_a_text_reply_has_no_tool_calls():
    result = ollama(lambda r: chat_reply({"role": "assistant", "content": "hi"})).complete(
        "s", QUESTION, tools=[SEARCH]
    )
    assert result.tool_calls == () and result.stop_reason == "end_turn"


# ------------------------------------------------------------------ AnthropicClient
class FakeSDK:
    def __init__(self, reply):
        self.calls, self._reply, self.messages = [], reply, self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._reply


def block(kind, **fields):
    return SimpleNamespace(type=kind, **fields)


def sdk_reply(content, stop_reason="end_turn"):
    return SimpleNamespace(
        content=content,
        usage=SimpleNamespace(input_tokens=40, output_tokens=12),
        model="claude-test",
        stop_reason=stop_reason,
    )


def test_anthropic_sends_the_tools_with_input_schema():
    sdk = FakeSDK(sdk_reply([block("text", text="ok")]))
    AnthropicClient(sdk=sdk).complete("sys", QUESTION, tools=[SEARCH])
    assert sdk.calls[0]["tools"] == [
        {"name": "search", "description": "Search the docs.", "input_schema": SEARCH.parameters}
    ]
    for banned in ("temperature", "top_p", "top_k"):
        assert banned not in sdk.calls[0]


def test_anthropic_sends_no_tools_key_without_tools():
    sdk = FakeSDK(sdk_reply([block("text", text="ok")]))
    AnthropicClient(sdk=sdk).complete("sys", QUESTION)
    assert "tools" not in sdk.calls[0]


def test_anthropic_translates_a_tool_turn_to_blocks_and_merges_the_results():
    sdk = FakeSDK(sdk_reply([block("text", text="ok")]))
    turn = [*TURN[:2]]
    turn[1] = {**turn[1], "content": "Let me look."}
    AnthropicClient(sdk=sdk).complete("sys", [*turn, *TURN[2:]], tools=[SEARCH])
    sent = sdk.calls[0]["messages"]
    assert len(sent) == 3  # the question, the assistant turn, ONE user message with both results
    assert sent[0] == QUESTION[0]
    assert sent[1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me look."},
            {"type": "tool_use", "id": "c1", "name": "search", "input": {"query": "cors"}},
            {"type": "tool_use", "id": "c2", "name": "search", "input": {"query": "middleware"}},
        ],
    }
    assert sent[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "result one"},
            {"type": "tool_result", "tool_use_id": "c2", "content": "result two"},
        ],
    }


def test_anthropic_leaves_out_an_empty_text_block():
    sdk = FakeSDK(sdk_reply([block("text", text="ok")]))
    AnthropicClient(sdk=sdk).complete("sys", TURN, tools=[SEARCH])
    assert [b["type"] for b in sdk.calls[0]["messages"][1]["content"]] == ["tool_use", "tool_use"]


def test_anthropic_reads_tool_use_blocks():
    content = [
        block("thinking", thinking="hmm"),
        block("text", text="Looking."),
        block("tool_use", id="toolu_1", name="search", input={"query": "cors"}),
    ]
    result = AnthropicClient(sdk=FakeSDK(sdk_reply(content, "tool_use"))).complete(
        "s", QUESTION, tools=[SEARCH]
    )
    assert result.text == "Looking." and result.stop_reason == "tool_use"
    assert result.tool_calls == (ToolCall("toolu_1", "search", {"query": "cors"}),)
    assert result.usage == Usage(40, 12)
