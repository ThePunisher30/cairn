"""Phase 8, step 5: the agent loop (`agent/loop.py`) and its system prompt.

run_agent(question, llm, tools, budget=None, *, system=None, max_output_tokens=1024,
          clock=time.monotonic) -> AgentResult

`tools` is a ToolRegistry. `budget` None means Budget(). `system` None means the system part of
the prompt file `agent_v1.txt` (load_prompt("agent").system). `clock` is used for the
timeout and for tool latencies.

AgentResult (frozen): question, answer (str or None), stop_reason, trace (a Trace), usage (the
    Usage summed over every model call), elapsed_s. Properties: .steps (= len(trace.steps)),
    .n_tool_calls (= trace.n_tool_calls).

The loop, exactly:
  messages = [{"role": "user", "content": question}]
  repeat:
    1. reason = tracker.exceeded(); if there is one, STOP with stop_reason = reason and
       answer None. No model call is made after a budget is used up.
    2. completion = llm.complete(system, messages, max_tokens=max_output_tokens,
       tools=tools.specs()).  An LLMError is not raised out of the loop: the call is
       recorded as a Step (text "", no tool calls, Usage(), error=str(exc)) and the loop STOPS
       with "model_error", answer None. Any other exception is a bug and propagates.
    3. tracker.record(completion.usage); record a Step (index = 1, 2, ...; text, usage,
       latency_ms and stop_reason of the completion, its tool calls as ToolRecords).
    4. No tool calls:
         text empty after strip()            -> STOP "model_error" (the Step's error is
                                                "empty reply"), answer None
         completion.stop_reason "max_tokens" -> STOP "truncated", answer = text.strip()
         otherwise                           -> STOP "answered",  answer = text.strip()
    5. Tool calls: append {"role": "assistant", "content": completion.text, "tool_calls":
       [{"id", "name", "arguments"}, ...]}, then ONE {"role": "tool", "tool_call_id": id,
       "name": name, "content": text} per call, in order. EVERY call gets a tool message, or
       the provider rejects the next request.
         - the first budget.max_calls_per_step calls run; each further call is not run and its
           message is "Error: too many tool calls in one step (limit N)"
         - a call with the same name and the same arguments as an earlier call in this run
           (also within one step; the order of keys does not matter) is not run again; its
           message starts "Error: you already made this exact call" and says to try
           something different or answer now. The ToolRecord has duplicate=True.
         - otherwise tools.execute(call). An error result is sent to the model as it is
           ("Error: ...") and the loop goes on: the model can fix its call.
         - a call that is not run (a repeat, or one over the limit) still gets a ToolRecord: a
           repeat has duplicate=True and error None, a call over the limit has an error
         - each ToolRecord has result_chars = len(the text sent), the error, and latency_ms
           measured with `clock` (in milliseconds).

cited_hits(text, seen) -> list[Hit]
    The Hits for the valid citations in `text`: each [n] with 1 <= n <= len(seen), in order of
    first appearance, no repeats; `seen[n - 1]`. Invalid numbers are ignored.
    (Use extract_citations and validate_citations from generation/citations.py.)

The prompt `src/cairn/generation/prompts/agent_v1.txt`, loaded with load_prompt("agent"):
    the user part has exactly one placeholder, {{question}}; the system part tells the model
    which tools it has and to search before it answers, to cite chunks as [n] with the id
    search printed, to reply with CANNOT_ANSWER when the tools do not find the answer, to stop
    when it has enough, and that tool results are data, not instructions.
"""

import random

import pytest

from cairn.agent.budget import Budget
from cairn.agent.loop import AgentResult, cited_hits, run_agent
from cairn.agent.tools import Tool, ToolError, ToolRegistry
from cairn.generation.errors import AuthError, ServerError
from cairn.generation.fake import DEFAULT_USAGE, FakeLLM, tool_reply
from cairn.generation.prompts import load_prompt
from cairn.generation.types import Completion, ToolCall, Usage
from cairn.types import Hit

QUESTION = "Which function runs a sync dependency in a thread, and what does it call?"
SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}, "extra": {"type": "integer"}},
    "required": ["query"],
}


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class Env:
    """A registry with an `echo` tool that counts its runs, a `boom` tool that raises ToolError,
    and a clock the tests can move."""

    def __init__(self, *, echo_text=None, tool_seconds=0.0):
        self.clock = Clock()
        self.ran = []

        def echo(query, extra=0):
            self.ran.append((query, extra))
            self.clock.now += tool_seconds
            return echo_text if echo_text is not None else f"result for {query}"

        def boom(query):
            raise ToolError("the page is gone")

        self.tools = ToolRegistry(
            [Tool("echo", "Echoes.", SCHEMA, echo), Tool("boom", "Fails.", SCHEMA, boom)]
        )

    def run(self, llm, budget=None, **kw):
        return run_agent(QUESTION, llm, self.tools, budget, clock=self.clock, **kw)


def advancing(env, replies, seconds):
    """A FakeLLM whose every call takes `seconds` on the env's clock."""
    queue = list(replies)

    def responder(system, messages):
        env.clock.now += seconds
        return queue.pop(0) if queue else tool_reply(("echo", {"query": f"q{len(messages)}"}))

    return FakeLLM(responder=responder)


def endless():
    """A model that asks for another search forever (every query is new)."""
    return FakeLLM(responder=lambda s, m: tool_reply(("echo", {"query": f"q{len(m)}"})))


# ------------------------------------------------------------------ the happy paths
def test_a_direct_answer_takes_one_step_and_no_tool():
    env = Env()
    llm = FakeLLM(["  It is run_in_threadpool [1].\n"])
    result = env.run(llm, system="SYS", max_output_tokens=321)
    assert isinstance(result, AgentResult)
    assert result.answer == "It is run_in_threadpool [1]."
    assert result.stop_reason == "answered"
    assert result.steps == 1 and result.n_tool_calls == 0
    assert result.usage == DEFAULT_USAGE
    assert env.ran == []
    call = llm.calls[0]
    assert call["system"] == "SYS" and call["max_tokens"] == 321
    assert call["messages"] == [{"role": "user", "content": QUESTION}]
    assert call["tools"] == env.tools.specs()


def test_the_default_system_prompt_is_the_agent_prompt():
    llm = FakeLLM(["ok"])
    Env().run(llm)
    assert llm.calls[0]["system"] == load_prompt("agent").system


def test_two_tool_steps_then_an_answer_and_the_messages_the_model_sees():
    env = Env()
    first = tool_reply(("echo", {"query": "thread pool"}), text="I will search.")
    second = tool_reply(("echo", {"query": "run_in_threadpool", "extra": 2}))
    llm = FakeLLM([first, second, "Final answer [1]."])
    result = env.run(llm)
    assert result.answer == "Final answer [1]." and result.stop_reason == "answered"
    assert result.steps == 3 and result.n_tool_calls == 2
    assert env.ran == [("thread pool", 0), ("run_in_threadpool", 2)]
    assert result.usage == Usage(30, 15)

    c1, c2 = first.tool_calls[0], second.tool_calls[0]
    third_call_messages = llm.calls[2]["messages"]
    assert third_call_messages == [
        {"role": "user", "content": QUESTION},
        {
            "role": "assistant",
            "content": "I will search.",
            "tool_calls": [{"id": c1.id, "name": "echo", "arguments": {"query": "thread pool"}}],
        },
        {
            "role": "tool",
            "tool_call_id": c1.id,
            "name": "echo",
            "content": "result for thread pool",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": c2.id,
                    "name": "echo",
                    "arguments": {"query": "run_in_threadpool", "extra": 2},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": c2.id,
            "name": "echo",
            "content": "result for run_in_threadpool",
        },
    ]


def test_several_calls_in_one_step_each_get_a_message_in_order():
    env = Env()
    reply = tool_reply(("echo", {"query": "a"}), ("echo", {"query": "b"}))
    llm = FakeLLM([reply, "done"])
    env.run(llm)
    tool_messages = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == [c.id for c in reply.tool_calls]
    assert [m["content"] for m in tool_messages] == ["result for a", "result for b"]


def test_the_trace_records_every_step_and_call():
    env = Env(echo_text="x" * 123)
    llm = FakeLLM([tool_reply(("echo", {"query": "a"})), "Answer."])
    result = env.run(llm)
    t = result.trace
    assert t.question == QUESTION and [s.index for s in t.steps] == [1, 2]
    record = t.steps[0].tool_calls[0]
    assert (record.name, record.arguments, record.result_chars) == ("echo", {"query": "a"}, 123)
    assert record.error is None and record.duplicate is False
    assert t.steps[0].stop_reason == "tool_use" and t.steps[1].stop_reason == "end_turn"
    assert t.steps[1].text == "Answer." and t.usage == result.usage


def test_a_tool_latency_is_measured_with_the_clock_in_milliseconds():
    env = Env(tool_seconds=2.5)
    result = env.run(FakeLLM([tool_reply(("echo", {"query": "a"})), "done"]))
    assert result.trace.steps[0].tool_calls[0].latency_ms == pytest.approx(2500.0)


def test_the_elapsed_time_comes_from_the_clock():
    env = Env()
    result = env.run(advancing(env, ["done"], seconds=7.0))
    assert result.elapsed_s == pytest.approx(7.0)


# ------------------------------------------------------------------ what the model gets wrong
def test_a_tool_error_goes_back_to_the_model_and_the_loop_goes_on():
    env = Env()
    llm = FakeLLM([tool_reply(("boom", {"query": "a"})), "Recovered."])
    result = env.run(llm)
    assert result.stop_reason == "answered" and result.answer == "Recovered."
    message = llm.calls[1]["messages"][-1]
    assert message["role"] == "tool" and message["content"] == "Error: the page is gone"
    record = result.trace.steps[0].tool_calls[0]
    assert record.error == "the page is gone"
    assert record.result_chars == len("Error: the page is gone")


def test_an_unknown_tool_and_bad_arguments_come_back_as_errors():
    env = Env()
    llm = FakeLLM(
        [
            tool_reply(("serach", {"query": "a"})),
            tool_reply(("echo", {"wrong": 1})),
            "ok",
        ]
    )
    result = env.run(llm)
    assert result.stop_reason == "answered"
    assert env.ran == []
    assert llm.calls[1]["messages"][-1]["content"].startswith("Error:")
    assert llm.calls[2]["messages"][-1]["content"].startswith("Error:")
    assert result.trace.n_errors == 2


def test_arguments_the_client_could_not_read_are_reported_and_not_run():
    env = Env()
    bad = ToolCall("call_x", "echo", {}, error="arguments are not a JSON object: {oops")
    llm = FakeLLM([tool_reply(bad), "ok"])
    result = env.run(llm)
    assert env.ran == []
    assert llm.calls[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_x",
        "name": "echo",
        "content": "Error: arguments are not a JSON object: {oops",
    }
    assert result.trace.n_errors == 1


def test_a_repeated_call_is_not_run_again_and_the_model_is_told():
    env = Env()
    llm = FakeLLM(
        [
            tool_reply(("echo", {"query": "a", "extra": 1})),
            tool_reply(("echo", {"extra": 1, "query": "a"})),  # same call, keys in another order
            "ok",
        ]
    )
    result = env.run(llm)
    assert env.ran == [("a", 1)]  # run once
    message = llm.calls[2]["messages"][-1]
    assert message["content"].startswith("Error: you already made this exact call")
    first, second = (s.tool_calls[0] for s in result.trace.steps[:2])
    assert first.duplicate is False and second.duplicate is True and second.error is None
    assert result.trace.n_duplicates == 1 and result.n_tool_calls == 2
    assert second.result_chars == len(message["content"])


def test_a_repeat_inside_one_step_is_caught_too():
    env = Env()
    llm = FakeLLM([tool_reply(("echo", {"query": "a"}), ("echo", {"query": "a"})), "ok"])
    result = env.run(llm)
    assert env.ran == [("a", 0)]
    assert result.trace.n_duplicates == 1
    assert len([m for m in llm.calls[1]["messages"] if m["role"] == "tool"]) == 2


def test_a_different_argument_is_not_a_duplicate():
    env = Env()
    llm = FakeLLM(
        [tool_reply(("echo", {"query": "a"})), tool_reply(("echo", {"query": "b"})), "ok"]
    )
    env.run(llm)
    assert env.ran == [("a", 0), ("b", 0)]


def test_too_many_calls_in_one_step_are_cut_but_each_still_gets_a_message():
    env = Env()
    reply = tool_reply(*[("echo", {"query": f"q{i}"}) for i in range(6)])
    llm = FakeLLM([reply, "ok"])
    result = env.run(llm, Budget(max_calls_per_step=4))
    assert env.ran == [(f"q{i}", 0) for i in range(4)]
    tool_messages = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == [c.id for c in reply.tool_calls]
    assert [m["content"].startswith("Error:") for m in tool_messages] == [False] * 4 + [True] * 2
    assert "limit 4" in tool_messages[5]["content"]
    assert result.n_tool_calls == 6 and result.trace.n_errors == 2


# ------------------------------------------------------------------ the budgets
@pytest.mark.parametrize("max_steps", [1, 2, 3, 5])
def test_max_steps_is_never_exceeded(max_steps):
    env = Env()
    llm = endless()
    result = env.run(llm, Budget(max_steps=max_steps))
    assert result.stop_reason == "max_steps" and result.answer is None
    assert len(llm.calls) == max_steps == result.steps


def test_the_last_step_may_still_run_its_tools_but_no_further_model_call_is_made():
    env = Env()
    llm = endless()
    env.run(llm, Budget(max_steps=2))
    assert len(env.ran) == 2 and len(llm.calls) == 2


def test_max_tokens_stops_the_loop_after_the_call_that_used_them_up():
    env = Env()
    llm = FakeLLM(
        responder=lambda s, m: Completion(
            "",
            Usage(400, 100),
            "m",
            "tool_use",
            0.0,
            tool_reply(("echo", {"query": str(len(m))})).tool_calls,
        )
    )
    result = env.run(llm, Budget(max_tokens=1000))
    assert result.stop_reason == "max_tokens" and result.answer is None
    assert len(llm.calls) == 2 and result.usage == Usage(800, 200)


def test_one_huge_call_stops_the_loop_straight_away():
    env = Env()
    huge = Completion(
        "", Usage(5000, 0), "m", "tool_use", 0.0, tool_reply(("echo", {"query": "a"})).tool_calls
    )
    llm = FakeLLM([huge], responder=lambda s, m: tool_reply(("echo", {"query": "b"})))
    result = env.run(llm, Budget(max_tokens=100))
    assert result.stop_reason == "max_tokens" and len(llm.calls) == 1


def test_timeout_stops_the_loop_between_model_calls():
    env = Env()
    llm = advancing(env, [], seconds=40.0)
    result = env.run(llm, Budget(timeout_s=100.0, max_steps=50))
    assert result.stop_reason == "timeout" and result.answer is None
    assert len(llm.calls) == 3  # 40, 80 and 120 seconds: only the third call ends past the limit
    assert result.elapsed_s == pytest.approx(120.0)


def test_slow_tools_count_towards_the_timeout():
    env = Env(tool_seconds=60.0)
    llm = endless()
    result = env.run(llm, Budget(timeout_s=100.0, max_steps=50))
    assert result.stop_reason == "timeout" and len(llm.calls) == 2


def test_an_answer_in_time_is_not_a_timeout():
    env = Env()
    result = env.run(advancing(env, ["fine"], seconds=99.0), Budget(timeout_s=100.0))
    assert result.stop_reason == "answered"


def test_the_default_budget_applies_when_none_is_given():
    env = Env()
    llm = endless()
    result = env.run(llm)
    assert result.stop_reason == "max_steps" and len(llm.calls) == Budget().max_steps


# ------------------------------------------------------------------ failures of the model call
def test_a_failing_model_call_ends_the_run_without_raising():
    env = Env()
    llm = FakeLLM(
        [tool_reply(("echo", {"query": "a"})), ServerError("overloaded", status_code=529)]
    )
    result = env.run(llm)
    assert result.stop_reason == "model_error" and result.answer is None
    last = result.trace.steps[-1]
    assert last.error == "overloaded" and last.text == "" and last.tool_calls == ()
    assert last.usage == Usage() and result.steps == 2


def test_a_fatal_error_also_ends_the_run():
    result = Env().run(FakeLLM([AuthError("bad key", status_code=401)]))
    assert result.stop_reason == "model_error" and result.trace.steps[0].error == "bad key"


def test_a_bug_is_not_swallowed():
    with pytest.raises(AssertionError):
        Env().run(FakeLLM([]))  # the fake has no reply left: a programming error


@pytest.mark.parametrize("text", ["", "   \n"])
def test_an_empty_reply_without_tool_calls_is_a_model_error(text):
    result = Env().run(FakeLLM([text]))
    assert result.stop_reason == "model_error" and result.answer is None
    assert result.trace.steps[0].error == "empty reply"


def test_a_reply_cut_off_by_the_output_limit_is_truncated():
    cut = Completion("The answer starts here and", Usage(5, 5), "m", "max_tokens")
    result = Env().run(FakeLLM([cut]))
    assert result.stop_reason == "truncated"
    assert result.answer == "The answer starts here and"


# ------------------------------------------------------------------ invariants over random runs
STOPS = {"answered", "truncated", "max_steps", "max_tokens", "timeout", "model_error"}


def random_replies(rng):
    replies = []
    for _ in range(30):
        usage = Usage(rng.randint(0, 150), rng.randint(0, 60))
        kind = rng.random()
        if kind < 0.45:
            calls = [
                (rng.choice(["echo", "echo", "boom", "nope"]), {"query": rng.choice("abc")})
                for _ in range(rng.randint(1, 6))
            ]
            replies.append(tool_reply(*calls, text=rng.choice(["", "hmm"]), usage=usage))
        elif kind < 0.55:
            replies.append(tool_reply(("echo", {"query": 5}), ("echo", {}), usage=usage))
        elif kind < 0.7:
            replies.append(Completion(rng.choice(["Answer [1].", "", "cut"]), usage, "m"))
        elif kind < 0.75:
            replies.append(ServerError("down"))
        else:
            replies.append(tool_reply(("echo", {"query": rng.choice("ab")}), usage=usage))
    return replies


def check_pairing(messages):
    """Every assistant turn with tool calls is followed by one tool message per call, in order."""
    i = 0
    while i < len(messages):
        message = messages[i]
        if message["role"] == "assistant" and message.get("tool_calls"):
            ids = [c["id"] for c in message["tool_calls"]]
            following = messages[i + 1 : i + 1 + len(ids)]
            assert [m["role"] for m in following] == ["tool"] * len(ids)
            assert [m["tool_call_id"] for m in following] == ids
            i += len(ids)
        i += 1


@pytest.mark.parametrize("seed", range(300))
def test_no_budget_is_ever_exceeded_whatever_the_model_does(seed):
    rng = random.Random(seed)
    budget = Budget(
        max_steps=rng.randint(1, 5),
        max_tokens=rng.randint(50, 400),
        timeout_s=rng.choice([30.0, 1000.0]),
        max_calls_per_step=rng.randint(1, 3),
    )
    env = Env(tool_seconds=rng.choice([0.0, 5.0]))
    llm = advancing(env, random_replies(rng), seconds=rng.choice([0.0, 4.0]))
    result = env.run(llm, budget)

    assert result.stop_reason in STOPS
    assert len(llm.calls) <= budget.max_steps
    assert len(result.trace.steps) == len(llm.calls) == result.steps
    if result.stop_reason in {"max_steps", "max_tokens", "timeout", "model_error"}:
        assert result.answer is None
    else:
        assert result.answer is not None
    if result.stop_reason == "max_steps":
        assert len(llm.calls) == budget.max_steps
    # tokens: the check comes before every call, so only the last call can carry the total past
    # the limit, and by at most its own size (a scripted call uses at most 150 + 60 tokens)
    used = result.usage.input_tokens + result.usage.output_tokens
    assert used < budget.max_tokens + 210
    for call in llm.calls:
        check_pairing(call["messages"])
    for step in result.trace.steps:
        assert len(step.tool_calls) <= 6


# ------------------------------------------------------------------ citations
def test_cited_hits_maps_valid_numbers_to_hits_and_ignores_the_rest():
    seen = [Hit("a.md", 1, 5), Hit("b.py", 10, 20), Hit("c.md", 3, 4)]
    assert cited_hits("A [2] and B [9], also [1, 2] and [0].", seen) == [seen[1], seen[0]]
    assert cited_hits("no citations here", seen) == []
    assert cited_hits("[1]", []) == []
    assert cited_hits("code `x[3]` is not a citation, but [3] is", seen) == [seen[2]]


# ------------------------------------------------------------------ the prompt
def test_the_agent_prompt_has_the_right_shape():
    template = load_prompt("agent")
    assert template.name == "agent" and template.version == 1
    assert template.placeholders() == {"question"}
    system = template.system
    assert "CANNOT_ANSWER" in system
    assert "[1]" in system or "[n]" in system
    for tool in ("search", "read_chunk", "read_file", "list_symbols"):
        assert tool in system
    assert "instruction" in system.lower() and "data" in system.lower()
    assert "{{" not in system  # the system part is a constant
