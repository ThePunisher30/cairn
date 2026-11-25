"""The agent loop: ask the model, run the tools it requests, give back the results, repeat.

The model never runs anything. Its output is untrusted input: every failure of a tool call goes
back to it as an "Error: ..." message instead of ending the run, and three budgets (steps, tokens,
time) are checked before every model call so the loop always ends.
"""

import json
import time
from dataclasses import dataclass

from cairn.agent.budget import Budget, BudgetTracker
from cairn.agent.trace import Step, ToolRecord, Trace
from cairn.generation.citations import extract_citations, validate_citations
from cairn.generation.errors import LLMError
from cairn.generation.prompts import load_prompt
from cairn.generation.types import Usage


@dataclass(frozen=True)
class AgentResult:
    question: str
    answer: str | None  # None when a budget ran out or the model call failed
    stop_reason: str  # answered | truncated | max_steps | max_tokens | timeout | model_error
    trace: Trace
    usage: Usage  # summed over every model call
    elapsed_s: float

    @property
    def steps(self):
        return len(self.trace.steps)

    @property
    def n_tool_calls(self):
        return self.trace.n_tool_calls


def cited_hits(text, seen):
    """The Hits that the valid [n] citations in `text` point to (`seen[n - 1]`), first use first."""
    valid, _ = validate_citations(extract_citations(text), len(seen))
    return [seen[n - 1] for n in valid]


def _call_key(call):
    # sorted keys: the same arguments in another order are the same call
    return (call.name, json.dumps(call.arguments, sort_keys=True, default=str))


def run_agent(
    question, llm, tools, budget=None, *, system=None, max_output_tokens=1024, clock=time.monotonic
):
    budget = budget or Budget()
    if system is None:
        system = load_prompt("agent").system
    tracker = BudgetTracker(budget, clock=clock)
    trace = Trace(question)
    messages = [{"role": "user", "content": question}]
    made = set()  # calls already run in this run
    total = Usage()

    def finish(answer, reason):
        return AgentResult(question, answer, reason, trace, total, tracker.elapsed())

    while True:
        reason = tracker.exceeded()
        if reason is not None:
            return finish(None, reason)

        index = len(trace.steps) + 1
        try:
            completion = llm.complete(
                system, messages, max_tokens=max_output_tokens, tools=tools.specs()
            )
        except LLMError as exc:  # anything else is a bug and propagates
            trace.add(Step(index, "", (), Usage(), 0.0, "error", str(exc)))
            return finish(None, "model_error")
        tracker.record(completion.usage)
        total = total + completion.usage

        if not completion.tool_calls:
            text = completion.text.strip()
            error = None if text else "empty reply"
            trace.add(
                Step(
                    index,
                    completion.text,
                    (),
                    completion.usage,
                    completion.latency_ms,
                    completion.stop_reason,
                    error,
                )
            )
            if error:
                return finish(None, "model_error")
            return finish(
                text, "truncated" if completion.stop_reason == "max_tokens" else "answered"
            )

        messages.append(
            {
                "role": "assistant",
                "content": completion.text,
                "tool_calls": [
                    {"id": c.id, "name": c.name, "arguments": c.arguments}
                    for c in completion.tool_calls
                ],
            }
        )
        records = []
        for position, call in enumerate(completion.tool_calls):
            started = clock()
            duplicate, error = False, None
            key = _call_key(call)
            if position >= budget.max_calls_per_step:
                error = f"too many tool calls in one step (limit {budget.max_calls_per_step})"
                content = "Error: " + error
            elif call.error is None and key in made:
                duplicate = True
                content = (
                    "Error: you already made this exact call; its result is above. "
                    "Try something different or answer now."
                )
            else:
                if call.error is None:
                    made.add(key)
                result = tools.execute(call)
                content, error = result.text, result.error
            records.append(
                ToolRecord(
                    call.name,
                    call.arguments,
                    len(content),
                    error,
                    duplicate,
                    (clock() - started) * 1000,
                )
            )
            # every call gets a message, or the provider rejects the next request
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content}
            )
        trace.add(
            Step(
                index,
                completion.text,
                tuple(records),
                completion.usage,
                completion.latency_ms,
                completion.stop_reason,
            )
        )
