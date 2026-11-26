# Phase 8 brief: an agent loop, and an honest comparison with single-shot RAG

**Goal:** let the model decide *what to look up and when to stop*. In Phase 7 the pipeline was fixed: retrieve once, answer once. Here the model gets **tools** (`search`, `read_chunk`, `read_file`, `list_symbols`) and runs in a loop you write: ask the model, run the tools it requests, give it the results, repeat until it answers or a budget runs out. Then you measure whether this beats single-shot RAG on questions that need several lookups, what it costs, and where it does worse.

**Time:** about 5-7 days. About 500 lines of code. The weight is in the bookkeeping (budgets, traces, failure handling) and in the experiment, not in the loop itself, which is about 30 lines.

**Done when**
- the test suite is green and `ruff check .` and `ruff format --check .` are clean
- the loop **never exceeds a budget** (steps, tokens, wall-clock), proven by tests against a scripted `FakeLLM`
- malformed tool calls, tool errors and repeated identical calls are handled and tested
- every run writes a structured trace (`agent/trace.py`)
- 15 multi-hop questions are added to the eval, with gold spans for every hop
- `python scripts/agent_eval.py` and the single-shot run on the **same questions** are in `docs/results.md`: accuracy, latency, tokens, calls per question, and the cases where the agent did worse
- `docs/adr/0007-agent-loop.md`, `eval/failures/phase8.md`, `docs/notes/phase8.md` are written; `README.md` and `PLAN.md` say Phase 8 is done
- the **test split has not been run**: one run is left, for Phase 11
- merged through a pull request on `phase-8-agent`

**Constraints:** standard library and `numpy`. No agent framework (no langchain, llama-index, no SDK tool runner, no "agent" helper from any library). The loop is yours. `httpx` only inside `OllamaClient`, `anthropic` only inside `AnthropicClient`, no SDK-side retries. Free local model by default (`gemma4:e4b`, thinking off). No paid run unless you ask for one.

---

## Before you start

1. **Tool use works locally.** A one-call probe on 2026-11-05 (Ollama 0.40.14, `gemma4:e4b`, `think=false`): `/api/show` lists the `tools` capability, the model answered a question with a well-formed `search` call (`{"query": "fastapi sync dependency thread pool"}`), and after a tool result was sent back it wrote a final answer. The first call took 50 s (model loading), the second 15.6 s. **One probe is not a reliability measure**: how often a small model emits a malformed call or loops is exactly what you will count.
2. **Cost:** $0 on the local model. Time is the cost: every loop step is a full model call, and a CPU reads about 25 prompt tokens a second, so a five-step run on a growing prompt can take minutes. Plan runs of 10-25 questions, not hundreds.
3. **The agent needs an interface change.** `LLMClient.complete(system, messages, max_tokens)` had no tools. The types are extended for you; you teach the two clients to send tools and read tool calls (see "What changes in `generation`").

---

## The ideas, in plain words

**An agent is a loop.** There is no magic: `while not done: reply = model(messages, tools); if reply asks for tools: run them and append the results; else: done`. Everything else is bookkeeping, and the bookkeeping is the difficult part.

**Tool use.** You describe each tool to the model with a name, a description and a JSON schema for its arguments. The model never runs anything. It returns a *request* ("call `search` with these arguments"); **your code** runs it and sends the result back as a message with the role `tool`. The model sees only text.

**The model's output is untrusted input.** A small model will sometimes send arguments that do not match the schema, name a tool that does not exist, ask for `k=10000`, or request the same call again. Your loop must validate every call and turn each failure into an *error message the model can read and fix*, not a crash.

**Budgets.** An agent can run forever and cost without limit. Three independent limits, all enforced before the next model call: **max steps**, **max tokens** (input plus output, summed over all steps, read from the provider's `usage`), and a **wall-clock timeout**. When one is hit, the loop stops and returns a *partial result with the reason*, not an exception. The prompt grows every step, so cost grows faster than the number of steps.

**Duplicate-call detection.** A model that does not find what it wants often repeats the same search. Detect an identical (tool, arguments) pair, do not run it again, and tell the model it already did that.

**Traces.** For every step record: what the model said, which tool with which arguments, the size of the result, tokens in and out, latency, and any error. Without a trace you cannot explain a failure, and the checkpoint question "where did the agent waste calls?" cannot be answered.

**When an agent is worth it.** Single-shot RAG fails when one search cannot bring all the facts ("which function handles X, and what does the dependency Y it uses do?"): the second fact is only searchable once the first is known. The agent can search twice. It also costs more calls and more time, and a small model can wander. Whether it is *worth it* is a measurement, and "the agent did worse here" is a valid and useful result.

**Multi-hop questions.** A question whose answer needs facts from two or more places where the second lookup depends on the first. Write them so that a single search on the question text would not retrieve both pieces (check this: run the Phase 6 retriever and record whether it did).

---

## What changes in `generation`

**Provided** (the shared vocabulary, like the Phase 7 types): `types.py` now has `ToolSpec` (name, description, JSON schema), `ToolCall` (id, name, arguments dict, and an `error` for arguments the client could not read), `Completion.tool_calls`, and a documented neutral message format for tool turns. `fake.py`: `FakeLLM.complete` accepts `tools=`, and `tool_reply(("search", {...}), ...)` scripts a reply that asks for tools. **This is what makes the loop testable without a model.**

**You write**
- `clients.py`: `OllamaClient` sends `tools` and parses `message.tool_calls` (Ollama returns the arguments as a dict, and some models as a JSON string). `AnthropicClient` maps to the Messages API `tools`, `tool_use` and `tool_result` blocks. Both translate between our message format and the provider's. The exact rules are in the docstring of `tests/test_agent_llm.py`.
- `retry.py`: `RetryingClient.complete` passes `tools` through.

---

## What the provided files do

- `tests/test_agent_llm.py`, `test_agent_budget.py`, `test_agent_trace.py`, `test_agent_tools.py`, `test_agent_loop.py`: the specification, module by module. **Read the docstring at the top of each before you write the module**: it lists the exact names, arguments, rules and error cases.
- `tests/test_multihop_questions.py`: checks on *your* `eval/multihop_questions.jsonl` (format in its docstring). It skips until the file exists.
- The tests need no network and no model: `FakeLLM` stands in. They were checked against a reference solution and against deliberately broken versions of it.
- `scripts/agent_eval.py` and its tests were written later, in this phase, by the assistant.

---

## The pieces

```
src/cairn/agent/
  tools.py     ToolRegistry: specs, argument validation, execution -> text for the model
  loop.py      run_agent(question, llm, tools, budget) -> AgentResult
  budget.py    Budget (max_steps, max_tokens, timeout_s) and the checks
  trace.py     Step / Trace records, written as JSON lines
eval/multihop_questions.jsonl   15 questions with a gold span per hop
scripts/agent_eval.py           runs the agent on the multi-hop (and dev) questions
```

**Tools** (all read-only, all results truncated to a size limit and marked when truncated):
- `search(query, k=5, path_filter=None)`: the Phase 6 hybrid retriever; returns numbered chunks with `path:start-end`.
- `read_chunk(id)`: the full text of a chunk returned earlier.
- `read_file(path, start, end)`: lines from a file in the corpus. Reject paths outside the corpus (**path traversal** such as `..\..\secrets` must be refused, and tested).
- `list_symbols(path)`: the classes, functions and methods in a Python file with their line ranges, from the standard library `ast` module.

**The loop returns** an `AgentResult`: the final answer (or `None`), the citations, a `stop_reason` (`answered`, `max_steps`, `max_tokens`, `timeout`, `model_error`), the trace, and total usage. Reuse `Answerer`'s citation validation and abstention marker on the final text.

---

## The experiment

Run the single-shot pipeline of Phase 7 and the agent on the **same** questions with the **same** model and judge:

| Measure | Single-shot | Agent |
|---|---|---|
| Correctness (judge_v1, as in Phase 7) | | |
| Tool calls per question | 1 | |
| Model calls per question | 1 | |
| Input and output tokens | | |
| Latency p50 / p95 | | |
| Stop reasons (how many hit a budget) | | |
| Estimated cost on a paid model | | |

Also run both on the 40 Phase 7 answerable questions, where the agent should **not** be much better: if it is worse there, say so. Read at least five traces end to end and write down where calls were wasted (repeated searches, reading files it did not need, giving up early, answering before searching).

**Caveats you must state:** 15 questions is tiny (one question is 0.067); the references are drafts; the judge is a small model grading itself (Phase 7 showed a 17.5-point swing from the judge prompt alone); one model, one run. Run the agent more than once if time allows, because tool-calling models are not deterministic.

---

## Suggested order

```powershell
cd path\to\cairn
git switch main
git pull
git switch -c phase-8-agent
```
1. Read `src/cairn/generation/types.py`, `fake.py` and the docstring of `tests/test_agent_llm.py`. Then `RetryingClient` and `OllamaClient` tool support (`test_agent_llm.py`). Run a one-call probe yourself through your client. `AnthropicClient` tool support is tested with a fake SDK only: no paid call.
2. `agent/budget.py`, `agent/trace.py` (create `src/cairn/agent/__init__.py` too).
3. `agent/tools.py`: the registry first (validation, errors, truncation), then the four corpus tools, with the path-traversal guard.
4. `agent/loop.py` and `prompts/agent_v1.txt`, against the `FakeLLM` scripts in `test_agent_loop.py`.
5. Write the 15 multi-hop questions and their gold spans (`test_multihop_questions.py`), and check that one search does not already solve them.
6. `scripts/agent_eval.py` (I will write its tests when you reach this step), a small real run (3-5 questions), then the full run.
7. Experiment, failure analysis, ADR 0007, notes, README and PLAN.

Commit after each step. Run `ruff format .`, `ruff check .` and the tests before each commit.

---

## Checkpoint questions (answer in `docs/notes/phase8.md`, in your own words, with your numbers)

1. What does the agent do that single-shot RAG cannot? Show one question where it won and the trace that shows why.
2. How do you stop infinite loops? Name every mechanism (steps, tokens, time, duplicates) and the test that proves each.
3. Where did the agent waste calls? Give counts from your traces.
4. What happens when the model sends a malformed tool call, and why is the error returned to the model instead of raised?
5. Why does cost grow faster than the number of steps?
6. Why must `read_file` refuse some paths, and what does your test try?
7. On the 40 single-hop questions, was the agent better, equal or worse than single-shot, and was that above the noise of 40 questions?
8. Is the agent worth it? State the trade-off with numbers, for this model only.
