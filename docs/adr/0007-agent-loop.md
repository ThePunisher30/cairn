# ADR 0007: A hand-written agent loop with hard budgets, errors as messages, and single-shot as the default

Status: accepted, 2026-11-25. Evidence: `docs/results.md`, section "Phase 8"; `eval/failures/phase8.md`.
Every comparison below is one local model (`gemma4:e4b`), one run per set, a judge that is the same model, 10 multi-hop questions and 40 paired single-hop questions, so the claims are narrow. Section "What I would not trust" says where.

## Context

Phase 8 lets the model decide what to look up. It gets four tools (`search`, `read_chunk`, `read_file`, `list_symbols`) and a loop runs the tools it asks for until it answers or a budget runs out. The loop, the registry, the budgets and the trace are in `src/cairn/agent/`; the clients' tool support is in `src/cairn/generation/`. No agent framework and no SDK tool runner: the provider SDKs are used only inside the two clients. The project had a hard constraint of spending no money, so the model is local.

**Who wrote it.** In this phase the AI assistant wrote the implementation (clients' tool support, `agent/*`, `scripts/agent_eval.py`) and the 15 multi-hop questions at the author's request; the tests were written first and checked against a reference solution and deliberately broken variants of it (24 of 26 broken variants were caught; the two that were not, the symlink-escape check and a missing budget check that hangs, led to a new test and to a call limit in `FakeLLM`).

## Decision

- **The loop** (`run_agent`): before every model call it checks the budget; it calls the model with the tool specs; with no tool calls the reply is the answer; otherwise it runs the calls, appends one `tool` message per call and repeats. It returns a partial `AgentResult` with a `stop_reason` (`answered`, `truncated`, `max_steps`, `max_tokens`, `timeout`, `model_error`) and never raises for a budget.
- **Budgets:** `Budget(max_steps=8, max_tokens=30_000, timeout_s=600, max_calls_per_step=4)`, checked in that order before each model call. Tokens are the provider's input plus output summed over all calls; a call can overshoot by one call's worth because its cost is known only afterwards.
- **The model's output is untrusted input.** `ToolRegistry.execute` never raises an ordinary exception: unknown tool, missing or extra argument, wrong type (a bool is not an integer), out-of-range value, unreadable arguments (the client sets `ToolCall.error`) and any exception inside a tool all come back as `Error: ...` text for the model to read and fix. `KeyboardInterrupt` is not caught. Results over 6,000 characters are cut with a marker saying how much.
- **Duplicate calls:** the same tool with the same arguments (key order ignored) is not run again; the model is told. A call whose arguments could not be read is never treated as a duplicate (two different broken calls both have empty arguments).
- **`read_file` and `list_symbols` refuse paths before opening anything**: absolute paths (also `\` and drive letters), any `..` component, NUL, and anything that resolves, through symbolic links or Windows junctions, outside the corpus.
- **Citations:** `search` numbers chunks 1, 2, 3, ... per session (the same chunk keeps its number), so the model cites `[n]` and the Phase 7 citation code maps it back to a `Hit`.
- **A trace** of every step (text, each tool call with arguments, result size, error, duplicate flag, latency, tokens) is written as JSON lines.
- **`LLMClient.complete(..., tools=None)`** with one neutral message format that each client translates; `RetryingClient` passes `tools` through only when given, so a client written before Phase 8 still works.
- **Single-shot stays the default.** On this model and these questions the agent did not beat it (see below), and it costs more.

## Why

- **A loop of about 140 lines, most of it bookkeeping.** The loop itself is simple; what makes it safe is that every failure becomes a message or a stop reason. The unit tests (300 random runs against a scripted model) show no budget is ever exceeded and every tool call always gets an answer message, which the providers require.
- **Errors as messages, not exceptions:** a small model sends bad arguments, and raising ends a run that one more try might have saved. The model can read "unknown argument 'colour'" and fix its call. Raising would also let one bad call stop an evaluation of many questions.
- **Budgets checked before each call, partial result on a stop:** an agent can run forever and cost without limit, and the prompt grows each step, so cost grows faster than the number of steps. Returning the reason makes "ran out of steps" a measurable outcome instead of a crash.
- **Duplicate detection:** a model that does not find what it wants repeats a search. (It never did in 50 runs, so this mechanism is covered by tests only.)
- **Refusing paths up front:** the model's path comes from untrusted text, and `read_file` would otherwise read any file the process can, including a `.env` file.

## Alternatives measured

- **Agent against single-shot, multi-hop (10 dev questions):** 4 of 10 against 4 of 9 graded, 2.7 times the tokens and about 2.6 times the estimated paid-model cost. Neither is better at this sample size.
- **Agent against single-shot, single-hop (40 paired dev questions):** 23 against 29 correct (-0.15, interval [-0.35, +0.05], p = 0.24); semantic questions 2 of 10 against 9 of 10; 1.8 times the tokens. Single-shot moves by about 2 questions between runs.
- **Not measured:** a second agent run (its own noise), `agent_v2` (a prompt that forces a first search), the first search being the question text, a larger or paid model.

## Trade-offs

- **Flexibility against cost and predictability:** the agent can run a second search that depends on the first (m02 got the exception handler's status code that way), and it costs 2 to 3 times the tokens and 2 to 3.4 model calls instead of 1. On a CPU the latency was similar (about 40 s p50 for both), because one long single-shot prompt is slow too; on a paid API the extra calls and tokens are money.
- **The model chooses the query, and that can be worse than the question.** On single-hop questions the agent's own rewrite saw the gold span in far fewer cases than searching with the question text (0.56 against 0.79). Giving the model the choice is only worth it when the question needs a choice.
- **Honest partial answers against abstention:** several wrong agent answers say what the chunks lack instead of replying `CANNOT_ANSWER`. That is better than a confident wrong answer, and the judge marks it wrong either way.

## What I would not trust

- **Any claim that the agent is worse or better.** Ten multi-hop pairs (4 against 4) and 40 single-hop pairs (p = 0.24) cannot separate them, the agent was run once, and the judge is the same small model whose verdict moved 17.5 points with its prompt wording in Phase 7. The semantic-category gap (p about 0.06 after correcting for four categories) was found after the fact.
- **Faithfulness numbers:** the judge sees different context in the two modes.
- **The multi-hop set as a benchmark:** 15 questions drafted with an assistant, one ambiguous (m08), references not independently checked, and 9 of 10 dev questions were not solved by one search, so the set is hard for retrieval in a way that may not favour an agent.
- **"The error handling works"**: the real model never triggered a tool error, a repeat or a budget stop. Only the scripted tests did.
- **Why the agent loses on semantic questions or skips its tools:** both are guesses (query rewriting, snippet length, a prompt that does not enforce a first search) until `agent_v2` and the question-as-query run are done.

## Consequences

- Phase 9 serving should call single-shot by default and offer the agent as an option with the budgets exposed; the trace is the debugging tool.
- The next experiment is `agent_v2` (first call must be a search, the first query is the question text, use `read_file` when a chunk stops short) and a second run of both, with the same judge, before any claim about "better".
- `read_file` and `list_symbols` were never used; either the prompt must make them attractive or they should be removed from the default tool set.
- A trace that stores a preview of each tool result would let wasted calls be counted; it does not exist yet.
- The 5 test questions of the multi-hop set belong to the Phase 11 run; they were not looked at.
