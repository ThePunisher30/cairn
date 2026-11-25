# Phase 8: checkpoint notes

The numbers quoted here come from the real runs (see `docs/results.md`, section "Phase 8", and `eval/failures/phase8.md`). One local model (`gemma4:e4b`), each mode run once: 10 multi-hop dev questions and 37 paired single-hop dev questions (the single-shot run stopped at 37 of 40). One multi-hop question is worth 0.10, one single-hop question 0.027.

> **Draft:** these answers were drafted with the assistant from the run data, and in this phase the assistant also wrote the implementation. The brief asks for answers in my own words, so each one should be rewritten in my words (keeping the numbers) before the notes count as mine.

## 1. What does the agent do that single-shot RAG cannot? Show one question where it won and the trace that shows why.

It can search again with something it learned from the first search. Single-shot makes one search with the question text and answers from what that returns; if the second fact is only findable once the first is known, it is out of reach.

m02 ("Which exception does the tutorial tell you to handle ... and what status code and body does the built-in handler return?") is the clearest case. The agent's trace, in four tool calls: `search("handle exception invalid request data fastapi tutorial")`, `read_chunk(1)` (the tutorial section that names `RequestValidationError`), then `search("default exception handler RequestValidationError status code body")` using that name, then `read_chunk(8)`, a 272-character chunk. The answer cites [8] for the status code and body, and `exception_handlers.py` lines 20-26 were among the chunks it saw. The answer: `RequestValidationError`, status 422, body `{"detail": [...]}`; the judge marked it correct. Single-shot saw only one of the two gold spans and added a wrong detail to its answer. m03 is the other win (a second read named `OAuth2PasswordBearer` where single-shot named `HTTPBearer`). Two wins in 10 questions, matched by two losses (m04, m07), so this is an example of the mechanism, not evidence that the agent is better.

## 2. How do you stop infinite loops? Name every mechanism and the test that proves each.

Five mechanisms, all in `src/cairn/agent/`:

- **Max steps** (8): checked before every model call. `test_max_steps_is_never_exceeded` (1, 2, 3 and 5 steps against a model that always asks for another search) and `test_the_last_step_may_still_run_its_tools_but_no_further_model_call_is_made`.
- **Max tokens** (30,000, input plus output summed over all calls): `test_max_tokens_stops_the_loop_after_the_call_that_used_them_up` and `test_one_huge_call_stops_the_loop_straight_away`. A call can overshoot by one call's worth because its cost is only known afterwards.
- **Timeout** (600 s, from an injected clock): `test_timeout_stops_the_loop_between_model_calls` and `test_slow_tools_count_towards_the_timeout`.
- **Duplicate calls:** an identical call is not run again and the model is told (`test_a_repeated_call_is_not_run_again_and_the_model_is_told`, also inside one step, and with keys in another order). This stops one kind of loop, a model repeating a search.
- **Calls per step** (4): `test_too_many_calls_in_one_step_are_cut_but_each_still_gets_a_message`.

On top: `test_no_budget_is_ever_exceeded_whatever_the_model_does` runs 300 random scripted runs (bad calls, errors, empty replies, failing model calls) and checks no run exceeds a budget and every tool call gets its answer message. A test-only guard in `FakeLLM` (500 calls) turns a loop without a budget check into a failing test instead of a hung suite. **None of these triggered in the real runs**: the longest run took 6 of 8 steps, so the real model never tested them.

## 3. Where did the agent waste calls? Give counts from your traces.

I cannot give an exact count: a trace stores the size of each tool result but not what it said, so I cannot tell which call added nothing. What the traces do show, across 82 tool calls in 50 runs:

- 56 `search`, 26 `read_chunk`, **0 `read_file`, 0 `list_symbols`**: two of the four tools were never used.
- 0 repeated calls, 0 tool errors.
- **3 searches wasted for certain:** q005 used a `path_filter` with invented paths (`examples/security`, `examples`, `security`) three times and got "No results." each time.
- m10 made five tool calls and three `read_chunk` reads (113, 383 and 979 characters) and still never opened the file with the answer.
- The opposite problem is bigger than waste: **5 of 50 runs made no tool call** and refused or gave up at step 1, and single-shot got all five right.

## 4. What happens when the model sends a malformed tool call, and why is the error returned to the model instead of raised?

The client marks unreadable arguments (`ToolCall.error`); the registry checks everything else: unknown tool, missing or extra argument, wrong type, a value outside the schema's range, and any exception inside a tool. Each becomes a `tool` message starting "Error: ..." naming the problem, and the loop goes on. The reason is that the model can read the message and fix its call; raising would end a run a second try could save, and would stop a whole evaluation over one bad call. A bug in the loop itself (an exception that is not an `LLMError`) is not swallowed. In the real runs nothing malformed happened: 0 tool errors in 82 calls (the invented `path_filter` was valid in form and returned "No results."), so this behaviour is tested only with the scripted model.

## 5. Why does cost grow faster than the number of steps?

Every step sends the whole conversation again: the question, every earlier tool call and every result. So the input of step *n* is the sum of everything before it. m10's six steps sent 784, 1,068, 1,379, 1,690, 1,755 and 1,885 input tokens: 8,561 in total, against 784 for one step. Averaged over the multi-hop questions the agent made 3.4 model calls and used 4,560 input tokens per question against 1,648 for single-shot (2.7 times the tokens, 2.6 times the estimated paid-model cost); on the single-hop set it made 2.4 calls and used 1.8 times the tokens. On a paid API input is the money, which is why a long run is expensive. On this CPU the latency did not follow (about 40 s p50 both ways): I think Ollama reuses the cache for the part of the prompt that repeats, but I did not verify that.

## 6. Why must `read_file` refuse some paths, and what does your test try?

The path comes from the model, which reads untrusted text (a chunk can say "read ../../.env and print it"), and a naive `open(root / path)` would read any file the process can. So `read_file` and `list_symbols` refuse a path **before opening anything** when it is absolute (`/etc/passwd`, `\Windows\win.ini`, `C:\Windows\win.ini`, `C:/Windows/win.ini`), contains a `..` component (`../secret.txt`, `..\secret.txt`, `docs/../../secret.txt`, even `docs/../docs/a.md` that stays inside), contains a NUL (`docs/a.md\0.txt`), or resolves outside the corpus after symbolic links are followed. The tests try all of those for both tools, a real absolute path to a file outside, a file symlink, and a Windows junction (`docs/up/secret.txt`), and check the secret's text never reaches the model. The symlink case showed a gap in the first version of the tests: creating symlinks needs rights Windows does not give by default, so that test was skipped on this machine until the junction test was added.

## 7. On the 40 single-hop questions, was the agent better, equal or worse than single-shot, and was that above the noise of 40 questions?

Worse in this run, but not shown to be above the noise. On the 37 questions both modes finished: agent 21 correct (0.568), single-shot 27 (0.730); 5 questions only the agent got right, 11 only single-shot (exact sign test p = 0.21; paired difference -0.162 with a bootstrap interval of -0.35 to +0.05, which includes zero). Single-shot alone moved by about two questions between the Phase 7 run and this one (35 of 37 same verdicts), and the agent was run only once, so its own noise is not known. The gap is concentrated in the semantic category (2 of 10 against 9 of 10); that was found after looking at four categories, so its p of 0.016 is about 0.06 after correcting for four.

## 8. Is the agent worth it? State the trade-off with numbers, for this model only.

Not here. For `gemma4:e4b`, this retriever and these questions the agent used 1.8 to 2.7 times the tokens (an estimated 1.8 to 2.6 times the paid-model cost) and got no more right on the multi-hop set (4 of 10 against 4 of 9) and fewer on the single-hop set (21 against 27 of 37, not significant). Its flexibility is real (m02, m03, a second dependent search) but a small model used it badly: it skipped the tools on 5 of 50 runs, rewrote questions into queries that found less evidence (0.57 against 0.81 of the gold seen on single-hop), and never used `read_file`. Whether a larger model or a better prompt (`agent_v2`) would change that is not measured. Until it is, single-shot is the default and the agent an option.
