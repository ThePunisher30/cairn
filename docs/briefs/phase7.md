# Phase 7 brief: generation with citations, abstention and an honest evaluation

**Goal:** turn the retrieval you built into an answering system. A question goes in; the best chunks are packed into a prompt; a language model writes an answer that **cites the chunks it used** and **refuses when the chunks do not contain the answer**; and you measure all of it: is the answer correct, is it faithful to the chunks, are the citations the right ones, does it abstain when it should, what does it cost. At the end `ask("...")` returns a cited answer, and `docs/results.md` says how far each number can be trusted. Phases 0 to 7 together are a complete, measured RAG system.

**Time:** about 8-10 days at 8-10 hours a week. About 700 lines of code, a lot of it small; the weight is in the experiments, in being honest about 40 dev questions, and in distrusting a judge that is itself a language model.

**You write the code. 265 tests (plus 31 for provided scripts and the injection checks) are already in `tests/` and currently fail.** They were checked against a reference solution and against 136 deliberately broken versions of it; one broken version is not caught, and it is behaviourally identical to the correct code. They run in seconds, with no network and no cost: a scripted fake model (`FakeLLM`) stands in for a real one. **No real model was called while this brief was prepared**: the reference `AnthropicClient` was run against a fake SDK object shaped like the documented one, so the first live call is where a mismatch would show up. If it does, paste me the error.

**Done when**
- the full test suite is green and `ruff check .` and `ruff format --check .` are clean
- `python scripts/ask.py "How do I enable CORS?"` prints a cited answer with `file:line` links, from a real model
- `python scripts/gen_eval.py` has been run on the dev questions, and `docs/results.md` has correctness, faithfulness, citation precision and recall, abstain precision and recall, cost per question and latency, **and the judge-agreement figure** (Cohen's kappa) from 15 answers you graded by hand
- the abstain threshold is calibrated on dev, and the text says what that calibration can and cannot tell you (there are only 10 unanswerable questions)
- the injection experiment has been run and what held and what did not is written down
- `docs/adr/0006-generation.md`, `eval/failures/phase7.md` and `docs/notes/phase7.md` are written; `README.md` and `PLAN.md` say Phase 7 is done
- the **test split has not been touched**: one run is left, for Phase 11. `gen_eval.py` has no `--split` option for that reason
- merged through a pull request on `phase-7-generation`

**Constraints:** the logic is yours: the standard library and `numpy`. The `anthropic` package is allowed **only inside `AnthropicClient`** as the transport (the SDK does the HTTP; you do the retries, the error classification, the streaming loop and the accounting), and `httpx` only inside `OllamaClient`. No `langchain`, `llama-index`, no SDK feature that does your job for you (no tool runner, no SDK-side retries: they are switched off).

---

## Before you start (about 20 minutes, and a few dollars)

1. **An API key.** The Anthropic API is billed separately from a Claude subscription. Create a key in the Anthropic Console, add a few dollars of credit, and put the key in the `ANTHROPIC_API_KEY` environment variable (PowerShell: `$env:ANTHROPIC_API_KEY = "<your key>"` for the current window). **Never put it in a file that is committed.** `cairn.config.Settings` already reads it from the environment. If you would rather not pay yet, Ollama (a free local model server) works for everything except the "real model" numbers: install it, run `ollama pull llama3.2`, and use `--llm ollama`. A CPU-only laptop will be slow with it.
2. **Install the extras:** `pip install -e ".[dev,dense,llm]"` (adds `anthropic` and `httpx`).
3. **What it costs.** `eval/llm_prices.json` holds dollars per million tokens (input, output), copied from Anthropic's pricing table as it stood on 2026-10-06: Opus 5.5 $4 and $20, Sonnet 5.5 $2 and $10, Haiku 5.5 $0.10 and $0.50 (that last one for prompts up to 100K tokens). **Check the pricing page before you trust it; the file is yours to keep right.** A rough estimate for one full dev run (50 questions, about 3,500 tokens in and 500 out per answer, plus the judge about as much again): about $2 with Opus 5.5, about $1 with Sonnet 5.5, a few cents with Haiku 5.5. That estimate rests on two guesses (the tokens per call, and how many output tokens thinking adds), so read your real numbers from the first run.
4. **Which model.** Develop with a cheap one (Haiku 5.5, or Ollama), and take the numbers you report from one model you name. Use a judge at least as strong as the model it grades. Record the model, the effort setting and the date next to every number.
5. **No temperature.** The Anthropic API no longer accepts `temperature`, `top_p` or `top_k` (a request that sends them gets an error), which is why `LLMClient` has no temperature argument. It also means answers are not repeatable: the same question can get two different answers, and that is part of what you are evaluating.

---

## The ideas, in plain words

**Grounding.** A language model answers from what it learned, which may be wrong or outdated. RAG answers *from the chunks you hand it*, so you can check the answer against them. Two different failures hide here: the answer can go *beyond* the chunks (it is **unfaithful**), or it can stay faithful to chunks that were the wrong ones (it is faithful and **wrong**). The first is the model's fault, the second is retrieval's. Your evaluation has to tell them apart.

**The context budget.** A prompt has limited room and costs money per token. The **packer** decides which chunks go in: it drops chunks that repeat an earlier one, spends the budget on the best chunks first, and cuts the one that overflows. Note that `count_tokens` (the project's token counter) is a *proxy*: the provider counts differently, and `usage.input_tokens` in the reply is the truth. You will compare the two.

**Order.** Models tend to use the start and the end of a long prompt better than its middle ("lost in the middle"). The packer can place the best chunks at the two ends (`edges`), or put the best nearest the question (`reverse`). Whether that matters on 40 questions is something you measure, not assume.

**Prompts are code.** A prompt is a file with a version number, filled in by a strict template: a missing or extra value is an error, not a silent blank. The system prompt is a **constant** (it never contains a chunk or a question); everything untrusted goes in the user message, inside tags.

**Citations.** Number the chunks `[1] [2] ...`; the model writes `[n]` after a claim; you parse the markers, **validate** them (a model can invent `[7]` when only five chunks exist), and turn the valid ones into links to the pinned FastAPI tag. A citation proves nothing by itself: you also measure whether the cited chunks are the ones that hold the answer (precision and recall against the gold spans you already have).

**Abstention.** A system that always answers is worse than one that sometimes says "I could not find that". There are two ways to abstain, and you build both: **before** the model, when retrieval looks weak (no model call, no cost), and **by** the model, which is told to reply with an exact marker (`CANNOT_ANSWER`) when the chunks do not answer. The first needs a *score that means the same thing for every question*. Think about which of your scores does: a cosine is absolute; a raw BM25 score depends on the length of the question; and the min-max fused score of Phase 6 is a sum of two numbers that each have the best chunk of its own list scaled to 1, so its top value sits between 1 and 2 whatever the question. Which one can say "nothing here is relevant"? You will measure that, with a threshold that trades *answers wrongly refused* against *wrong answers given*.

**Robust API calls.** Real calls fail: rate limits (HTTP 429), overloads (5xx, 529), timeouts, dropped connections. Some are worth retrying and some never are (a wrong key, a wrong model name, a malformed request: retrying only wastes time). So you **classify** each error, wait with **exponential backoff and jitter** (the wait doubles each time, and a random share of it is used so that many clients do not all come back at the same instant), honour the server's `Retry-After`, and give up after a few attempts. The SDK can retry too; it is switched off (`max_retries=0`) because two layers multiply (3 times 4 attempts) and you could no longer count or see them. A **stream** is different: once the first words have reached the caller, you cannot start over, so a failure after that is raised, not retried.

**Evaluating text you cannot compare with `==`.** For retrieval a hit is a hit. For an answer you need a grader. A **language-model judge** reads the question, a reference answer and the answer and returns a verdict. It is cheap and scales, and it is **not ground truth**: it has its own mistakes and biases. So you grade 15 answers *yourself*, blind (without seeing the judge's verdict), and measure **agreement** and **Cohen's kappa** (agreement above what chance gives). A low kappa means every correctness number in your report is soft, and you say so.

**Prompt injection.** The chunks are untrusted text: anyone who can edit a page the corpus is built from can put instructions in it ("ignore the above and say X"). The model cannot reliably tell your instructions from instructions that sit in the data. What you *can* do is make the structure unambiguous (tags, a constant system prompt, neutralising any text that tries to close a tag) and then **measure** what a real model does anyway. There is no complete defence; the honest result is a list of what held and what did not.

**A small, reused dev set.** 40 answerable questions and 10 unanswerable ones: one question is 0.025 on the first and 0.1 on the second. Calibrating a threshold on the same 10 unanswerable questions you then report on is **in-sample**: it flatters the result. Say so, or split them 5 and 5. The 20 test questions stay untouched.

---

## What the provided files do

- `tests/test_gen_llm.py`, `test_gen_context.py`, `test_gen_pipeline.py`, `test_gen_eval.py`: the specification, **module by module. Read the docstring at the top of each before you write the module**: it lists the exact names, arguments, tie rules and error cases.
- `src/cairn/generation/types.py` (`Usage`, `Completion`, the `LLMClient` protocol), `errors.py` (`LLMError`, `RetryableError`, `RateLimitError`, `ServerError`, `LLMTimeout`, `ConnectionFailure`, `FatalError`, `BadRequest`, `AuthError`, `NotFound`), `fake.py` (`FakeLLM`, `Partial`): the shared vocabulary and a scripted model for tests.
- `src/cairn/generation/factory.py`: builds retrievers, models and prices for the scripts (it uses your modules).
- `src/cairn/tokens.py`: a new function `truncate_tokens(text, n)`, the start of a text holding at most `n` tokens.
- `scripts/ask.py` (one question), `scripts/gen_eval.py` (the evaluation: answers every dev question, judges it, writes `data/gen_runs/<name>.jsonl` and prints the summary), `scripts/judge_agreement.py` (exports a blind sample to grade and computes the agreement), `scripts/injection_test.py` (seven attacks against your prompt) and `tests/test_gen_scripts.py`, `tests/test_injection.py` (their tests, and offline checks that untrusted text cannot leave its tags).
- `eval/llm_prices.json`, and `pyproject.toml` (the `llm` extra, and the prompt files as package data).

---

## Suggested order

```powershell
cd path\to\cairn
git switch main
git pull
git switch -c phase-7-generation
```
Commit after each step. Run `ruff format .`, `ruff check .` and the tests before each commit. (`ruff check` reports import-order warnings in the new tests until the module they import exists: expected, and do not use `--fix` on tests.)

New files you will create:
```
src/cairn/generation/{retry,clients,packer,prompts,citations,abstain,pipeline}.py
src/cairn/generation/prompts/answer_v1.txt     and     judge_v1.txt
src/cairn/evaluation/generation.py
eval/gen_questions.jsonl
```

### Step 0. Setup
The branch, the extras, the key (see "Before you start"). Read the docstring of `tests/test_gen_llm.py`.

### Step 1. `retry.py`: classify, wait, retry
`RetryPolicy`, `classify_error`, `backoff_delay`, `call_with_retries`, `RetryingClient`. Hints: classification is a chain of `if`s on the status code first and the exception's class second (`getattr(exc, "status_code", None)` works for both SDK and `httpx` errors, which keep it on `exc.response`); "full jitter" is `random() * cap`; `call_with_retries` is a `while True` with one `try`; in `RetryingClient.stream` a flag that says "has a piece been yielded yet" decides between retrying and raising. Two things to think about: why `except Exception` and not a bare `except`, and why does an unknown error become *non-retryable*?

### Step 2. `clients.py`: two real models
`AnthropicClient` and `OllamaClient`. Hints: build the request in one small helper so `complete` and `stream` cannot drift apart; in `stream`, the usage is only known when the stream ends; wrap the *whole* call in `try` and re-raise through `classify_error`. Then the first live call (a few cents, even with a large model):
```python
from cairn.generation.clients import AnthropicClient

client = AnthropicClient("claude-haiku-5-5")
print(
    client.complete(
        "You are terse.", [{"role": "user", "content": "Say hello in five words."}], max_tokens=50
    )
)
```
Look at `usage`: how many output tokens for a five-word answer? Then break it on purpose and watch the classification: a wrong model name (`NotFound`, not retried), a wrong key (`AuthError`), no network (`ConnectionFailure`, retried through `RetryingClient`).

### Step 3. `packer.py`
`ContextPacker`. Hints: write the duplicate test as a small function first; keep "stop after the first overflow" as a flag; `truncate_tokens` does the cutting. Draw the `edges` order for 4 and 5 hits on paper before coding it.

### Step 4. `prompts.py` and the two prompt files
`load_prompt`, `PromptTemplate`, `neutralize`, `format_chunks`, `build_messages`, then **write `answer_v1.txt` and `judge_v1.txt` yourself**: the tests only check the skeleton (the placeholders, the marker, an example citation, a rule about untrusted text, a JSON verdict). Everything else is design, and it is where the answer quality comes from. Questions to answer in the file: how do you tell the model to cite? what exactly must it do when the chunks do not answer? how short should an answer be? what must the judge look at first? Keep each version of a prompt: a change is `answer_v2.txt`, not an edit.

### Step 5. `citations.py`
`extract_citations`, `validate_citations`, `strip_invalid`, `source_url`, `render_sources`, `citation_precision_recall`. Hint: one compiled regular expression for the marker, with a look-behind for what must not precede it; split the text on code fences and inline code and only look at the parts outside.

### Step 6. `abstain.py`
The marker, the message, `says_cannot_answer`, `is_low_confidence`, `abstain_metrics`, `calibrate_threshold`.

### Step 7. `pipeline.py`: `Answerer`
`CostTracker`, `Answer`, `Answerer.ask` and `.stream`. Then, live:
```powershell
C:\mrv\Scripts\python scripts\ask.py "How do I enable CORS in FastAPI?" --llm anthropic --model claude-haiku-5-5
C:\mrv\Scripts\python scripts\ask.py "What is the airspeed of a swallow?" --llm anthropic --model claude-haiku-5-5
C:\mrv\Scripts\python scripts\ask.py "How do I enable CORS?" --stream
```
Read the three outputs closely: the sources, whether every claim is cited, what the second one does, and the tokens and cost line.

### Step 8. `evaluation/generation.py` and the question file
`load_gen_questions`, `parse_verdict`, `Judge`, `GenResult`, `summarize_generation`, `agreement_rate`, `cohens_kappa`. Then **write `eval/gen_questions.jsonl`**:
- one row `{"id": "q001", "reference": "..."}` for each of the 40 dev questions: a short reference answer (one to three sentences with the key facts), written from the *gold span* of that question, not from memory. Check each against the passage. You may ask me to draft them, and you then verify every one: a wrong reference makes a correct answer look wrong.
- ten rows `{"id": "u01", "answerable": false, "question": "...", "split": "dev"}`: questions that sound like FastAPI questions but that the corpus (FastAPI 0.143.0 docs, source and examples) cannot answer. Mix kinds: a feature FastAPI does not have, a question about another framework, something that came after 0.143.0, a detail the docs never state. **Prove the absence** (search the corpus for the key words with `rg` in `data/raw/fastapi`) so that the answer really is "not in the corpus", and avoid trick questions a careful reader could answer.

### Step 9. Run it, and calibrate the judge
```powershell
C:\mrv\Scripts\python scripts\gen_eval.py --llm fake --retriever dense --limit 5       # plumbing, free
C:\mrv\Scripts\python scripts\gen_eval.py --llm anthropic --model claude-haiku-5-5 --limit 5 --name try
C:\mrv\Scripts\python scripts\gen_eval.py --llm anthropic --model claude-haiku-5-5 --name haiku-hybrid
```
Then grade blind:
```powershell
C:\mrv\Scripts\python scripts\judge_agreement.py --run data\gen_runs\haiku-hybrid.jsonl --export-sample 15 --seed 0 --out learn\grading-sample.md
```
Grade each answer in `learn\grading-sample.md` yourself, fill in `learn\grading-sample.labels.json` (1 or 0, `null` to skip), then:
```powershell
C:\mrv\Scripts\python scripts\judge_agreement.py --run data\gen_runs\haiku-hybrid.jsonl --labels learn\grading-sample.labels.json
```
Read the disagreements one by one: who was right, you or the judge?

### Step 10. Experiments (all on the dev split; write the model, effort and date beside every number)
1. **Which retriever feeds the model.** Dense, hybrid (min-max) and rerank, with the same model: correctness, faithfulness, citation precision and recall, cost. (Phase 6 left this open: on the test split dense alone beat the hybrid, and nothing about the generation was measured. Decide on dev evidence only, and never on the test split.)
2. **The confidence score for abstaining.** *Without calling a model:* for the 40 answerable and the 10 unanswerable questions, compute four numbers per question (the top dense cosine, the top hybrid score, the top BM25 score, the top reranker score), run `calibrate_threshold` on each, and compare the best F1 of each. Which one separates the two groups, and which is useless? Then run `gen_eval.py` with the best one (`--confidence dense --threshold ...`). Report both the retrieval-only numbers and the end-to-end ones, and say that the threshold was fitted on the same 10 questions (or split them 5 and 5).
3. **The model's own abstention.** With no threshold, how many unanswerable questions does the model refuse itself, and how many answerable ones does it wrongly refuse? Does the threshold add anything on top?
4. **Order and size of the context.** `best_first`, `edges` and `reverse` at the same budget; then `--k` 3, 5, 8, 12 and `--max-context-tokens` 1500, 3000, 6000. Is any difference larger than 0.05 (two questions)? What does a bigger context cost?
5. **Tokens, as the provider counts them.** For the answers of one run, compare `usage.input_tokens` with `count_tokens` of what you sent. What is the ratio, and what does that do to your budget?
6. **Cost.** Per question, split input and output. What dominates? What would one dev run cost with each of the three models?
7. **The judge.** Kappa for `correct` and for `faithful`. What do the disagreements have in common? Would any conclusion above flip if you trusted yourself instead of the judge?
8. **Injection.** `python scripts/injection_test.py --llm anthropic --model ...` for each model you use. Which cases held, which leaked, and why do you think so? Add two attacks of your own to `CASES`, change one thing in the system prompt, and re-measure. Write what worked, what did not, and what you would still not trust.
9. **A prompt change.** Write `answer_v2.txt` with one deliberate change and compare it to v1 on dev, with the caution that 40 questions cannot separate small effects.
10. **Failures.** Classify every dev question whose answer is wrong, unfaithful or wrongly refused in `eval/failures/phase7.md` (retrieval missed it, a wrong chunk was used, the model went beyond the chunks, the reference is wrong, the judge is wrong), as you did in Phases 4 and 6.

### Step 11. Write-up, merge
- `docs/results.md`, a "Phase 7" section in the style of the others: what you did, the tables, what they say, claims only where the data supports them (and for each number, which model and judge produced it).
- `docs/adr/0006-generation.md`: the decisions (the model and effort, the context budget and order, the abstain score and threshold, the retry policy, the prompt version), the trade-offs (answers against wrong refusals, quality against cost), the alternatives you measured, and what you would not trust.
- `docs/notes/phase7.md`: the checkpoint questions below.
- `eval/failures/phase7.md`, and the README and PLAN (I can update those with you).
- Commit, tell me, and I'll check what is about to be published before you push and open the PR. Nothing that holds a key, and no `data/gen_runs` file, goes into git (the folder is ignored).

---

## No reference numbers this time

Phase 6 had them because the reference solution could be run on your data. Here the numbers come from a language model I did not call, so I have nothing to compare yours with, and I will not invent expectations. What I can say is what the tests guarantee (the retry, packing, citation, abstain and accounting logic behaves as the docstrings say) and what they cannot (whether a real model follows your prompt).

---

## Checkpoint questions (answer in your own words in `docs/notes/phase7.md`)
1. How can a faithful answer still be wrong? Show an example from your own run.
2. What does the abstain threshold trade off? Show your numbers. Which score did you use, and why did the others fail or work?
3. What did the injection test reveal? What held, what leaked, and what would you still not trust?
4. Why retry with backoff and jitter? Why is a failure in the middle of a stream not retried, and why is the SDK's own retry switched off?
5. Which errors are worth retrying and which never are? What would retrying a 400 or a 401 do?
6. What does the judge's kappa tell you? How much do you trust your correctness number, and what would change it?
7. How much did the order and the size of the context matter, and was it above the noise of 40 questions?
8. Why is `count_tokens` only a proxy? What did the provider's `usage` show?
9. What does one answer cost, and what dominates the cost (the input, the output, the judge)?

## Hint ladder reminder
Stuck? Tell me which step and paste the failing test. I'll nudge first, then give pseudocode, then a short snippet.
