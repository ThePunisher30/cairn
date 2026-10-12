# Phase 1 brief: the yardstick

**Goal:** build the way you will *measure* every search method in this project, before building any of them. By the end you can point any retriever at 60 hand-written questions and get back numbers you trust.

**Time:** about 8-12 days at 8-10 hours a week (the code is about 3 days; reading the corpus and writing 60 good questions is the long, human part).

**You write all the code and all the questions. The tests are already in `tests/` and currently fail** (about 155 tests in total, 121 of them new).

**Done when**
- the full test suite is green, **with no skipped tests** in `tests/test_question_set.py` (a skip means your question file or the corpus is missing)
- `ruff check .` is clean
- `eval/questions.jsonl` has 60 valid questions
- `docs/results.md` has your baseline table (random, first-N, oracle)
- the work is merged through a pull request on branch `phase-1-yardstick`

---

## What changed from `PLAN.md`
`PLAN.md` put the evaluation *code* in the top-level `eval/` folder. Code that tests import has to live in the package, so:
- **code** goes in `src/cairn/evaluation/` (`metrics.py`, `questions.py`, `baselines.py`, `harness.py`)
- **data** stays in `eval/` (`questions.jsonl`, later `failures/`)
- command-line runners stay in `scripts/`

---

## The ideas, in plain words

**Why build the measuring stick first.** Without it, "I improved the search" is a feeling. With it, it is "hit@5 went from 0.41 to 0.58 on 40 questions". Every later phase is judged by this phase.

**What a question looks like.** A question, and *where the answer lives*, as a file and line range (the **gold span**):
```json
{"id": "q001", "category": "lexical", "split": "dev",
 "question": "What does the response_model_exclude_unset parameter do?",
 "gold": [{"path": "docs/en/docs/tutorial/response-model.md", "start_line": 120, "end_line": 138}]}
```
Gold is a file and lines, **not** a chunk ID, because in Phase 2 you will try many ways of cutting files into chunks, and chunk IDs change every time. Lines don't.

**What counts as a hit.** A retrieved passage is correct if it **shares at least one line** with a gold span in the same file (ranges are inclusive).

**The credit rule (this keeps the scores honest).** Walk down the ranked results. A result earns credit only if it overlaps at least one gold span that **no earlier result has already covered**. Repeats of an already-found span earn nothing, so duplicates can't inflate a score, and nDCG can never exceed 1.

**The metrics, with one worked example.** Question with gold span G. The retriever returns, in order: `[wrong, wrong, G-hit, wrong, wrong]`.

| Metric | Question it answers | Value here |
|---|---|---|
| **hit@k** | Is the answer anywhere in the top k? (1 or 0) | hit@1 = 0, hit@3 = 1, hit@5 = 1 |
| **recall@k** | What fraction of the gold spans are covered in the top k? (matters when a question has 2+ gold spans) | recall@5 = 1.0 |
| **reciprocal rank** | How high up was the first correct result? 1/rank | 1/3 = 0.333 |
| **MRR** | The average of reciprocal rank over all questions | (averaged over questions) |
| **nDCG@k** | How good is the whole ordering? Correct results near the top count more, using the weight 1/log2(rank + 1) | DCG = 1/log2(4) = 0.5, ideal DCG = 1, so nDCG = 0.5 |

hit@k and recall@k ignore the order *inside* the top k. MRR and nDCG reward putting the answer higher. You need both kinds.

**Why baselines.** You can't know your harness works until it gives the right answer on cases where you know the answer:
- **oracle:** returns the gold spans themselves, so it must score **1.0** on everything
- **random:** returns random windows of the corpus, so it must score **close to 0**
- **first-N:** always returns the first N windows, ignoring the question: also **close to 0**

If the oracle doesn't get 1.0, your harness or metrics have a bug. If random scores well, your questions or gold spans are too easy or too broad.

**The frozen test set.** Within each category, 10 questions are `dev` (you may look at results and tune on them) and 5 are `test` (a final exam). You are allowed to run the test set **twice in the whole project**: after Phase 6 and at the end. The runner script makes you say so explicitly and keeps a log.

---

## Suggested order
Work on the branch `phase-1-yardstick`. Steps 1-4 first (the foundations, about 2-3 days). Then alternate between Step 5 (writing questions, in batches of 5-8) and Steps 6-8. Commit after each step.

```powershell
cd path\to\cairn
git switch main
git pull
git switch -c phase-1-yardstick
```

### Step 1. Promote your Lesson 2 code into the package
Copy your Lesson 2 solutions into the package, with one rename:
- `src/cairn/jsonl.py`: `write_jsonl`, `read_jsonl`, and the exception **`JsonlError`** (it was `CorpusError` in the lesson, but this module isn't about the corpus)
- `src/cairn/timing.py`: your `Timer`

Tests: `tests/test_jsonl.py`, `tests/test_timing.py`. Also create an empty `src/cairn/evaluation/__init__.py`.

### Step 2. `src/cairn/types.py`: the shared vocabulary
```python
CATEGORIES = ("lexical", "semantic", "multi_chunk", "code")
SPLITS = ("dev", "test")
```
- `Span(path: str, start_line: int, end_line: int)`: **frozen dataclass**. In `__post_init__`, raise `ValueError` unless `1 <= start_line <= end_line` (a single line is fine).
- `Hit(path, start_line, end_line, score: float = 0.0, text: str = "")`: frozen dataclass.
- `Retriever`: a `Protocol` with `search(self, query: str, k: int) -> list[Hit]`.
- `Question(id, category, split, text, gold: tuple[Span, ...])`: frozen dataclass.

Test: `tests/test_types.py`. (`__post_init__` is a method a dataclass calls right after it builds itself, a good place to validate.)

### Step 3. `src/cairn/evaluation/metrics.py`: the heart of the phase
All functions take `hits` (a ranked sequence of `Hit`) and `gold` (a sequence of `Span`):

| Function | Behaviour |
|---|---|
| `overlaps(a, b) -> bool` | same `path` and the inclusive line ranges share a line; works for any two objects with `path`, `start_line`, `end_line` |
| `relevance_flags(hits, gold) -> list[bool]` | one flag per hit, using the **credit rule** |
| `hit_at_k(hits, gold, k) -> float` | `1.0` or `0.0` (a float, so it averages easily); `ValueError` if `k < 1` |
| `recall_at_k(hits, gold, k) -> float` | fraction of gold spans covered by the top k hits; `ValueError` for empty gold or `k < 1` |
| `reciprocal_rank(hits, gold, k=None) -> float` | `1/rank` of the first relevant hit, `0.0` if none; `k=None` means look at all hits |
| `ndcg_at_k(hits, gold, k) -> float` | binary relevance from the flags; DCG = sum of `1/log2(rank + 1)` over relevant hits in the top k; ideal DCG = the same sum for ranks `1..min(len(gold), k)`; result = DCG / ideal |
| `percentile(values, p) -> float` | **nearest-rank**: sort, `rank = ceil(p * n / 100)`, return the item at that rank (1-based). Multiply before dividing. `ValueError` for empty input or `p` outside `(0, 100]` |

Test: `tests/test_metrics.py`. **Work the expected values out by hand first** (the file's comments explain each case), exactly as you did with `chunk_by_words`. If a test disagrees with your code, don't assume the test is wrong, and don't copy the code's output into your own head.

### Step 4. `src/cairn/evaluation/questions.py`: loading and validating the question file
- `class QuestionError(Exception)`
- `load_questions(path, split=None) -> list[Question]`, reading with your `read_jsonl`.
  - JSON keys: `id`, `category`, `split`, `question` (becomes `Question.text`), `gold` (a non-empty list of `{path, start_line, end_line}`)
  - every problem raises `QuestionError` with a message that says **which file and which question id** and names the problem: bad JSON line (wrap `JsonlError`), missing field (name it), unknown category or split (show the value), blank question text, empty gold, impossible line numbers (wrap the `ValueError` from `Span`), duplicate ids
  - gold `path` must be **relative with forward slashes**: reject `docs\a.md`, `/docs/a.md`, `C:/docs/a.md`. (You're on Windows, where backslashes sneak in; the corpus is always addressed with forward slashes.)
  - `split="dev"` or `"test"` returns only those questions; any other `split` argument raises `ValueError`

Test: `tests/test_questions.py`.

### Step 5. Read the corpus and write the 60 questions (the human part)
Do this in batches. Run `tests/test_question_set.py` after each batch: it validates what you have so far.

**The helper:** to see exact lines and check your gold span:
```powershell
C:\mrv\Scripts\python scripts\show_span.py docs/en/docs/tutorial/background-tasks.md 1 40
```
Line numbers match your editor (VS Code shows them on the left).

**What's in the corpus** (`data\raw\fastapi`):
| Where | What | Notes |
|---|---|---|
| `docs/en/docs/tutorial/` | 51 pages: the step-by-step guide | the richest source of questions |
| `docs/en/docs/advanced/` | 35 pages | websockets, middleware, settings, events... |
| `docs/en/docs/reference/`, `how-to/`, `deployment/` | 24 / 12 / 9 pages | |
| `docs/en/docs/*.md` (top level) | `async.md`, `python-types.md`, `features.md`... | |
| `fastapi/` | the library's source: `routing.py` (6.5k lines), `applications.py` (4.8k), `params.py`, `param_functions.py`, `dependencies/utils.py`, `security/oauth2.py`, `openapi/utils.py`... | source for the `code` questions |
| `docs_src/` | 464 Python example files used by the docs | allowed for `code` questions |

**Skip** `release-notes.md` (a 7,500-line changelog full of noise), `translations.md`, `fastapi-people.md`, `newsletter.md`, `_llm-test.md`. A test enforces that gold paths start with `docs/en/docs/`, `fastapi/` or `docs_src/`, and are not `release-notes.md`.

**A reading plan (about 3 hours):** skim `tutorial/first-steps.md`, `path-params.md`, `query-params.md`, `body.md`, `response-model.md`, `handling-errors.md`, `background-tasks.md`, `dependencies/index.md`, a `security/` page, `advanced/websockets.md`; then in the source, open `fastapi/security/oauth2.py`, `fastapi/dependencies/utils.py`, `fastapi/openapi/utils.py` and skim `routing.py`. You are not memorising: you are learning the *shape* so you can find question-worthy passages.

**The 4 categories, 15 questions each:**
| Category | What the question is like | Example *style* (verify against the real files; don't copy) |
|---|---|---|
| `lexical` | contains an exact term that also appears in the answer passage: a parameter, class, function or error message | "What does `response_model_exclude_unset` do?" |
| `semantic` | the same thing asked in **different words**, sharing few key words with the passage | "How do I run some work after the response has been sent?" (answer talks about *background tasks*) |
| `multi_chunk` | needs **two or more separate places**: list each as its own gold span | a docs page *and* the function in `fastapi/` that implements it |
| `code` | asks where something is implemented, or what a function does, in `fastapi/` or `docs_src/` | "Where does FastAPI build the OpenAPI schema?" |

**Rules for good questions**
1. **Write the question yourself, in your own words.** Do not generate the primary set with an AI: generated questions copy the wording of the source and make every method look good.
2. **Gold spans are tight:** at most 60 lines (a test enforces it). Include the lines that actually contain the answer, not a whole page.
3. **One clearly best place.** Avoid questions that many pages could answer equally well; if you find yourself unsure where the gold is, change the question.
4. **For `semantic`:** after writing the question, scan the gold passage. If it shares more than about two of your question's key words, reword the question.
5. **Spread them out:** don't take all 60 from `tutorial/`. Use `advanced/`, `how-to/`, the source, `docs_src/`.
6. **Realistic:** write questions a developer would actually type.
7. **Splits:** within each category, the 3rd, 6th, 9th, 12th and 15th question you write are `"split": "test"`; the other ten are `"dev"`. Decide it when you write the question, and don't change it.
8. Ids: `q001`... `q060`, unique.

Write the questions into `eval/questions.jsonl`, one JSON object per line (the format is above). Always run `show_span.py` on a span before you save it. I'm happy to review a batch of ten for ambiguity before you go on: ask me.

Tests: `tests/test_question_set.py`. The first checks pass as soon as the file exists. `test_question_set_is_complete` stays red until you have exactly 60 (10 dev + 5 test in each category).

### Step 6. `src/cairn/evaluation/baselines.py`
- `line_windows(root, window=40, suffixes=(".md", ".py")) -> list[Hit]`: cut every matching file under `root` into consecutive windows of `window` lines (the last one may be shorter). Paths are relative to `root`, with forward slashes; sorted by `(path, start_line)`. `ValueError` if `window < 1`. **Count lines the way an editor does:** iterate the file (`for line in f`), do not use `str.splitlines()`, which also splits on form-feed and other invisible characters. Empty files give no windows.
- `RandomRetriever(candidates, seed=0)`: `search(query, k)` returns `k` distinct candidates chosen with a seeded `random.Random` (same seed gives the same sequence of answers); all of them if `k` exceeds the pool; `[]` if `k <= 0`.
- `FirstNRetriever(candidates)`: returns the first `k` candidates and ignores the query.
- `OracleRetriever(questions)`: for a known question text, returns the gold spans as `Hit`s with `score=1.0`, in order, cut to `k`; `[]` for an unknown text.

Test: `tests/test_baselines.py`.

### Step 7. `src/cairn/evaluation/harness.py`
- `QuestionResult(question, hits: tuple[Hit, ...], latency_ms: float)`: frozen dataclass
- `run_retriever(retriever, questions, k) -> list[QuestionResult]`: calls `retriever.search(q.text, k)` for each question, times it with your `Timer`, and keeps at most `k` hits (a retriever that returns too many must not get extra credit)
- `summarize(results, ks=(1, 5, 10)) -> dict` returns:
  ```python
  {"overall": {...}, "by_category": {"lexical": {...}, ...}, "misses": ["q003", ...]}
  ```
  each group holds `n`, then for each `k` in `ks`: `hit@k`, `recall@k`, `ndcg@k`; then `mrr` (cutoff = the largest k), `latency_p50_ms`, `latency_p95_ms`. Averages are over questions. Only categories that appear are listed, in the order of `CATEGORIES`. `misses` are the ids with `hit@(largest k) == 0`. `ValueError` for no results.
- `format_report(summary) -> str`: a readable text table: an `overall` row, one row per category, a latency line, and the list of misses. Print numbers with 3 decimals. The test only checks for key words and a few numbers, so the layout is yours.

Test: `tests/test_harness.py`.

### Step 8. `scripts/run_eval.py`
Options (all have defaults):
| Option | Default | Meaning |
|---|---|---|
| `--questions` | `eval/questions.jsonl` | question file |
| `--corpus` | `<data_dir>/raw/fastapi` | folder the `random` and `first` baselines cut into windows |
| `--retriever` | `random` | one of `random`, `first`, `oracle` (more come in later phases) |
| `--split` | `dev` | `dev` or `test` |
| `--k` | `10` | how many results to ask for |
| `--window`, `--seed` | `40`, `0` | for the baselines |
| `--use-test-set` | off | must be passed to run `--split test` |
| `--test-log` | `eval/test_set_runs.log` | each test-set run appends one line: timestamp and retriever name |

Behaviour: asking for the test split without `--use-test-set` is a **usage error** (use `parser.error(...)`, exit code 2, message mentioning `--use-test-set`) and writes nothing to the log. A broken question file prints one line `error: ...` to stderr and exits 1, with no traceback. Otherwise print `format_report(summarize(...))` and exit 0. Same `main()`/`sys.exit(main())` shape as `fetch_corpus.py`.

Test: `tests/test_run_eval_script.py`.

### Step 9. Run the baselines and record them
```powershell
C:\mrv\Scripts\python scripts\run_eval.py --retriever oracle
C:\mrv\Scripts\python scripts\run_eval.py --retriever random --seed 0
C:\mrv\Scripts\python scripts\run_eval.py --retriever first
```
Put the three tables in `docs/results.md` (with the date and the git commit). Check the expectations: **oracle = 1.000 across the board; random and first-N close to 0.** Run random with three different seeds and note how much the numbers move. Then write two sentences on what each result proves about the harness. Do **not** run `--split test`.

### Step 10. Notes and pull request
Answer the checkpoint questions in `docs/notes/phase1.md`, in your own words. Commit, push the branch, open a PR, and merge it (same as Phase 0; tell me and I'll check what is being published).

---

## Checkpoint questions
1. Why is gold defined as file and line spans instead of chunk IDs?
2. What is the difference between hit@k and recall@k? Give an example where they differ.
3. Give two result lists with the same hit@5 but different MRR.
4. What is the credit rule, and what would go wrong without it?
5. What does the oracle baseline prove about your harness? What does the random baseline prove?
6. Why is the test set frozen, and what goes wrong if you tune on it?

## Hint ladder reminder
Stuck? Tell me which step and paste the failing test. I'll nudge first, then give pseudocode, then a short snippet.
