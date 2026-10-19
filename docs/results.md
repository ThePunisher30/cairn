# Results

Lab notebook. Every number in the README must trace back to a row here.

## Corpus (FastAPI 0.143.0)

| Part | Location | Files |
|---|---|---|
| Markdown docs | docs/en/docs | 156 |
| Python source | fastapi/ | 52 |

## Baselines on the dev questions

40 dev questions, k = 10. Date: 2026-10-13. Commit: 2e69929.

Candidates are 40-line windows of **every** `.md` and `.py` file under `data/raw/fastapi`: 10,608 windows. That is a bigger and noisier pool than the corpus I actually mean (`docs/en/docs`, `fastapi/` and `docs_src/`): it also contains the translated docs (for example `docs/de/...`) and the repository's own `tests/` folder. Narrowing the corpus is a decision for Phase 2.

| Retriever | hit@1 | hit@5 | hit@10 | recall@1 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|
| oracle | 1.000 | 1.000 | 1.000 | 0.875 | 1.000 | 1.000 | 1.000 |
| first-N | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| random, seed 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| random, seed 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| random, seed 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Oracle, `multi_chunk` questions only: recall@1 = 0.500 (all other categories: 1.000).

Commands: `python scripts/run_eval.py --retriever oracle` (and `first`, `random --seed N`).
The test split was not used.

### What this shows

1. If the oracle returned anything other than 1.000, I would suspect a bug in the loader, the metrics or the harness. It returns exactly the gold spans, so it must score perfectly. It does, so the three agree with each other.
2. Random and first-N both score about 0, so the questions cannot be answered by luck or by position. Random picks 10 windows out of 10,608, which is roughly a 0.1-0.3% chance per question, so guessing does not work. First-N returns the same windows for every question (the alphabetically first files, a German translation page `docs/de/docs/_llm-test.md`), so it measures what you get from ignoring the question completely: nothing.
3. The oracle's recall@1 is 0.875, not 1.0, on purpose. The 10 `multi_chunk` questions each have **two** gold spans, and at k = 1 only one result fits, so recall@1 is 1/2 = 0.500 for them. The other 30 questions have one gold span, so they score 1.0: (30 x 1.0 + 10 x 0.5) / 40 = 0.875.


## Phase 2: chunking strategies on the dev questions

40 dev questions. Date: 2026-10-20. Commit: 7598ace. Command: `python scripts/compare_chunkers.py`.

Corpus: 671 documents (155 prose, 516 code): `docs/en/docs/**/*.md` without `release-notes.md`, plus `fastapi/**/*.py` and `docs_src/**/*.py`. Translations and the repository's own `tests/` are excluded.

"intact" is the share of gold spans that fit completely inside one chunk. It is an upper bound on retrieval quality: a retriever cannot return a whole answer that no single chunk contains.

| Strategy | Chunks | Mean tokens | >512 | Intact | lexical | semantic | multi_chunk | code |
|---|---|---|---|---|---|---|---|---|
| fixed-100 | 5112 | 85.3 | 1 | 0.120 | 0.100 | 0.100 | 0.200 | 0.000 |
| fixed-200 | 2576 | 169.2 | 1 | 0.460 | 0.600 | 0.400 | 0.550 | 0.200 |
| fixed-400 | 1477 | 295.1 | 1 | 0.780 | 0.900 | 0.500 | 0.900 | 0.700 |
| recursive-200 | 2767 | 157.5 | 1 | 0.620 | 0.700 | 0.400 | 0.700 | 0.600 |
| recursive-400 | 1561 | 279.2 | 1 | 0.800 | 0.800 | 0.500 | 0.950 | 0.800 |
| recursive-200 + overlap 3 | 2767 | 185.6 | 1 | 0.680 | 0.800 | 0.400 | 0.750 | 0.700 |
| structure-150 | 3962 | 110.0 | 76 | 0.660 | 0.800 | 0.300 | 0.750 | 0.700 |
| **structure-300** | 3267 | 133.4 | 76 | 0.880 | 0.900 | 0.900 | 0.950 | 0.700 |
| structure-600 | 3087 | 141.2 | 105 | 0.920 | 0.900 | 1.000 | 1.000 | 0.700 |

"structure-N" is the heading-aware Markdown chunker (budget N tokens) for docs plus the AST Python chunker for code. For code, N only sets the fallback used when a file does not parse, which is why the code column does not move.

The test split was not used.

### Ingest statistics for structure-300

Command: `python scripts/ingest.py`. 671 documents, 3,267 chunks (1,392 prose, 1,875 code), mean 133.4 tokens, max 7,124, 76 chunks over 512 tokens.

| Tokens | <=50 | 51-100 | 101-200 | 201-300 | 301-400 | 401-512 | >512 |
|---|---|---|---|---|---|---|---|
| Chunks | 1226 | 770 | 685 | 468 | 31 | 11 | 76 |

### What this shows

1. **Structure-aware chunking wins at a smaller size.** structure-300 keeps 0.88 of the gold spans intact with chunks of 133 tokens on average. fixed-400 keeps 0.78 with chunks of 295 tokens, more than twice as large. Cutting at headings and function boundaries keeps answers together without needing big chunks. The semantic questions show it best: 0.40-0.50 for every size-based strategy, 0.90 for structure-300.
2. **Bigger is better for this metric, but the metric cannot see the cost.** Fixed chunking goes from 0.12 to 0.46 to 0.78 as the size doubles. A chunk containing the whole file would score 1.0 and be useless for search, so the table also lists chunk size.
3. **The biggest chunk is `FastAPI.__init__`**: `fastapi/applications.py` lines 64-1052, 7,124 tokens. The AST chunker makes one function one chunk, and this signature has about 60 parameters, each documented inline with a long `Doc("...")` string. The next largest are 2,800-3,400 tokens: `get_request_handler` in `routing.py`, then `APIRouter.put/post/head/patch/trace` and `FastAPI.put`, all with the same huge parameter documentation.
4. **All 76 chunks over 512 tokens are code; none is prose.** They cluster in `routing.py` (21), `applications.py` (13), `param_functions.py` (9), `params.py` (8) and `dependencies/utils.py` (6). Prose never goes over because the Markdown chunker splits an oversized section by paragraph; the Python chunker never splits a function. An embedding model with a 512-token limit would truncate these chunks.
5. **Six dev spans are cut by structure-300** (`--show-broken structure-300`), for three different reasons:

| Question | Gold span | Why it is cut |
|---|---|---|
| q034 (code) | `fastapi/datastructures.py` 153-168, a whole class | split into a header chunk plus one chunk per method |
| q043 (code) | `docs_src/handling_errors/tutorial004_py310.py` 1-26, a whole example file | split into imports plus one chunk per function |
| q044 (code) | `docs_src/dependencies/tutorial005_py310.py` 1-20, a whole example file | same |
| q023 (semantic) | `dependencies-with-yield.md` 187-200 | the section is about 333 tokens, over the 300 budget, so it was split by paragraph into 187-195 and 197-200 |
| q049 (multi_chunk) | `dependencies-with-yield.md` 268-288 | its section is also over budget and is split at line 283, inside the gold span |
| q005 (lexical) | `http-basic-auth.md` 95-102 | the chunk is 95-101: line 102 of the gold span is a blank line, and every chunker trims blank edge lines |

   The code failures are **structural**: the questions ask for a whole class or a whole example file, and the AST chunker splits at exactly those boundaries. No size setting changes that, which is why code stays at 0.70 for structure-150, -300 and -600. q023 and q049 are budget cuts: structure-600 fixes them (semantic 1.000). q005 is an **annotation flaw**, not a chunker flaw: the gold span ends on a blank line. It stays broken at every size (lexical stays at 0.900 for structure-300 and -600). I have not edited the gold span, because changing the answer key after seeing a result has to be a deliberate, documented change.
6. **Choice for the rest of the project: structure-300.** It has the best balance: 0.88 intact, small chunks, and fewer oversized chunks than structure-600 (76 against 105). structure-600 gains only 0.04, entirely from the two over-budget prose sections, and puts 29 more chunks past the 512-token limit.

Open problems for later: 76 code chunks over 512 tokens, and whole-class or whole-file answers that the AST chunker splits (code intact 0.70).

Commands: `python scripts/ingest.py`; `python scripts/compare_chunkers.py`; `python scripts/compare_chunkers.py --show-broken structure-300`.
