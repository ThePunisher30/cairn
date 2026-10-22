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


## Phase 3: BM25 baseline

40 dev questions, k = 10. Date: 2026-10-21. Commit: ca9c4f4.

Setup: `structure-300` chunks (the Phase 2 choice, 3,267 chunks), the code-aware tokenizer (identifiers kept whole and also split into their parts, no stopwords), `k1 = 1.2`, `b = 0.75`. The index has 7,184 distinct terms, a mean chunk length of 86.9 tokens, and is a 1.1 MB pickle file (`chunks.jsonl` is 2.7 MB).

| Retriever | hit@1 | hit@5 | hit@10 | recall@1 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|
| oracle (Phase 1) | 1.000 | 1.000 | 1.000 | 0.875 | 1.000 | 1.000 | 1.000 |
| random / first-N (Phase 1) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **BM25** | 0.600 | 0.750 | 0.800 | 0.525 | 0.750 | 0.660 | 0.658 |

By question category:

| Category | n | hit@1 | hit@10 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| lexical | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| semantic | 10 | 0.300 | 0.800 | 0.800 | 0.432 | 0.518 |
| multi_chunk | 10 | 0.600 | 0.800 | 0.600 | 0.658 | 0.551 |
| code | 10 | 0.500 | 0.600 | 0.600 | 0.550 | 0.563 |

Search time over 800 searches (the 40 dev questions, 20 times each): p50 2.62 ms, p95 4.13 ms, max 8.39 ms. The target was p95 under 20 ms.

Commands: `python scripts/ingest.py`, `python scripts/build_index.py`, `python scripts/run_eval.py --retriever bm25`, `python scripts/bench_search.py`. The test split was not used.

### What this shows

1. **BM25 is perfect on exact-word questions and weakest on meaning.** All 10 lexical questions have the answer at rank 1 (MRR 1.000). The semantic questions need rank 10 to get 0.800 but only 0.300 at rank 1. A keyword engine finds the answer when the question uses the passage's own words, and struggles when the question paraphrases it.
2. **It is far above the floor and well below the ceiling.** Random and first-N score 0.000, the oracle 1.000. BM25 gets MRR 0.660, so it already works, and 8 of 40 questions are still missed at k = 10.
3. **It is fast.** p95 of 4.1 ms is about five times under the 20 ms target, from a plain dictionary of lists and a heap, with no model and no special hardware.

### The 8 misses (no gold passage in the top 10)

For each, where the first chunk that overlaps the gold span ranked, out of all chunks that share at least one word with the question.

| Question | Category | Gold rank | Why BM25 missed it |
|---|---|---|---|
| q017 spellings accepted as "true" | semantic | 212 | **Vocabulary mismatch.** The passage contains `true` but none of the content words of the question (`spellings`, `accepted`, `boolean`, `option`, `url`). |
| q026 turn off the interactive docs in production | semantic | 1217 | **Vocabulary mismatch.** The gold chunk shares only `can` and `the` with the question: it talks about disabling the OpenAPI URLs, the question says "turn off the interactive documentation pages". |
| q031 status codes with no body | code | 188 | **Docs outrank code, and there is no stemming.** The tutorial page about status codes is dense in the question's words and scores 36; the function `is_body_allowed_for_status_code` has `code` but the question says `codes`. |
| q041 yield dependency into an async context manager | code | 136 | **Docs outrank code.** "How does the code ..." is a question about the implementation, but the docs pages on `yield` contain the same words more often. BM25 cannot tell that the user wants source code. |
| q043 what the HTTPException handler returns in the example | code | 16 | **The answer is split across chunks.** The gold span is a whole example file, which `structure-300` cuts into 4 chunks (41, 46, 86 and 73 tokens); the docs page about error handling matches the words better than any one small chunk. This is the Phase 2 finding again. |
| q044 one dependency uses another: what is returned with no query | code | 972 | **Vocabulary mismatch plus a split file.** The 4 chunks of the file share only `query` with the question (the code says `query_extractor`, `read_query`). |
| q058 why filter data through the response model; what if validation fails | multi_chunk | 20 | **Many chunks compete.** The question is long and generic, and several chunks of the response-model page share its words. The two specific gold chunks rank 20th and lower, and each lacks some of the question's content words (`serialization`, `fails`, `happens`). |
| q059 declare routes in a separate module and attach them to the main app | multi_chunk | 47 | **Vocabulary mismatch.** The passages talk about `APIRouter` and `include_router`; the question says `attach`, `separate` and `module`. |

Five kinds of failure: vocabulary mismatch (q017, q026, q044, q059), docs ranked above the code the user asked about (q031, q041), no stemming (`codes` against `code`, q031), answers cut across chunks (q043, q044), and a long generic question with many competing chunks (q058). Embeddings (Phase 4) should help with the first, hybrid retrieval and reranking (Phase 6) with the second, a stemmer is a cheap tokenizer experiment, and parent-child retrieval is the idea for the fourth.


### Experiments: tokenizer variants x chunkers

Dev questions only, k = 10, `k1 = 1.2`, `b = 0.75`. Date: 2026-10-22. Command: `python scripts/ablate_bm25.py --by-category`. `plain` = identifiers kept whole only; `split` = whole and parts; `+stop` = the English stopword list removed.

| Chunker | Tokenizer | Terms | hit@1 | hit@10 | MRR | nDCG@10 | MRR lexical | MRR semantic | MRR multi_chunk | MRR code |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed-200 | plain | 6606 | 0.375 | 0.675 | 0.459 | 0.464 | 0.683 | 0.355 | 0.472 | 0.325 |
| fixed-200 | split | 7184 | 0.425 | 0.725 | 0.514 | 0.512 | 0.750 | 0.355 | 0.625 | 0.325 |
| fixed-200 | plain+stop | 6573 | 0.400 | 0.700 | 0.478 | 0.505 | 0.650 | 0.454 | 0.464 | 0.344 |
| fixed-200 | split+stop | 7151 | 0.450 | 0.700 | 0.523 | 0.523 | 0.614 | 0.433 | 0.668 | 0.377 |
| fixed-400 | plain | 6606 | 0.400 | 0.800 | 0.502 | 0.529 | 0.633 | 0.437 | 0.674 | 0.267 |
| fixed-400 | split | 7184 | 0.525 | 0.850 | 0.597 | 0.611 | 0.850 | 0.415 | 0.702 | 0.421 |
| fixed-400 | plain+stop | 6573 | 0.450 | 0.725 | 0.524 | 0.544 | 0.675 | 0.428 | 0.661 | 0.333 |
| fixed-400 | split+stop | 7151 | 0.500 | 0.775 | 0.564 | 0.581 | 0.760 | 0.407 | 0.662 | 0.429 |
| recursive-400 | plain | 6606 | 0.375 | 0.725 | 0.467 | 0.492 | 0.639 | 0.382 | 0.632 | 0.217 |
| recursive-400 | split | 7184 | 0.475 | 0.750 | 0.562 | 0.567 | 0.850 | 0.370 | 0.700 | 0.327 |
| recursive-400 | plain+stop | 6573 | 0.425 | 0.725 | 0.495 | 0.506 | 0.675 | 0.418 | 0.648 | 0.239 |
| recursive-400 | split+stop | 7151 | 0.475 | 0.750 | 0.544 | 0.559 | 0.750 | 0.398 | 0.681 | 0.347 |
| structure-300 | plain | 6606 | 0.450 | 0.750 | 0.552 | 0.564 | 0.820 | 0.439 | 0.614 | 0.334 |
| **structure-300** | **split** | 7184 | **0.600** | 0.800 | **0.660** | 0.658 | **1.000** | 0.432 | 0.658 | 0.550 |
| structure-300 | plain+stop | 6573 | 0.450 | 0.725 | 0.549 | 0.561 | 0.750 | 0.433 | 0.667 | 0.348 |
| structure-300 | split+stop | 7151 | 0.575 | 0.825 | 0.654 | 0.663 | 0.833 | 0.448 | 0.725 | 0.611 |

With 40 questions, moving one question from rank 1 to not found changes hit@1 by 0.025, so a difference of one or two questions is not evidence.

### Experiments: k1 x b

Command: `python scripts/ablate_bm25.py --tune structure-300 --tokenizer split`. MRR on the dev questions.

| | b=0 | b=0.25 | b=0.5 | b=0.75 | b=1 |
|---|---|---|---|---|---|
| k1=0 | 0.314 | 0.314 | 0.314 | 0.314 | 0.314 |
| k1=0.6 | 0.331 | 0.577 | 0.628 | 0.633 | 0.638 |
| k1=0.9 | 0.329 | 0.604 | 0.644 | 0.654 | 0.637 |
| k1=1.2 | 0.320 | 0.603 | 0.649 | **0.660** | 0.619 |
| k1=1.5 | 0.280 | 0.592 | 0.642 | 0.651 | 0.558 |
| k1=2 | 0.280 | 0.590 | 0.633 | 0.633 | 0.518 |
| k1=3 | 0.242 | 0.586 | 0.611 | 0.584 | 0.465 |

### What the experiments show

1. **Splitting identifiers helps, and the direction is the same for all four chunkers.** MRR goes from 0.459 to 0.514 (fixed-200), 0.502 to 0.597 (fixed-400), 0.467 to 0.562 (recursive-400) and 0.552 to 0.660 (structure-300). For structure-300, hit@1 rises from 0.450 to 0.600, which is 6 questions. Per category (structure-300), lexical goes from 0.820 to 1.000 and code from 0.334 to 0.550, while semantic is flat (0.439 to 0.432). Comparing the two rankings question by question: splitting is better on 15 questions, worse on 7 and the same on 18 (counted with a throwaway script, not committed). Most of the 7 are small moves (for example rank 2 to 3, or 4 to 7). It helps because the passages and code contain identifiers and the questions use plain words: for q032, "How is the default unique operation ID of a route assembled?", the gold chunk contains the identifiers `generate_unique_id` and `operation_id`. With the plain tokenizer only `route` matches (gold at rank 226); with splitting `id`, `operation`, `route` and `unique` all match (rank 2). It does not help semantic questions, which fail for a different reason: the passage uses other words.
2. **Stopwords are noise here.** For structure-300 + split they change MRR from 0.660 to 0.654 and hit@10 from 0.800 to 0.825 (one question); lexical drops from 1.000 to 0.833 while code rises from 0.550 to 0.611. For the other chunkers the effect has no consistent sign: +0.019 and +0.022 and +0.028 without splitting, but 0.597 to 0.564 (fixed-400) and 0.562 to 0.544 (recursive-400) with it. Differences of 0.01 to 0.04 are one or two questions. BM25's idf already gives common words almost no weight, so removing them changes little. No stopwords is the simpler choice and is not worse.
3. **The chunker matters, and the ranking agrees with Phase 2 on the low end.** With splitting, structure-300 has the best MRR (0.660), then fixed-400 (0.597), recursive-400 (0.562) and fixed-200 (0.514), the same worst chunker as in the Phase 2 integrity table (0.46). The orders do not match exactly: fixed-400 has the best hit@10 (0.850 against 0.800) although only 0.78 of its answers are intact. Bigger chunks contain more of the question's words, so they are found within 10 results more often but rank lower, and hit@10 gives credit for any overlap with the answer, even when the answer is cut across chunks.
4. **The defaults are about the best.** The best cell of the grid is `k1 = 1.2, b = 0.75` (0.660). Nearby cells are within 0.02 (`k1 = 0.9`: 0.654, `k1 = 1.5`: 0.651), which is noise on 40 questions, so the defaults stay. Two cells are very informative: `b = 0` gives about 0.32 for every `k1`, half the score, because without length normalisation the chunks that are very long (the 76 code chunks over 512 tokens, up to 7,124) collect matches from many words and outrank the focused chunks. `k1 = 0`, where only the presence of a word counts and not how often it occurs, gives 0.314 for every `b`: term frequency is worth about 0.35 of MRR. `b = 1` hurts at large `k1` (0.465 at `k1 = 3`) because it penalises long chunks too much, and some correct chunks are long.
5. **Keyword baseline for Phases 4-6: `structure-300` chunks, the `split` tokenizer, no stopwords, `k1 = 1.2`, `b = 0.75`.** MRR 0.660, hit@10 0.800, p95 search time 4 ms.

## Phase 4: dense retrieval

Same 3,267 `structure-300` chunks and the same 40 dev questions as Phase 3. Each chunk (its heading plus text) is embedded once with `bge-small` (384 dimensions) and a question is matched to chunks by dot product of unit vectors (cosine). The index is a flat matrix searched by brute force.

| Retriever | hit@1 | hit@5 | hit@10 | recall@10 | MRR | nDCG@10 | p50 search |
|---|---|---|---|---|---|---|---|
| BM25 (Phase 3) | 0.600 | 0.750 | 0.800 | 0.750 | 0.660 | 0.658 | 2.6 ms |
| **Dense, bge-small** | 0.600 | 0.825 | 0.925 | 0.863 | 0.711 | 0.723 | 111.7 ms |
| Dense, bge-small, no query prefix | 0.450 | 0.800 | 0.825 | 0.762 | 0.605 | 0.634 | 134.8 ms |
| Dense, minilm | 0.350 | n/a | 0.800 | n/a | 0.492 | 0.529 | n/a |

By question category (MRR, with hit@10 in brackets):

| Category | n | BM25 | Dense bge-small |
|---|---|---|---|
| lexical | 10 | 1.000 (1.000) | 0.817 (1.000) |
| semantic | 10 | 0.432 (0.800) | 0.724 (0.900) |
| multi_chunk | 10 | 0.658 (0.800) | 0.700 (0.900) |
| code | 10 | 0.550 (0.600) | 0.602 (0.900) |

Commands: `python scripts/embed_chunks.py`, `python scripts/run_eval.py --retriever dense` (also with `--no-query-prefix`), `python scripts/ablate_dense.py --by-category`, `python scripts/bench_embed.py --model bge-small`. The test split was not used. One question is 0.025 of a hit rate, so differences below about 0.05 are noise.

### What this shows

1. **Dense finds the answer more often than BM25.** hit@10 goes from 0.800 to 0.925 and recall@10 from 0.750 to 0.863: 3 questions are missed instead of 8. The largest gain is on `code` (hit@10 0.600 to 0.900).
2. **Dense wins on meaning and loses on exact words.** Semantic MRR goes from 0.432 to 0.724 (hit@1 from 0.300 to 0.700). Lexical MRR falls from 1.000 to 0.817: every lexical answer is still in the top 10, but only 7 of the 10 are at rank 1. The other three are at rank 2 or 3 (q004, q005, q007), and for two of them (q005, q007) the top result is another chunk of the right page.
3. **hit@1 is the same (0.600).** The gain is in recall within the top 10, not in putting the answer first, so "dense ranks better" is not supported by this data, "dense recalls more" is.
4. **The two retrievers fail on different questions.** At rank 1: both are right on 17, only BM25 on 7, only dense on 7, neither on 9. At least one is right on 31 of 40 (0.775), against 0.600 for each alone. This is the case for hybrid retrieval in Phase 6.
5. **The query prefix is worth a lot, on this setup.** Without bge's "Represent this sentence for searching relevant passages: " prefix, MRR falls from 0.711 to 0.605, hit@10 from 0.925 to 0.825, and 4 more questions are missed (q016, q026, q031, q034). Every category is lower. It is not a general law: with the three other chunkers the prefix changes MRR by at most 0.012 in either direction (see below), so the gain is large only for `structure-300`. I did not find out why.
6. **The model matters.** `minilm` (384 dimensions too, no prefix) reaches MRR 0.492 against 0.711 for `bge-small`, and its lexical MRR is only 0.526. The bigger gain comes from the model, not from the chunker or the settings.
7. **It costs a lot more than BM25.** A search takes 112 ms (p50) against 2.6 ms, about 40 times. The brute-force scan over the 3,267 vectors takes 0.9 ms; embedding a question alone takes 38.5 ms (30 repeated runs of one question, measured separately), so the model is almost all of it. The eval's p50 of 112 ms is higher than the 38.5 ms and I did not find out why. Building the index took 471 s for the 2,802 distinct chunk texts (465 chunks are exact duplicates of another chunk, mostly repeated tutorial code, and were taken from the cache). A second run takes 0.0 s. The index itself is small: 3,267 x 384 x 4 bytes = 5.0 MB, which scales to 154 MB per 100,000 chunks.
8. **83 chunks are over the model's 512-token limit** and are truncated: the model never sees their tail. (BM25 had 76 over the same limit with a different tokenizer, and the two counts are not comparable.)

### Experiments: chunkers (bge-small)

| Chunker | Chunks | Prefix | hit@1 | hit@10 | MRR | nDCG@10 | MRR lexical | MRR semantic | MRR multi_chunk | MRR code |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed-200 | 2,576 | on | 0.325 | 0.675 | 0.431 | 0.453 | 0.408 | 0.472 | 0.544 | 0.298 |
| fixed-200 | 2,576 | off | 0.325 | 0.700 | 0.443 | 0.470 | 0.408 | 0.485 | 0.517 | 0.363 |
| fixed-400 | 1,477 | on | 0.250 | 0.750 | 0.448 | 0.496 | 0.400 | 0.550 | 0.562 | 0.281 |
| fixed-400 | 1,477 | off | 0.250 | 0.775 | 0.440 | 0.494 | 0.375 | 0.536 | 0.552 | 0.298 |
| recursive-400 | 1,561 | on | 0.300 | 0.750 | 0.457 | 0.501 | 0.408 | 0.544 | 0.583 | 0.292 |
| recursive-400 | 1,561 | off | 0.300 | 0.750 | 0.450 | 0.492 | 0.408 | 0.509 | 0.588 | 0.295 |
| **structure-300** | 3,267 | on | **0.600** | **0.925** | **0.711** | **0.723** | 0.817 | 0.724 | 0.700 | 0.602 |
| structure-300 | 3,267 | off | 0.450 | 0.825 | 0.605 | 0.634 | 0.767 | 0.600 | 0.533 | 0.520 |

1. **The chunker matters more than anything else tried.** `structure-300` has MRR 0.711 against 0.431 to 0.457 for the three size-based chunkers, and hit@1 0.600 against 0.250 to 0.325. It does this with the most chunks (3,267), so there are more places to go wrong, not fewer. The order of the chunkers agrees with BM25 in Phase 3 (structure-300 first) and with the integrity table of Phase 2.
2. **Dense hardly helps the size-based chunkers' ranking.** For `fixed-400` and `recursive-400` the dense MRR (0.448, 0.457) is below their BM25 MRR with the split tokenizer (0.597, 0.562). Only for `structure-300` is dense above BM25 (0.711 against 0.660). I did not investigate why.
3. **Do not read the prefix on/off differences of the size-based chunkers.** They are at most 0.012 MRR and change sign from row to row, which is one or two questions.

Dense retrieval baseline for Phases 5-6: `structure-300` chunks, `bge-small`, trained query prefix, flat index. MRR 0.711, hit@10 0.925, p50 search time 112 ms.

### Embedding speed

Sorting the chunks by length before batching makes the model faster (padding is wasted work): on a sample of 160 chunks, 5.4 chunks/s in the original order (batch 32) against 8.9 chunks/s sorted (batch 16), 1.65 times faster, and batch 16 was better than 8 or 32. The real run (5.9 chunks/s) was slower than the benchmark predicted (about 366 s for the corpus, 471 s measured); I did not find out why.
