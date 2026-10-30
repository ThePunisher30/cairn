# ADR 0005: Hybrid retrieval with min-max fusion; the reranker and the expansion stay optional

Status: accepted, 2026-10-31. Evidence: `docs/results.md`, section "Phase 6"; `eval/failures/phase6.md`.
The test run of this phase did not support the decision (see "Consequences"); the decision was not changed after it.

## Context

BM25 (Phase 3) and dense retrieval (Phase 4) fail on different questions: on the 40 dev questions each has the
gold passage at rank 1 for 24, and at least one of them does for 31. Phase 6 combines them (fusion), re-scores
a short list with a cross-encoder (reranking), and widens a small retrieved chunk to its section
(small-to-big expansion). I wrote `fusion.py`, `hybrid.py`, `rerank.py` and `context.py`
(`src/cairn/retrieval/`) myself; the only library involved is `sentence-transformers`, for the cross-encoder
model (`cross-encoder/ms-marco-MiniLM-L-6-v2`, on the CPU). No `rank_bm25`, `scikit-learn`, `langchain` or
similar. Tuning used the 40 dev questions only. The 20 test questions were run once, after the configuration
and a prediction were committed.

## Decision

- **The default retriever for the later phases:** `HybridRetriever([bm25, dense])` with `method="weighted"`,
  `normalize="minmax"`, equal weights (0.5 and 0.5), fetch depth 50 from each retriever, final k = 10.
  No reranker, no expansion.
- **How the choice was made:** a rule fixed before the numbers. The cheap option (min-max, no model) is kept
  unless the best reranked row beats it with a paired bootstrap 95% interval of the per-question reciprocal
  rank difference that excludes 0. The best reranked row (min-max + rerank, depth 10) had a difference of
  +0.068 with an interval of -0.019 to +0.162 (better on 8 questions, worse on 3), so min-max stays.
- **Fusion:**
  - Min-max rescales each list's scores to 0 to 1 (all equal gives 1.0), multiplies by the list's weight and
    adds them per chunk; a chunk missing from a list adds 0. Z-score and "none" (the raw sum) are options.
  - RRF adds `1 / (k + rank)` per list, `k` = 60 by default, a repeated key counted once at its best rank.
  - In all of them equal scores go to the chunk met first, reading the lists top to bottom. The same
    `(path, start_line, end_line)` is one chunk; the text is taken from the first retriever that returned it,
    and the returned score is the fused one.
- **Why min-max and not RRF:** MRR 0.766 against 0.732 on dev, better on 7 questions and worse on 1
  (difference +0.034, interval +0.006 to +0.074, sign test p = 0.07, one of seven comparisons on the same
  questions). Z-score (0.750) is 0.016 below min-max, less than one question, and was not tested further.
- **Why not the raw sum:** the scores are not on one scale (BM25 top-1 median 30.4, dense 0.789; spread over the
  top 50 about 15.4 against 0.115). The raw sum is BM25 with dense as a tie-breaker (MRR 0.672 against 0.660
  for BM25; hit@10 exactly BM25's 0.800).
- **The reranker is available, not the default.** If it is used: rerank depth 10 (not 30), batch size 4 and
  `max_chars` 2000. Depth 10 was as good as or better than 20, 30 and 50 on every dev column at a quarter to
  half the time; batch 4 gives the same scores as 16 and was faster in all four comparisons (13 to 25%);
  500 characters lost four questions at hit@1, and 1000 lost one (inside the noise), so 2000 stays. Scores are
  cached under `(model name, query, passage)`.
- **Small-to-big expansion is available, off by default.** `ContextExpander(chunks, max_tokens=800,
  mode="parent")` is the mode that did anything on dev. A chunk is matched to its place, grown through the
  chunks of the same file that share its parent heading, previous chunk first, within the token budget; the
  hit chunk is always kept; `expand` drops a hit that lies inside an earlier expanded hit.

## Consequences

- **Dev numbers (k = 10) of the choices:**

| row | hit@1 | hit@5 | hit@10 | MRR | p50 |
|---|---|---|---|---|---|
| dense | 0.600 | 0.825 | 0.925 | 0.711 | 20 to 44 ms |
| hybrid min-max (chosen) | 0.700 | 0.850 | 0.925 | 0.766 | 23 to 50 ms |
| hybrid rrf + rerank, depth 10 | 0.750 | 0.900 | 0.900 | 0.806 | 0.9 to 1.2 s |
| hybrid min-max + rerank, depth 10 | 0.775 | 0.900 | 0.925 | 0.834 | about 1.1 s |

- **The test run did not support the choice.** The chosen row has hit@1 0.350 (predicted 0.65, range 0.50 to
  0.80) and MRR 0.533 (predicted 0.70, range 0.58 to 0.80); hit@5 0.800 and hit@10 0.900 were as predicted.
  Dense alone has hit@1 0.550 and MRR 0.659, and was the best row without a reranker. MRR gaps from dev to test:
  dense -0.052, min-max -0.233, RRF -0.237, BM25 -0.208. On dev the advantage of fusion over dense was not
  statistically established (better on 9 questions, worse on 9, difference +0.056, interval -0.046 to +0.160),
  and on test it did not reproduce. With 20 test questions, 0.20 is four questions.
- **The decision stays, for a reason of process:** choosing dense after seeing the test split would make the
  split a second dev set. The second and last test run (Phase 11) measures the whole system.
- **Speed:** BM25 takes 5 ms, embedding the question 42 ms, the flat search 0.5 ms and the fusion 0.3 ms. The
  fused retriever costs about as much as running both retrievers. The reranker costs 0.7 to 1.2 s at depth 10 and
  3 to 4 s at depth 30 (2.6 to 3.2 s on test), 94% or more of the total time. Nothing measured got under 300 ms.
- **What a reranker did on dev:** it moves the top rank (hybrid RRF hit@1 26 to 29 of 40) and semantic questions
  (BM25 MRR 0.432 to 0.800), it did not raise hit@10 (dense 37 to 35 questions), and it lowered code questions
  in both splits (code MRR 0.548 on dev against 0.550 to 0.625 without; 0.200 to 0.267 on test against 0.319 to
  0.433). It took rank 1 away from four questions RRF had right. The cause for code is not established.
- **A reranker only reorders its pool.** The ceiling of a pool of 30 is hit@30 = 0.975 for RRF; the three golds
  RRF finds at ranks 11 to 30 end at rank 16 after reranking, outside the top 10.
- **Expansion bought little:** `parent` at 800 tokens covers 0.832 of the gold lines against 0.802 for the plain
  chunks, for 2.1 times the tokens (1,351 to 2,889); `section` mode changed nothing. Expanded hit@5 is not
  comparable with the chunk table, so the table reports coverage and tokens.
- **Labels:** five dev labels (q004, q005, q007, q043, q044) have a second passage that also answers the question
  and were kept; dev hit@1 understates the rows that rank the other passage first, by up to 0.125.
- **Failures at rank 1 for hybrid RRF + rerank (11 on dev):** a second answering passage 3, wrong chunk on the
  right page 3, prose outranks code 2, sibling code 1, vocabulary mismatch 1, wrong chunk on another page 1.
  In 10 of the 11 the gold is inside the reranker's pool.
- **Not found out:** why the fused rows keep their top-5 recall on test and lose rank 1 (the test questions were
  not read, on purpose); why the reranker prefers a related passage over the gold in four questions; whether the
  small function-sized chunks of q044 (15 to 53 tokens) are why it is never found; why dense is the more stable
  row (it has fewer settings chosen on dev, but BM25 also dropped by 0.208).
- **Not done:** rerank depth 10 and min-max + rerank were never run on the test split; a smaller cross-encoder,
  reranking only when the first stage is unsure, and choosing the fusion per question type were not tried.
  Latency is from one laptop with other programs running, and a repeat of the same setting varied by up to 50%.

## Alternatives considered

- **Dense alone:** best on test (MRR 0.659). Not chosen on dev (0.711 against 0.766, a difference not shown to be
  real). Choosing it now would use the test split to decide.
- **BM25 alone:** the worst MRR on both splits (0.660 dev, 0.452 test), but 1.000 on lexical dev questions.
- **RRF:** k = 60 gives 0.732; k = 1 gives 0.761 and every constant is within 0.05 of the best. It hurts semantic
  questions (0.667 against 0.724 for dense).
- **Z-score fusion:** 0.750 on dev, 0.016 below min-max (less than one question).
- **A different weight:** 0.5 and 0.75 are within 0.05 of the best; 0.25 and 1.0 are not. 0.5 was kept as the round
  default instead of searching for a finer value on 40 questions.
- **The reranker as the default:** best on every dev column, but the rule above was not met, it costs about 20 to
  45 times the latency (for min-max + rerank at depth 10: three more questions at hit@1 and two more at hit@5),
  and it lowers code questions.
- **A deeper rerank pool (20, 30, 50):** the ceiling rises from 0.900 to 0.975 and hit@10 stays at 0.900.
- **Section-mode expansion, or larger budgets:** no gain, or 3 times the tokens for 0.039 more coverage.
- **Libraries (`rank_bm25`, `langchain`, `llama-index`, vector databases):** ruled out by the project constraints
  until Phase 11, where the same questions can be used to compare against them.
