# ADR 0002: BM25 keyword retrieval and its settings

Status: accepted, 2026-10-22. Evidence: `docs/results.md`, sections "Phase 3".

## Context

Phases 4-6 need a keyword baseline to compare embeddings against, and a keyword half for hybrid
retrieval. The corpus is the FastAPI docs and source code, so questions mix plain English with exact
identifiers (`HTTPException`, `generate_unique_id`). I built BM25 from the formula (no search library)
and compared tokenizer variants, four chunkers and a grid of `k1` and `b` on the 40 dev questions.

## Decision

- **Retriever:** BM25 with `idf = ln(1 + (N - df + 0.5) / (df + 0.5))`, `k1 = 1.2`, `b = 0.75`. Each
  distinct query word counts once. Top 10 with `heapq.nlargest`; equal scores go to the lower chunk number.
- **Index:** an inverted index (word -> list of `(chunk number, term frequency)`) over each chunk's heading
  path plus text, with chunk lengths. It stores its own tokenizer and the chunk IDs it was built from, and
  the retriever refuses chunks that do not match.
- **Tokenizer:** lowercase words; an identifier such as `get_request_handler` or `BackgroundTasks` is kept
  whole and also split into its parts. No stopwords.
- **Chunks:** the Phase 2 choice, `structure-300`.
- **File format:** a pickle of a plain dict with `"version": 1` (no classes of ours inside).
- **Measured on dev:** MRR 0.660, hit@10 0.800 (lexical questions 1.000, semantic 0.432, code 0.550);
  p95 search time 4.1 ms; 3,267 chunks, 7,184 terms, a 1.1 MB file built in under a second.

## Consequences

- 8 of 40 dev questions are missed at k = 10. Causes: vocabulary mismatch (the question paraphrases the
  passage), docs outranking the code the question is about, no stemming (`codes` against `code`), and
  answers cut across chunks. Embeddings (Phase 4) and hybrid retrieval with reranking (Phase 6) are
  the planned fixes; a stemmer is a cheap experiment before them.
- Length normalisation matters on this corpus: `b = 0` halves MRR (0.660 to 0.320) because the very long
  code chunks from Phase 2 collect matches from many words.
- The settings were chosen on 40 dev questions, where one question is worth 0.025. Differences below about
  0.04 are noise, so the defaults stay. The test split has not been used.
- Pickle can run code when loaded: index files must only come from me. The whole index lives in memory.
- An index must be rebuilt whenever `chunks.jsonl` changes; a mismatch is an error, not a wrong answer.
- A compact on-disk format (delta-encoded postings) is not built; revisit if the index gets large.

## Alternatives considered

- **No identifier splitting:** MRR 0.552 against 0.660 (15 questions better with splitting, 7 worse).
- **Stopword removal:** no consistent gain (MRR 0.654 against 0.660, hit@10 one question better); BM25's
  idf already gives common words little weight, and it is one more thing to maintain.
- **Other chunkers:** `fixed-400` has a higher hit@10 (0.850) but a lower MRR (0.597); `fixed-200` and
  `recursive-400` are worse on both.
- **Other `k1` and `b`:** the best grid cell is the default; cells nearby are within noise.
- **Textbook idf `ln((N - df + 0.5) / (df + 0.5))`:** goes negative for a word in more than half of the
  chunks, so containing a common word would lower a score. Not used.
- **Sorting all scores instead of a heap:** gives the same result; the saving is small at this size and
  grows with the corpus.
- **Library implementations** (`rank_bm25`, Whoosh): ruled out by the project constraints.
