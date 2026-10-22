# ADR 0003: Dense retrieval with bge-small and a flat index

Status: accepted, 2026-10-23. Evidence: `docs/results.md`, section "Phase 4"; `eval/failures/phase4.md`.

## Context

BM25 (ADR 0002) misses 8 of 40 dev questions at k = 10, mostly because the question paraphrases the passage.
Phases 5-6 need a dense baseline to speed up (ANN indexes) and to combine with BM25 (hybrid retrieval). The
corpus is the 3,267 `structure-300` chunks of the FastAPI docs and code. I built the embedding wrapper, the
cache, the index and the retriever myself; only the model's weights come from a library
(`sentence-transformers`), and the model registry and the offline test model were provided.

## Decision

- **Model:** `bge-small` (384 dimensions, reads at most 512 tokens). Documents are embedded as their heading
  path plus text (`chunk.index_text`); questions get bge's query prefix ("Represent this sentence for
  searching relevant passages: "), documents get none.
- **Vectors:** `float32`, each one scaled to length 1, so the dot product equals the cosine similarity.
  A zero vector stays zero.
- **Index:** a flat matrix, searched by brute force: `vectors @ query`, then `np.argpartition` for the top
  k and a fixed tie rule (equal scores go to the lower chunk number). It stores the chunk IDs it was built
  from, and the retriever refuses chunks that do not match.
- **File format:** an `.npz` with a version number, loaded with `allow_pickle=False` (no pickle).
- **Cache:** one file of vectors keyed by `sha256(model name + doc prefix + "\n" + text)`, where the text is
  the exact string the model saw. Different models share the file. Identical texts are embedded once.
- **Batching:** texts are sorted by length before batching (batch size 16) and the vectors are put back in the
  original order.
- **Measured on dev:** MRR 0.711, hit@1 0.600, hit@10 0.925, recall@10 0.863; 3 questions missed (BM25: 8);
  p50 search time 112 ms, p95 265 ms; 3,267 x 384 vectors = 5.0 MB; the first build took 471 s.

## Consequences

- Dense recalls more than BM25 (hit@10 0.925 against 0.800) but does not rank better at the top (hit@1 is
  0.600 for both), and it is worse on exact-word questions (lexical MRR 0.817 against 1.000).
- The two fail on different questions: at rank 1, both are right on 17, only BM25 on 7, only dense on 7,
  neither on 9. At least one is right on 31 of 40. This is the evidence for hybrid retrieval in Phase 6;
  dense does not replace BM25.
- Most dense failures are a neighbouring chunk of the right page (7 of 16), or a chunk about the same
  topic from another page. Two failures (q005, q007) may have a debatable label.
- A search is about 40 times slower than BM25 (112 ms against 2.6 ms), almost all of it the model embedding the
  question on the CPU (38.5 ms alone; the eval's 112 ms is higher and unexplained). The brute-force scan takes
  0.9 ms; ANN indexes (Phase 5) cut the scan, not the embedding.
- 83 chunks are longer than the model's 512-token limit and are truncated; the model never sees their tail. I
  have not checked whether that caused any dev failure.
- The query prefix is part of the model setup: dropping it costs 0.106 MRR on `structure-300`. The cache is not
  affected by it (only documents are cached), but the retriever must be built with the same prefix setting that
  was measured.
- An index must be rebuilt, and the cache keyed, whenever the model or the document prefix changes; the key and
  the chunk-ID check make a stale one an error or a recompute, not a wrong answer.
- Memory is `n x d x 4` bytes: 153.6 MB per 100,000 chunks. A flat index stays practical to a few hundred
  thousand chunks, and is not tried beyond that.
- The settings were chosen on 40 dev questions, where one question is worth 0.025. The test split has not been
  used.

## Alternatives considered

- **`minilm` (`all-MiniLM-L6-v2`):** MRR 0.492 and hit@10 0.800, with lexical MRR 0.526. Clearly worse.
- **No query prefix:** MRR 0.605, hit@10 0.825, 4 more questions missed.
- **Other chunkers with dense:** `fixed-200`, `fixed-400` and `recursive-400` score MRR 0.431 to 0.457 (hit@1
  0.250 to 0.325) against 0.711. For `fixed-400` and `recursive-400` this is below their BM25 MRR (0.597,
  0.562); only `structure-300` is above BM25. The prefix changes those three by at most 0.012 MRR.
- **Cache key from `chunk.content_hash`:** it covers only the chunk's text. Two chunks with the same text under
  different headings would share one vector although the model sees different strings, and switching the model
  or the document prefix would return the old vectors (`bge-small` and `minilm` both have 384 dimensions, so
  nothing would even crash).
- **Unsorted batches:** 5.4 chunks/s against 8.9 chunks/s sorted on a sample of 160 chunks.
- **Pickle for the index:** can run code when loaded; `.npz` without pickle cannot.
- **Approximate indexes now:** not needed at 5 MB; this is Phase 5, measured against this flat index as the
  exact answer.
- **Vector databases and libraries** (`faiss`, `chroma`, `pgvector`): ruled out by the project constraints
  until Phase 11.
