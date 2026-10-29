# Cairn

A from-scratch RAG engine over the [FastAPI](https://github.com/fastapi/fastapi) docs and source code (pinned to tag 0.143.0). Chunking, BM25, dense retrieval, the vector indexes (flat, IVF, HNSW) and the evaluation harness are written by hand: no LangChain, LlamaIndex, FAISS, Chroma or `rank_bm25`. The only library doing model work is `sentence-transformers`, which runs the embedding model.

The point of the project is to understand every layer and to measure each choice, so every number below traces to a row in [`docs/results.md`](docs/results.md).

**Status: Phases 0-5 of 11 are done.** Retrieval works and is measured. Generation, the agent loop, serving and the demo are not built yet (see [Roadmap](#roadmap)).

## Results so far

40 dev questions, 3,267 chunks, k = 10. One question is worth 0.025 on a hit rate, so differences under about 0.05 are noise.

| Retriever | hit@1 | hit@5 | hit@10 | recall@10 | MRR | nDCG@10 | p50 search |
|---|---|---|---|---|---|---|---|
| Oracle (upper bound) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| Random / first-N (floor) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | n/a |
| BM25 | 0.600 | 0.750 | 0.800 | 0.750 | 0.660 | 0.658 | 2.6 ms |
| Dense (`bge-small`) | 0.600 | 0.825 | 0.925 | 0.863 | 0.711 | 0.723 | 111.7 ms |

What the measurements say:

- **Dense recalls more, but does not rank better at the top.** hit@10 goes from 0.800 to 0.925 (3 questions missed instead of 8), but hit@1 is 0.600 for both.
- **Each wins where the other loses.** Dense is far better on paraphrased questions (semantic MRR 0.432 to 0.724). BM25 is better on exact-word questions (lexical MRR 1.000 against 0.817).
- **The two fail on different questions.** At rank 1, both are right on 17 questions, only BM25 on 7, only dense on 7, neither on 9. At least one is right on 31 of 40. That is the evidence for hybrid retrieval (Phase 6).
- **Chunking mattered most.** Structure-aware chunking (Markdown headings plus a Python AST chunker) keeps 0.88 of the answers whole in 133-token chunks, against 0.78 for fixed 400-token chunks. With dense retrieval it scores MRR 0.711 against 0.43 to 0.46 for the size-based chunkers.
- **Small implementation details were worth a lot.** Splitting `snake_case` and `camelCase` identifiers lifted BM25 MRR from 0.552 to 0.660. Dropping bge's query prefix cost dense retrieval 0.106 MRR. Turning off BM25's length normalisation halves its score.
- **Dense search is about 40 times slower** than BM25 per query, and almost all of it is the model embedding the question on a CPU: the scan over the 3,267 vectors takes 0.9 ms.

- **Approximate indexes (Phase 5).** IVF and HNSW, written from scratch, are measured against the exact flat index on a **synthetic** 100,000-vector set (mixes of real chunk vectors plus noise). To reach recall 0.99, HNSW computes 560 distances per query against 100,000 for flat, and is about 3 to 5 times faster in time (the walk is a Python loop); it takes 482 s to build against 31 s for IVF. The neighbour-selection heuristic is the biggest single factor (recall 0.993 against 0.913 at the same search width). On the real 3,267-chunk corpus neither index helps: a flat scan takes 0.36 ms, so the flat index stays.

Failures are analysed one by one in [`eval/failures/phase4.md`](eval/failures/phase4.md) (16 dense failures, classified by cause).

## How it is evaluated

- **60 questions** about FastAPI in four categories (lexical, semantic, multi-chunk, code lookup), 15 each, split into 40 dev and 20 test questions. The test split is frozen: it will be run twice in the whole project (after Phase 6 and at the end) and `run_eval.py` refuses it without `--use-test-set`. All tuning is done on the dev questions.
- **Gold answers are file and line ranges, not chunk IDs**, so they survive any change of chunking. A retrieved chunk is a hit if its lines overlap a gold range.
- **The questions were drafted with AI assistance** from passages chosen with a helper script. Gold spans are checked by tests, and `scripts/question_overlap.py` reports how many of a question's words also appear in its answer passage, per category. Questions written after reading a passage tend to echo it somewhat, and 60 questions is a small set.
- **Metrics:** hit@k, recall@k, MRR and nDCG@k, plus latency p50 and p95. Two baselines (random, first-N) and an oracle check that the harness can tell good from bad.

## Architecture (built so far)

```
 FastAPI repo (pinned tag)
        │  scripts/fetch_corpus.py
        ▼
 loaders ─► chunkers (fixed / recursive / Markdown / Python AST) ─► data/chunks.jsonl
                                                                         │
                         ┌───────────────────────────────────────────────┤
                         ▼                                               ▼
              BM25: tokenizer ─► inverted index                 Embedder (bge-small) + disk cache
              (heap top-k, pickle of a plain dict)              ─► flat index (.npz, brute-force top-k)
                         │                                               │
                         └──────────────► Retriever.search(query, k) ◄───┘
                                                      │
                                      evaluation harness: metrics, baselines, reports
```

Every retriever has the same interface (`search(query, k) -> list[Hit]`), so the harness scores any of them the same way.

## Reproduce the numbers

Python 3.11 or newer. On Windows use a short-path virtual environment (the long-path limit breaks the install in deep folders).

```bash
pip install -e ".[dev,dense]"

python scripts/fetch_corpus.py        # clone FastAPI 0.143.0 into data/raw (idempotent)
python scripts/ingest.py              # chunk it into data/chunks.jsonl
python scripts/build_index.py         # BM25 index
python scripts/embed_chunks.py        # dense index (about 8 minutes on a laptop CPU the first time, then cached)

python scripts/run_eval.py --retriever bm25
python scripts/run_eval.py --retriever dense
python scripts/compare_retrievers.py  # where BM25 and dense disagree, question by question
```

Tests and lint: `pytest -q` (602 tests, no network or model download needed: the dense tests use a small offline stand-in model) and `ruff check .`.

The ablation scripts behind the tables in `docs/results.md` are `scripts/compare_chunkers.py`, `scripts/ablate_bm25.py`, `scripts/ablate_dense.py`, `scripts/bench_search.py` and `scripts/bench_embed.py`.

## Repository layout

```
src/cairn/
  ingest/       documents, chunkers, pipeline
  retrieval/    tokenizer, inverted index, BM25
  dense/        embedder, embedding cache, flat index, dense retriever, model registry
  evaluation/   metrics, questions, baselines, harness, chunk-integrity metric
scripts/        one runnable script per task (ingest, build_index, embed_chunks, run_eval, ablations)
eval/           questions.jsonl (data only) and failures/ (failure analyses)
tests/          unit and script tests
docs/           results.md (every number), adr/ (decisions), notes/ (what I learned per phase), briefs/
PLAN.md         the full 11-phase plan and its status
```

Decisions are recorded as short ADRs: [chunking](docs/adr/0001-chunking.md), [BM25 settings](docs/adr/0002-bm25-settings.md), [dense retrieval](docs/adr/0003-dense-retrieval.md), [approximate indexes](docs/adr/0004-ann-indexes.md).

## Roadmap

| Phase | Topic | Status |
|---|---|---|
| 0 | Foundations: package, config, pinned corpus fetch | done |
| 1 | The yardstick: 60 questions, metrics, harness, baselines | done |
| 2 | Ingestion and chunking (4 chunkers, gold-integrity metric) | done |
| 3 | BM25 from scratch | done |
| 4 | Dense retrieval: embedder, cache, flat index | done |
| 5 | Approximate nearest-neighbour indexes (k-means, IVF, HNSW) from scratch | done |
| 6 | Hybrid retrieval (rank fusion) and cross-encoder reranking; first run of the test split | next |
| 7 | Generation with citations and abstention, LLM client, generation eval | planned |
| 8 | Agent loop with tools | planned |
| 9 | FastAPI service: streaming, incremental indexing, caching, rate limiting, load test | planned |
| 10 | Frontend, Docker, CI with a retrieval-regression gate, deployment | planned |
| 11 | Benchmark against LangChain / LlamaIndex and a vector database; write-up | planned |

The details of each phase are in [`PLAN.md`](PLAN.md).

## How it was built

This is a learning project built with an AI assistant (Claude) as tutor and pair: it wrote the phase briefs and the tests that each phase has to pass, helped with the implementation, and reviewed the code. The evaluation questions were drafted the same way (see above). The experiments, the measurements and the failure analyses are run on the real corpus, and the notes in `docs/notes/` record what each phase taught.

The API key (needed from Phase 7) is read from the `ANTHROPIC_API_KEY` environment variable and is never in the code.
