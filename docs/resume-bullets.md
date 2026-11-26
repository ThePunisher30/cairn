# Resume bullets (draft)

Rules for these: every number is in `docs/results.md`, every bullet comes with the limit you must be ready to state, and none claims more than the data supports. Write them in your own words and cut any you cannot defend. Say that AI assistance was used if asked: the README says it (the assistant wrote Phases 8 to 11 and drafted the evaluation questions at your request).

## Candidates

1. **Built a retrieval engine from scratch in Python (BM25, dense retrieval, IVF and HNSW indexes, rank fusion, citations) over the FastAPI docs and source, evaluated on 60 questions with file-and-line answers (gold spans verified by tests; 40 development, 20 held out, the held-out set run twice in total).**
   *Be ready to say:* the questions were drafted with AI help; the held-out set has 20 questions.

2. **Compared it with LlamaIndex + FAISS using a protocol and predictions written before measuring: the default LlamaIndex pipeline scored hit@5 0.525 against 0.825 for my dense retriever on development questions; a tuned one matched it (0.775), and on 20 held-out questions no system was statistically distinguishable.**
   *Be ready to say:* I tuned Cairn on the same development questions; 3 of my 8 predictions were wrong; the held-out set is small.

3. **Diagnosed why: 46% of the framework's default chunks exceed the embedding model's 512-token window (3% in mine), and found that LlamaIndex's reciprocal-rank fusion silently reverses a dense ranking when used with the FAISS L2 index its documentation shows (MRR 0.391, 0.614 once fixed).**
   *Be ready to say:* the first is measured (chunk sizes, shrinking chunks raised hit@5 step by step); the second was checked by reading score directions directly.

4. **Wrote an HTTP service (FastAPI) with a live index (tombstone deletes, a SQLite manifest, incremental re-indexing: 3 of 45 new chunks re-embedded after changing 3 of 671 files), caching, rate limiting, a circuit breaker with a retrieval-only fallback, request ids and Prometheus metrics; load-tested it and fixed the first bottleneck (embedding threads fighting each other) for 1.5 to 1.7 times the throughput.**
   *Be ready to say:* the load test used a fake language model, one machine and one process; run-to-run spread was 20% or more.

5. **Added CI with a retrieval-regression gate (deterministic, 2 seconds, no network) that failed a chunk-size change which passed all 1,738 unit tests, and a Docker image that starts in 10 seconds in 93 MiB (BM25 only).**
   *Be ready to say:* it passed all unit tests only after I fixed one hard-coded test of my own; it was one of seven regressions I tried, and both the gate and the unit tests caught the other six.

6. **Built an agent loop with tools, hard budgets and a path-traversal-safe file reader, and measured it honestly against single-shot answering: it did not help with the small local model I used (23 of 40 against 29, p = 0.24) at 1.8 to 2.7 times the tokens.**
   *Be ready to say:* one run each, a small model, a same-model judge.

## What not to write

- "Beats LlamaIndex", "better than the frameworks", "state of the art", any speed-up claim without "on a synthetic 100,000-vector set".
- Anything about a live demo: it is not deployed.
- Anything that hides that an AI assistant wrote the later phases.

## For the mock interview

I can interrogate you phase by phase, using `docs/notes/phase11.md` (defend every number) and the checkpoint notes of Phases 7 to 10. It needs you in the conversation, so it is the one thing here I cannot do alone.
