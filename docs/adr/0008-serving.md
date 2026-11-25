# ADR 0008: A service with a live index: tombstones, a manifest committed last, thread-pool routes, and caches that are keyed on the index version

Status: accepted, 2026-11-26. Evidence: `docs/results.md`, section "Phase 9".
Every number was measured on one laptop, with the load generator on the same machine, against a **fake language model** that answers instantly, so what is claimed about the service is narrow. Section "What I would not trust" says where.

## Context

Phase 9 turns the Phase 7 pipeline into a service (`src/cairn/serve/`): `POST /ask` (JSON or server-sent events), `POST /search`, `POST /ingest` (a background job), `GET /health`, `GET /metrics`. It has to be updated without a restart, stay up when the language model does not, and be debuggable. Libraries: FastAPI and uvicorn only (the `serve` extra); everything else (the live index, the manifest logic, the caches, the limiter, the breaker, the metrics) is written here.

**Who wrote it.** In this phase the AI assistant wrote the implementation at the author's request, with the tests; they were checked by breaking the code on purpose (about thirty broken variants; those that survived led to new tests, except one guard, never caching a degraded answer, that no route can reach today because the failure path returns before caching).

## Decision

- **Load in the background, report readiness.** The server listens at once; a `Service` loads the indexes and the embedder in a thread and warms them with one search. `GET /health` is `503` with the state (`loading`, `warming`, `failed` with the error) until it is ready, so a load balancer holds traffic back and a failed start is visible instead of a hang. The other routes answer `503` with `Retry-After` while not ready, after the key is checked (a stranger learns nothing).
- **One live index, slots and tombstones** (`serve/live.py`). Chunks, BM25 postings and dense vectors share slot numbers that are never reused. Removing a file marks its slots dead; adding appends. The BM25 statistics (document count, document frequencies, average length) count only living slots, so scores equal those of an index built fresh from the same chunks (a randomized test checks this against `InvertedIndex.build`). Compaction rebuilds without the dead slots when more than 25% (and at least 50) are dead.
- **A reader-writer lock** (written here, writer-preferring). Searches read concurrently; an update takes the write lock only for the in-memory change, after the slow part (chunking and embedding) is done outside it. A reader sees a file entirely old or entirely new, never a mix (tested with threads). The query embedding is computed before the read lock is taken.
- **Incremental indexing with a SQLite manifest** (`path -> hash of the file's text`). A re-index diffs the manifest against the folder (added, changed, removed, unchanged), re-chunks only the changed files, embeds only chunks whose text has no old vector (vectors of dropped chunks are reused by a hash of the chunk text), applies the change atomically, **saves the index files, and only then commits the manifest**. A crash in between leaves the manifest behind the index and the next re-index redoes those files; the reverse order could claim files were indexed that were not. An empty manifest is filled at start by re-chunking and comparing chunk ids with the index, so it never claims more than is true.
- **Routes are plain `def`**, so FastAPI runs them in a thread pool. A CPU-heavy call on the event loop freezes every request, health checks included (measured below). The model client is **forked per request** (own `last_usage`, shared HTTP client), because the streaming path reads token counts from the client after the stream ends and two requests at once would read each other's.
- **Caches key on the index generation.** An answer is stored under (generation, k, model, normalised question); any change to the index bumps the generation, so an answer written from old chunks cannot be found (a test removes the explicit clear and checks this). Failures (`degraded`) are never cached. A semantic cache exists but is **off by default**.
- **Protection:** one API key from the environment (never the command line; at least 16 characters, compared in constant time); a token bucket per client address (`429` with `Retry-After`); a concurrency limit on retrieval that refuses with `503` after 2 s (backpressure); a circuit breaker on the model (3 failures in a row open it for 30 s, then one probe).
- **When the model fails, return what was retrieved** (`200`, `degraded: true`, the chunks and the error) instead of an error; a stream that fails before its first token sends a `fallback` event, after it an `error` event.
- **Observability:** an `X-Request-ID` per request (a client's id is kept only if it is harmless, since it goes into logs and headers), one JSON access-log line per request, per-stage timings (`retrieve`, `llm_first_token`, `llm_total`) passed explicitly down the call (a thread-local would be lost across the threads of a stream), Prometheus text at `/metrics` with route labels taken from the matched route (a scanner cannot create endless labels).
- **Torch uses one thread per embedding call** (`--torch-threads`, default 1).

## Why

- **Tombstones instead of deleting in place:** deleting from the middle of a postings structure or a matrix means renumbering everything after it. A tombstone is one write; the price (measured: no speed-up and no slow-down, 6 times the dense scan after 75% dead, a 42 ms compaction pause) is paid in memory and a periodic rebuild.
- **The manifest committed last, in SQLite:** a transaction makes a batch of rows change together or not at all, and the order of the two writes decides which way a crash fails. Failing towards "redo a file" is harmless; failing towards "think it is indexed" is not.
- **Reusing vectors by text hash:** the embedding is the only expensive step (about 35 ms per query-sized text, much more for a chunk). In the real-index run, 3 of 45 new chunks were embedded.
- **Torch threads:** measured, not assumed. With torch's default (all cores) the server did about 12 requests/s from 5 to 20 users; with one thread 18 to 24, because the threads of different requests stopped competing for the same cores.
- **Semantic cache off:** with these 15 paraphrases and 18 near-misses no similarity threshold separates "the same question" from "a similar question" (medians 0.93 and 0.94). Serving the answer to a different question is worse than a miss.
- **Fallback to chunks, not an error:** the retrieved chunks are the part of the answer that does not need the model, and the user can read them. The `degraded` flag keeps it honest.

## Trade-offs

- **A shared API key** is simple and enough for one deployer; it gives no per-user identity, so the rate limit is per address.
- **State in the process** (caches, limiter, job history): fast and simple, lost on restart, not shared between processes. Running several processes (which the single-process ceiling of about 20 to 29 searches/s would call for) needs these moved out.
- **The flat dense index** is exact and trivially supports tombstones; it scans every slot. The approximate indexes of Phase 4 are not in the service.
- **Searches during a compaction wait** for the write lock (42 ms on 3,267 chunks, proportional to the index size).

## What I would not trust

- **Any throughput or latency number as "what the service does".** The model was a fake, the load generator shared the machine, levels ran 15 to 20 s, repeats differed by 20% or more, one process, one laptop. The `/ask` rate at 5 users being higher than `/search`'s is unexplained.
- **That one torch thread is right everywhere.** It is the best of 1, 2, 4 and 12 on this 12-core machine under concurrent load; a single request alone was not shown to be slower or faster.
- **The semantic cache evaluation as a statement about semantic caches.** One model, 15 and 18 pairs, written by one person.
- **The failure handling against a real outage.** The breaker, the fallback and the stream events are tested with a scripted model; a real Ollama was never stopped.
- **That incremental indexing equals a fresh index for large changes.** The equality is tested on random small corpora and chunk ids; the real-index run changed three files.

## Consequences

- The next load test should use the real model for a handful of requests (not a sweep), and a second machine for the load generator.
- If more throughput is needed: several worker processes behind one address, with the caches and the limiter moved out of the process (or accepted as per-process).
- If the corpus grows, the dense scan and the compaction pause grow with it; that is where the approximate index would come in, together with its own deletion problem.
- `docs/notes/phase9.md` holds the draft answers to the Phase 9 checkpoint questions, for the author to rewrite.
