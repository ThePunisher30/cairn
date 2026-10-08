# Cairn: Project Plan

A from-scratch RAG engine over a real codebase and its docs. Retrieval, indexing, generation, agents and evaluation are written by you. Frameworks and vector databases are used only at the end, as a benchmark opponent.

**Purpose:** learn how every layer of a RAG system works, and finish with a deployed, measured, SDE-1-level resume project.

---

## 1. Ground rules

### Who writes what

| You write | I write | Libraries allowed |
|---|---|---|
| All core logic: chunkers, BM25, k-means, IVF, HNSW, fusion, prompt builder, LLM client wrapper, agent loop, eval harness, caching, rate limiter, ingestion diffing | Phase briefs, interfaces, **failing tests** for each phase, fake/mocked LLM, CI config, boilerplate (Dockerfile, CSS), code review, explanations | `numpy`, `sentence-transformers` (model runtime only), `tiktoken` or `tokenizers` (token counting), `anthropic` SDK, `httpx`, `fastapi`, `pydantic`, `pytest`, `ruff`, `matplotlib` |

**Banned until Phase 11:** `langchain`, `llama-index`, `chromadb`, `faiss`, `pgvector`, `rank_bm25`, `whoosh`, `scikit-learn` (so you write your own k-means), any "RAG-in-a-box" library.

### The loop for every phase
1. **Brief:** I give you the goal, the interfaces, and a test file that currently fails.
2. **Build:** You implement until the tests pass. Ask for help any time using the hint ladder below.
3. **Measure:** You run the eval and record numbers in `docs/results.md`. Every claim in the final README must trace back to a row there.
4. **Review:** I review your code and explain where it differs from production practice. You fix what matters.
5. **Checkpoint:** You answer the phase's checkpoint questions **in your own words** in `docs/notes/phaseN.md`. A phase isn't done until you can.

### The hint ladder
When you're stuck, climb one rung at a time. Don't skip to the bottom.
1. **Nudge:** a conceptual pointer ("think about what the score does when a term appears in every doc").
2. **Pseudocode:** the algorithm's outline, no code.
3. **Snippet:** at most about 10 lines of the hard part.
4. **I implement it, you rewrite it from memory** the next day without looking.

### Definition of done (every phase)
Tests green, numbers logged in `docs/results.md`, checkpoint notes written, code reviewed, and changes committed on a feature branch with a clean history.

---

## 2. Decisions made (veto them now, because they're costly to change later)

| Decision | Choice | Why |
|---|---|---|
| **Corpus** | The **FastAPI repo**, pinned to one git tag: `docs/en/docs/**/*.md` plus `fastapi/**/*.py` | Mixed prose and code, so you get Markdown chunking *and* AST chunking. It's about 10–20k chunks, laptop-sized. You know Python, so you can write good eval questions. Alternatives with the same shape: `httpx`, `rich`, `pydantic` |
| **LLM contamination caveat** | Models have seen FastAPI in training | Pure generation scores will look inflated. Mitigated by: (a) retrieval metrics don't involve the LLM, (b) eval questions target version-pinned specifics, (c) unanswerable questions test whether the model abstains instead of answering from memory |
| **Embeddings** | `BAAI/bge-small-en-v1.5` (384-d), with a second model for comparison in Phase 4 | Fast on CPU, and you already ran it |
| **Reranker** | A small cross-encoder from `sentence-transformers` | Runs on CPU, and writing a cross-encoder is out of scope |
| **LLM** | Anthropic API behind your own `LLMClient` interface. Ollama implementation as a free local option | Quality for answers, and a swappable boundary to design well |
| **Judge model** | The cheapest model available | Eval runs many calls, and the budget stays small |
| **Storage** | Flat files (`.npy` vectors, `.jsonl` chunks, a binary or JSON postings file) plus SQLite for the manifest | Nothing hidden. You see exactly what's on disk |
| **Language/runtime** | Python 3.12, venv at a **short path** (Windows long-path issue you already hit) | |

---

## 3. End-state architecture

```
                         INGESTION (offline / incremental)
 repo files ──► loaders ──► chunkers ──► Chunk{id,text,path,lines,heading_path,hash}
                                              │
                      ┌───────────────────────┼───────────────────────┐
                      ▼                       ▼                       ▼
                 BM25 index           Embedder + cache         chunk store (jsonl)
              (inverted index)       ──► HNSW / flat index      + manifest (SQLite)

                         QUERY (online)
 question ─► [cache?] ─► BM25 top-50 ┐
                          dense top-50┴─► RRF fusion ─► cross-encoder rerank ─► top-k
                                                                  │
                                           small-to-big expansion ▼
                          prompt builder (budget, order, dedupe, citations)
                                                                  ▼
                       LLM (single-shot)  or  agent loop with tools:
                       search / read_chunk / read_file / list_symbols
                                                                  ▼
                       answer + citations + abstain ─► SSE stream ─► UI
                       every stage emits: latency, tokens, cost, trace
```

---

## 4. Repo layout

```
cairn/
├── PLAN.md
├── pyproject.toml
├── src/cairn/
│   ├── config.py
│   ├── types.py              # Chunk, Hit, Document, Retriever protocol
│   ├── ingest/               # loaders.py, chunkers/{fixed,recursive,markdown,python_ast}.py, pipeline.py
│   ├── retrieval/            # tokenizer.py, bm25.py, embedder.py, flat.py, kmeans.py, ivf.py, hnsw.py,
│   │                         # fusion.py, rerank.py, hybrid.py, context.py
│   ├── llm/                  # base.py, anthropic_client.py, ollama_client.py, retry.py, tokens.py
│   ├── generation/           # prompts/, packer.py, citations.py, answer.py
│   ├── agent/                # tools.py, loop.py, trace.py
│   ├── serve/                # app.py, schemas.py, cache.py, ratelimit.py, jobs.py, logging.py, metrics.py
│   └── indexing/             # manifest.py, incremental.py
├── eval/                     # questions.jsonl, metrics.py, run.py, gen_eval.py, judge.py, failures/
├── bench/                    # ann.py, load_test.py, frameworks/ (Phase 11)
├── scripts/                  # fetch_corpus.py, make_scale_set.py
├── tests/                    # I write these per phase, you make them pass
├── docs/                     # results.md, notes/phaseN.md, adr/
└── data/                     # raw/ (gitignored), chunks.jsonl, indexes/
```

---

## 5. Phases

Time estimates assume about **8–10 focused hours a week**. Doing it full-time cuts the calendar roughly in half.

### Phase 0: Foundations (2–3 days)
**Goal:** a reproducible repo where everything later runs with one command.

**You write**
- Git repo, `pyproject.toml` with pinned deps, `ruff` + `pytest` config, the package skeleton above.
- `scripts/fetch_corpus.py`: clones FastAPI at a pinned tag into `data/raw/`. Re-running changes nothing.
- `config.py`: env-driven settings (paths, model names, API key from env, never from code).
- Port `chunk_demo.py` from the playground into the package as a first smoke test.

**Learn:** Python packaging, reproducible environments, config hygiene, secret handling.

**Done when:** `pytest` runs green on a fresh clone and the corpus fetch is idempotent.

**Checkpoint:** Why pin the corpus tag? Why must the API key never be in the repo?

---

### Phase 1: The yardstick, built before anything it measures (4–5 days)
**Goal:** a trustworthy way to score any retriever, and a labeled question set.

**You write**
- **Read the corpus.** Skim the docs and the main modules. This is where you learn the domain, and you need it to write good questions.
- `eval/questions.jsonl`: **60 hand-written answerable questions**, 15 in each category:
  - *lexical* (exact names, error text, parameter names),
  - *semantic* (paraphrases that share no words with the answer),
  - *multi-chunk* (answer needs two or more passages),
  - *code lookup* ("where is X implemented", "what does Y do").
- Gold labels at the **source level**: `(file, start_line, end_line)`, **not** chunk IDs. Chunk boundaries will change in every experiment, so gold must survive re-chunking. A retrieved chunk is a **hit** if its line range overlaps a gold span.
- Stratified **dev/test split (about 70/30)**. The test set is frozen: you may run it exactly twice, once after Phase 6 and once in Phase 11.
- `eval/metrics.py`: `hit_at_k`, `recall_at_k`, `mrr`, `ndcg_at_k`.
- `eval/run.py`: takes any object implementing `Retriever.search(query, k) -> list[Hit]` and prints a table by category, latency p50/p95, and a per-question failure list.
- Two dumb baselines (random, first-N chunks) to prove the harness can tell good from bad.

**Learn:** information-retrieval metrics, why evals come first, labeling bias, why gold must be defined independently of the system under test.

**Done when:** metric unit tests (hand-computed cases, supplied by me) pass, and the harness scores both baselines near zero.

**Checkpoint:** When is MRR better than recall@k, and when is it worse? What breaks if gold labels are chunk IDs? Why is a frozen test set necessary?

---

### Phase 2: Ingestion and chunking (5–7 days)
**Goal:** turn raw files into well-formed, traceable chunks, with several strategies you can compare.

**You write**
- `Document` and `Chunk` types: stable `id` (a hash of path and span and text), `text`, `path`, `start_line`, `end_line`, `heading_path`, `kind` (`prose` or `code`), `content_hash`, `token_count`.
- Loaders for Markdown and Python files.
- **Four chunkers** behind one `Chunker.split(doc) -> list[Chunk]` interface:
  1. Fixed-size (the naive baseline).
  2. Recursive (paragraph, then sentence, then word) with overlap.
  3. Markdown heading-aware with breadcrumb prefixes (the one you saw in the demo).
  4. **Python AST chunker** using the stdlib `ast` module: one chunk per function, class or method, with signature, docstring and module path.
- Token counting with a real tokenizer, so sizes are in tokens and not characters.
- `ingest/pipeline.py`: deterministic ingest to `data/chunks.jsonl`, plus a stats report (chunk count, token-length histogram, how many chunks exceed the embedding model's 512-token limit).
- **Gold-integrity metric** in the eval folder: for each chunker, the fraction of gold spans that fall entirely inside one chunk. This is an "oracle" score that measures how badly chunking fragments answers *without needing a retriever yet*.

**Learn:** the chunk-size trade-off with real data, AST parsing, deterministic IDs, why truncation silently hurts embeddings.

**Done when:** chunker tests pass, ingest is idempotent (same input gives identical chunk IDs), and `docs/results.md` has a gold-integrity table per chunker.

**Checkpoint:** Why do overlapping chunks help integrity but hurt precision? What happens to a 900-token chunk fed to a 512-token model? Why a stable chunk ID?

---

### Phase 3: BM25 from scratch (5–6 days)
**Goal:** a real keyword search engine, built from the formula up.

**You write**
- `tokenizer.py`: lowercase, regex split, **code-aware** (split `snake_case` and `camelCase` while also keeping the original token), optional stopwords.
- Inverted index: `term -> postings list of (chunk_idx, tf)`, plus doc lengths and average length.
- BM25 scoring with `k1` and `b`, written from the formula. Top-k with a heap, not a full sort.
- Persist and load the index (start with `pickle`, then design a compact on-disk format).
- First **real eval run**, which becomes your baseline row in `results.md`. Tune `k1`, `b` and tokenizer variants **on dev only**.

**Learn:** TF-IDF to BM25, IDF intuition, length normalization, why tokenization decides code-search quality, index complexity.

**Constraint:** no `rank_bm25`, no `scikit-learn`, no Whoosh.

**Done when:** scores match a hand-computed reference on a toy corpus (my test), p95 query latency is under 20 ms on the full corpus, and the ablation table (tokenizer variants × chunker) is logged.

**Checkpoint:** What does `b=0` do? Why does a term in every document contribute about nothing? Why did splitting `camelCase` help or hurt on your data?

---

### Phase 4: Dense retrieval (4–5 days)
**Goal:** semantic search over your own embeddings, with disk caching.

**You write**
- `Embedder` wrapper: batching, L2 normalization, **query-vs-document prefix handling**, truncation awareness.
- **Embedding cache** on disk, keyed by `(content_hash, model_name)`, so re-ingesting unchanged chunks costs nothing.
- `FlatIndex`: brute-force matrix multiply with `np.argpartition` for top-k. Save and load as `.npy`.
- Eval runs: dense × each chunker. **Experiments:** prefix on/off, two embedding models, embedding throughput on CPU.
- `eval/failures/phase4.md`: classify at least 15 failures by cause (wrong chunk, exact-token miss, too generic, label error). Naming failure modes is the skill being practiced.

**Learn:** what embeddings capture and miss, normalization, cache design, throughput and memory math.

**Done when:** dense numbers sit in `results.md` next to BM25, and you can explain from your own failure analysis where each wins.

**Checkpoint:** Why does dot product equal cosine here? What did dropping the query prefix cost, and why? What's your memory per 100k chunks?

> **Shippable checkpoint A:** a working retrieval demo (CLI that prints ranked, cited chunks).

---

### Phase 5: ANN indexes from scratch (7–10 days, the hardest phase)
**Goal:** understand why vector databases exist by building the index inside them.

**You write**
- **Part A, warm-up:** your own **k-means** (k-means++ init), then **IVF-Flat** (`nlist` clusters, `nprobe` probes). Measure recall@10 against brute force and the speedup.
- **Part B, the main event:** **HNSW**. Layer assignment (`ml`), greedy search per layer, insertion with neighbor selection, parameters `M`, `efConstruction`, `efSearch`. Use numpy for batched distance computation.
- `scripts/make_scale_set.py`: your corpus is only about 10–20k chunks, where brute force wins. Build a 100k–200k vector set (embed extra Python-ecosystem docs, or generate synthetic clusters) so ANN has something to prove.
- `bench/ann.py`: **recall@10 vs queries/sec curves** for flat, IVF and HNSW, plus build time and memory. Plot with `matplotlib`.
- Persistence for both indexes.

**Learn:** curse of dimensionality, navigable small-world graphs, the recall/latency/memory triangle, why index build parameters are a product decision.

**Done when:** HNSW reaches recall@10 of at least 0.95 at some `efSearch`, beats brute force on queries/sec at 100k+ vectors, and the curves are in `results.md`. If pure Python is too slow to build, that's a finding to write up, with an optional `numba` fix.

**Checkpoint:** Why can't you build a perfect index for nearest-neighbor in high dimensions? What does raising `efSearch` do, and what does raising `M` do? Why does your corpus not need ANN?

**Stretch:** product quantization (vector compression).

---

### Phase 6: Hybrid retrieval and reranking (5–6 days)
**Goal:** combine retrievers and measure what each addition actually buys.

**You write**
- **Reciprocal Rank Fusion**, plus weighted score fusion with min-max or z-score normalization. Observe why raw BM25 and cosine scores can't be added directly.
- Cross-encoder reranker wrapper (batching, truncation, score caching).
- `HybridRetriever`: BM25 top-50 and dense top-50, fuse, rerank the top 30, return 5–8.
- **Small-to-big expansion:** retrieve a small precise chunk, then hand the LLM its parent section or surrounding function (`retrieval/context.py`).
- The **ablation table** (the most valuable artifact so far): BM25, dense, hybrid-RRF, hybrid+rerank, ± expansion, per category, with latency.
- **First run of the frozen test set** to measure your dev/test gap.

**Learn:** why scores from different systems aren't comparable, rank fusion, the retrieve-then-rerank pattern and its latency cost, overfitting to a dev set.

**Done when:** hybrid+rerank beats the best single retriever on dev hit@5, or you've written a documented explanation of why it didn't (that's an acceptable outcome if you can defend it).

**Checkpoint:** Why does RRF need no score normalization? Which question category improved most from reranking, and why? What was your dev-vs-test gap?

---

### Phase 7: Generation (6–8 days)
**Goal:** turn retrieved context into grounded, cited, abstaining answers.

**You write**
- `LLMClient` interface (`complete`, `stream`) with an Anthropic implementation and an Ollama implementation. **You** write timeouts, retries with exponential backoff and jitter, and error classification.
- **Context packer:** token budget, de-duplication, ordering (experiment with best-first vs "lost in the middle"), truncation policy.
- Versioned prompt templates as files in `generation/prompts/`.
- **Citations:** number the chunks in the prompt, parse `[n]` from the output, validate that every citation refers to a real chunk, and render `file:line` links.
- **Abstention:** refuse when retrieval confidence is low (rerank-score threshold you calibrate on dev) or when the model says it can't find the answer.
- **Streaming** generator and per-call token and cost accounting.
- **Prompt-injection tests:** plant a malicious instruction inside a corpus chunk and verify your delimiters and system prompt hold. Document what does and doesn't work.
- **Generation eval** (`eval/gen_eval.py`): reference answers for the dev questions, plus **10 unanswerable questions**. Metrics: answer correctness (LLM judge with a rubric), faithfulness, citation precision and recall, abstain precision and recall, cost per question.
- **Calibrate your judge:** hand-grade 15 answers and measure agreement with the judge. This teaches you how much to trust LLM-as-judge numbers.

**Learn:** context windows in practice, grounding, prompt design as code, evaluating non-deterministic output, API robustness.

**Done when:** end-to-end `ask("...")` returns a cited answer, abstains on the unanswerable set at a measured rate, and all numbers are in `results.md` with the judge-agreement figure.

**Checkpoint:** How can a faithful answer still be wrong? What does your abstain threshold trade off? What did the injection test reveal?

> **Shippable checkpoint B:** an end-to-end RAG system with measured quality. **Phases 0–7 alone make a strong resume project.** Phases 8–11 add depth and polish.

---

### Phase 8: Agent loop (5–7 days)
**Goal:** understand agents as "a `while` loop, a tool schema, and good bookkeeping".

**You write**
- Tool definitions with JSON schemas: `search(query, k, path_filter)`, `read_chunk(id)`, `read_file(path, start, end)`, `list_symbols(path)`.
- **The loop yourself** (using the model API's native tool-use format): call the model, execute requested tools, feed back results, repeat until a final answer or the budget runs out. Handle malformed tool calls, tool errors, and duplicate-call detection.
- **Budgets:** max steps, max tokens, wall-clock timeout. All enforced, all tested.
- `agent/trace.py`: a structured trace of every step (thought, tool, args, result size, tokens, latency).
- **15 multi-hop questions** added to the eval ("which function handles X, and what does its dependency Y do?").
- Experiment: single-shot RAG vs the agent on accuracy, latency and cost.

**Learn:** tool use, control flow of agents, failure handling, why agents cost more, when they're worth it.

**Done when:** the loop never exceeds its budget (tested against my scripted `FakeLLM`), and you've written up the accuracy-vs-cost trade-off with numbers, including cases where the agent did worse.

**Checkpoint:** What is the agent actually doing that single-shot RAG can't? How do you stop infinite loops? Where did the agent waste calls?

---

### Phase 9: Serving and production concerns (7–9 days)
**Goal:** a service that behaves under load, can be updated, and can be debugged.

**You write**
- **FastAPI** app: `POST /ask` (SSE streaming), `POST /search`, `POST /ingest` (async job), `GET /health`, `GET /metrics`. Pydantic request and response schemas, API-key header auth.
- **Index persistence:** load at startup, with a clear warm-up and readiness state.
- **Incremental indexing:** a manifest (`path -> content hash`) in SQLite, a diff on re-ingest, and add, update and delete of chunks in BM25 and the vector index. Deleting from HNSW forces you to learn **tombstones** and rebuild thresholds.
- **Caching:** exact-match LRU first, then an optional **semantic cache** (embedding-similarity threshold). Measure hit rate, and collect examples of false hits.
- **Rate limiter:** a token bucket you write yourself.
- **Observability:** request IDs, structured JSON logs, per-stage latency (retrieve, rerank, time-to-first-token, total), tokens and cost per request, and a `/metrics` endpoint.
- **Resilience:** what happens when the LLM provider is down or slow? Timeouts, a graceful error, and a fallback to returning retrieved chunks only.
- **Load test** (`bench/load_test.py`): p50/p95/p99 at 1, 5 and 20 concurrent users. Find the first bottleneck, and fix it. Learn why CPU-bound embedding must go to an executor, not block the event loop.

**Learn:** async vs threads, streaming protocols, cache invalidation, incremental data pipelines, observability, backpressure.

**Done when:** a re-ingest after changing three files re-embeds only those files (proven by logs and a test), the load test is in `results.md` with a bottleneck analysis, and killing the LLM provider doesn't crash the service.

**Checkpoint:** Why does your event loop stall during embedding, and how did you fix it? What's a semantic cache's failure mode? What does a tombstone cost you?

> **Shippable checkpoint C:** a production-style service.

---

### Phase 10: Frontend, packaging, CI and deployment (5–7 days)
**Goal:** a link a recruiter can click and a repo they can trust.

**You write** (I scaffold the HTML and CSS, since design isn't the goal)
- UI logic: consume the SSE stream, render streamed tokens, citation chips that open a source viewer at the right file and line, an "I don't know" state, and a latency and cost footer.
- **Dockerfile** (multi-stage) and `docker-compose.yml` (API plus optional Ollama). I review.
- **GitHub Actions:** lint, tests, and a **retrieval-regression gate**: the deterministic CPU-only eval on a small fixture corpus fails the build if hit@5 drops below a threshold. No API cost in CI.
- **Deploy** to a free or cheap host. The embedding model needs real RAM, so check limits first (Hugging Face Spaces gives the most free RAM; free tiers change, so verify at the time). A fallback is deploying with BM25 plus the small model.
- **README:** architecture diagram, results table, demo GIF, a "reproduce my numbers" section. Short **ADRs** in `docs/adr/` for 4–5 major decisions (chunking, hybrid design, ANN choice, caching).

**Learn:** shipping, CI as a quality gate, writing for a skeptical reader.

**Done when:** a stranger can clone, run one command, and reproduce your retrieval numbers, and the deployed demo answers questions.

**Checkpoint:** What does your CI gate catch that unit tests don't? What did the deploy force you to change?

---

### Phase 11: Benchmark against the frameworks, then write it up (4–5 days)
**Goal:** prove you understand what the frameworks hide.

**You write**
- Rebuild the same pipeline in **LangChain or LlamaIndex**, with **pgvector, FAISS or Chroma**, using defaults first, then tuned. Plug it into your eval harness through a thin adapter.
- Compare **retrieval quality, latency, memory, build time, and lines of code**. Explain every gap using what you built. For example, "their default chunker split mid-function, which cost 9 points of hit@5".
- **Final frozen test-set run.** Report dev and test numbers side by side.
- Write-up (blog post or long README section) with the results table.
- **Resume bullets** with real numbers, and a **mock interview**: I'll interrogate you on every phase.

**Done when:** you can defend every number in the README without opening the code.

---

## 6. Evaluation design (cross-cutting)

| Question set | Count | Added in | Purpose |
|---|---|---|---|
| Lexical | 15 | Phase 1 | BM25's home turf |
| Semantic | 15 | Phase 1 | Where embeddings must win |
| Multi-chunk | 15 | Phase 1 | Tests recall at higher k and expansion |
| Code lookup | 15 | Phase 1 | Tests AST chunking and code tokenization |
| Unanswerable | 10 | Phase 7 | Tests abstention |
| Multi-hop | 15 | Phase 8 | Tests the agent |

- **Retrieval metrics:** hit@k, recall@k, MRR, nDCG, latency p50/p95. **Generation metrics:** correctness, faithfulness, citation precision/recall, abstain precision/recall, cost per question.
- **Dev/test discipline:** tune on dev. The test set is touched twice in the whole project.
- **Synthetic questions** (LLM-generated) are allowed as a **supplement** only. They bias toward lexical overlap with the source, so your hand-written set stays the primary yardstick.
- Every experiment is a row in `docs/results.md`: date, git commit, config, numbers. No unlogged claims.

---

## 7. Timeline

| Phase | Topic | Days | Cumulative weeks (at 8–10 h/week) |
|---|---|---|---|
| 0 | Foundations | 2–3 | 0.5 |
| 1 | Yardstick (corpus, eval, metrics) | 4–5 | 1.5 |
| 2 | Ingestion and chunking | 5–7 | 3 |
| 3 | BM25 | 5–6 | 4 |
| 4 | Dense retrieval | 4–5 | 5 |
| 5 | ANN (IVF, HNSW) | 7–10 | 7 |
| 6 | Hybrid and rerank | 5–6 | 8 |
| 7 | Generation | 6–8 | 9.5 |
| 8 | Agent loop | 5–7 | 11 |
| 9 | Serving and production | 7–9 | 13 |
| 10 | Frontend, CI, deploy | 5–7 | 14.5 |
| 11 | Framework benchmark and write-up | 4–5 | 15.5 |

That is about **15 weeks part-time**, or 7–8 weeks near full-time, and roughly 4–6k lines of your own code. If time is tight, the **cut line is after Phase 7** (about 9–10 weeks). Phases 8–11 are depth and polish, but Phases 10–11 are what make it presentable, so don't skip them entirely: shrink 8 and 9 instead.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Overfitting to the dev set | Frozen test set, used twice |
| Gold labels break when chunking changes | Source-level gold spans, overlap-based hits |
| Pure-Python HNSW too slow | Numpy batching, then optional `numba`, and write up the finding |
| LLM cost creep from eval runs | Cheapest model as judge, cache LLM responses by prompt hash during experiments |
| Training-data contamination | Version-pinned questions, unanswerable set, retrieval metrics independent of the LLM |
| Scope creep ("add one more feature") | New ideas go to the stretch list; a phase only ends via its done-when |
| Windows quirks (long paths, symlinks) | Short-path venv; avoid features needing symlinks |
| Losing momentum | Every phase ends with something that runs and a logged number |

## 9. Stretch goals (only after Phase 11)
Product quantization · HyDE and query rewriting · contextual chunking (LLM-written chunk blurbs before embedding) · a fine-tuned reranker or embedding model on your eval pairs · multi-corpus support · a web crawler loader · a Postgres-backed index.
