# ADR 0004: IVF and HNSW indexes, and keeping the flat index for the real corpus

Status: accepted, 2026-10-27. Evidence: `docs/results.md`, section "Phase 5".

## Context

Phase 4 searches by comparing the question with every chunk (flat index). That is exact and, for 3,267 chunks,
takes 0.36 ms. Phase 5 builds the two standard ways of not scanning everything, to find out what they buy and
cost. I wrote k-means, the IVF index and the HNSW index myself with numpy (`src/cairn/ann/`); no `faiss`,
`scikit-learn`, `hnswlib` or `annoy`. The real corpus is too small to show any gain, so the indexes were
measured on a **synthetic** set of 100,000 vectors (each a mix of two real chunk vectors plus noise) and
checked on the real 3,267 vectors.

## Decision

- **The real corpus keeps the flat index.** At 3,267 vectors it is exact and as fast as anything: IVF at
  recall 0.94 is equal to it (2,961 against 2,795 queries/s) and HNSW at recall 0.993 is 3 times slower
  (959 queries/s). The ANN indexes stay in the code as measured, tested alternatives and are not used by the
  retrievers by default.
- **IVF defaults:** `nlist = round(sqrt(n))` (316 for 100,000), `nprobe = 8` (recall 0.995, 3,731 distances;
  `nprobe` 2 gives 0.953 for 1,207). k-means: spherical, k-means++ starts, at most 25 iterations, an empty
  cluster is refilled with the points that fit their centre worst. Searching the `nprobe` closest lists
  with `nprobe = nlist` gives exactly the flat answer, equal scores going to the lower row.
- **HNSW defaults:** `M = 16`, `ef_construction = 100`, `ef_search = 80` (recall 0.993 at 695 distances,
  100,000 vectors), the neighbour-selection heuristic on. Links are capped at `2M` on layer 0 and `M` above;
  the layer of a vector is `floor(-ln(1 - U) / ln(M))`. `ef` is never lower than `k`.
- **File formats:** both are `.npz` files of plain numpy arrays with a `version` number and a `kind`
  (`"ivf"` or `"hnsw"`), loaded with `allow_pickle=False`. IVF stores the vectors, ids, the centres, a row
  order sorted by list and the list start positions. HNSW stores the vectors, ids, the layer of every node,
  the entry point and the links as one flat `edges` array with a `ptr` array of start positions per
  (node, layer). `FlatIndex.load` refuses a file that has a `kind`: before that rule it read an IVF or HNSW
  file as a flat index without an error, because those files also contain `vectors` and `doc_ids`.
- **Cost measure:** `last_n_distances` counts the dot products of the last search (for IVF, the centre scores
  plus the vectors scanned). It does not depend on the machine, unlike queries per second.

## Consequences

- On the scale set, at recall 0.95 IVF computes 1,207 distances and HNSW 355 (flat: 100,000); at 0.99 IVF
  needs 2,156 to 2,908 and HNSW 560. In time the gain is much smaller: HNSW at 0.99 is about 3 to 5 times
  faster than flat (it computes 144 to 179 times fewer distances), since the walk is a Python loop and flat is
  one matrix product.
- **IVF is the cheap one:** 31 s to build against 482 s, 1.29 MB against 11.38 MB extra memory (vectors
  153.6 MB). At recall 0.95 their speeds are not clearly different. HNSW is ahead from about 0.98 up, by
  roughly 1.3 to 3 times.
- **Settings that matter, in order of effect (scale set):** the neighbour-selection heuristic (without it
  recall never reaches 0.99; 0.913 against 0.993 at `ef` 80); `M` 8 versus 16 (0.975 against 0.993 at
  `ef` 80); `efConstruction` 40 versus 100 (0.954 against 0.993). `M` 32 and `efConstruction` 200 cost
  build time (1.9 and 1.4 times) and, for `M` 32, memory, and gained no recall.
- **The upper layers of HNSW:** worth 0.025 recall and 15% of the distances at `ef` 10 at 100,000 vectors, and
  nothing measurable at 5,000 or on the real corpus. They are kept because the data says they pay off with
  scale, not because they helped here.
- **The build is slow:** HNSW takes 4.8 ms per inserted vector in Python (482 s for 100,000, 917 s with
  `M` 32). Real libraries do the same loop in compiled code. Rebuilding is the reason built graphs are
  saved and reused. The HNSW graph is built in row order with one seed: different seeds give different
  graphs, and on the real corpus the seed alone moved recall at `ef` 10 by 0.065.
- **Recall is not the metric that matters, answers are:** on the real dev questions IVF at recall 0.943 keeps
  MRR (0.708 against 0.711) and hit@10 within 0.025; recall 0.495 loses a third of the answers. An ANN index
  must be judged on hit@10 and MRR as well as on recall.
- HNSW never reached recall 1.000 (0.999 at `ef` 160 to 320, `M` 16 and 32): about 2 of 2,000 true neighbours
  are never found. I did not find out why.
- IVF recall on the real vectors is much lower than on the scale set at the same `nprobe` fraction
  (0.495 at `nprobe` 1 against 0.868), so a scale set made from mixes of real vectors is a poor predictor of
  IVF on real embeddings.
- Not done: indexes with fewer bits per number (quantisation), adding or deleting vectors in a built index
  (an index is built once from all vectors; HNSW's `_insert` is internal), and measurements above 100,000
  vectors. All speeds are from one laptop with other
  programs running.

## When to switch (an extrapolation, not a measurement)

A flat scan of 100,000 vectors took about 6.7 ms (150 queries/s); embedding a question takes 38.5 ms. Scan time
is linear in `n`, so the two are equal at roughly 600,000 vectors, and below that the model, not the search, is
the main cost of a query. I would consider an ANN index from several hundred thousand chunks, or sooner with
many simultaneous queries, and measure again at that size.

## Alternatives considered

- **Use HNSW for the real corpus:** 13 s to build, 0.38 MB, and 3 times slower than flat at recall 0.993.
- **IVF with `nlist` 100 or 1,000:** 100 needs 3 to 7 times more distances; 1,000 is better only for recall
  0.99 or more (2,156 against 2,908 distances) and costs 1.8 times the build time.
- **HNSW `M` 8 or 32; `efConstruction` 40 or 200:** measured and rejected above.
- **The simple "M closest" neighbour choice:** builds 2.4 times faster (198 s against 482 s) and is much worse.
- **Skipping the upper layers:** no loss at 5,000 vectors or on the real corpus, but worse at 100,000 vectors
  (recall -0.025 and 15% more distances at `ef` 10).
- **Pickle for index files:** can run code when loaded; plain arrays cannot.
- **Libraries (`faiss`, `hnswlib`, `annoy`, vector databases):** ruled out by the project constraints until
  Phase 11, where the same data can be used to compare against them.
