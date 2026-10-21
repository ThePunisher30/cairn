# Phase 4 brief: dense retrieval

**Goal:** search by meaning. Turn every chunk and every question into a vector with an embedding model, and find the chunks whose vectors point the same way as the question. You build the careful parts yourself: the embedder wrapper, the on-disk cache, the brute-force index and the retriever. Then you measure it against BM25 and find out, from your own failures, where each one wins.

**Time:** about 4-5 days at 8-10 hours a week. The code is about 200 lines. The time goes into the first real embedding runs (minutes of CPU each), the experiments, and the failure analysis, which is the part that teaches the most.

**You write the code. 135 tests are already in `tests/` and currently fail** (they were checked against a reference solution and against 48 deliberately broken versions of it, so they are strict but fair). They run in seconds with no download, because they use a tiny offline stand-in model that is provided.

**Done when**
- the full test suite is green and `ruff check .` and `ruff format --check .` are clean
- `python scripts/embed_chunks.py` embeds the corpus with `bge-small`, and running it a second time embeds **nothing** (everything comes from the cache)
- `python scripts/run_eval.py --retriever dense` gives the dense baseline, and `docs/results.md` shows it next to BM25
- `docs/results.md` has the experiment tables (two models, query prefix on and off, every chunker) and the throughput and memory numbers, with your analysis
- `eval/failures/phase4.md` classifies at least 15 failures by cause (Step 8)
- `docs/adr/0003-dense-retrieval.md` records the model, the prefix handling, the cache key and the index choice, and why
- merged through a pull request on `phase-4-dense`

**Constraints:** all the maths is `numpy`. The only library you did not write is `sentence-transformers`, and only as the thing that runs the model: it is wrapped for you in `src/cairn/dense/models.py`. No `faiss`, no `chroma`, no `scikit-learn`.

**Before you start:** the first `import sentence_transformers` after installing can take **two minutes** on Windows (it imports PyTorch). Later imports are fast. The models download once (about 130 MB for `bge-small`, 90 MB for `minilm`); `bge-small` is already on your machine.

---

## The ideas, in plain words

**An embedding is a list of numbers that stands for the meaning of a text.** A model reads a sentence and returns a vector, here 384 numbers. Texts with similar meaning get vectors that point in similar directions, even when they share no words. So "How do I turn off the interactive docs in production?" and "disable OpenAPI with an environment variable" end up close together. That is exactly what BM25 could not do (Phase 3's misses q017, q026, q059).

**Searching is then geometry.** Embed every chunk once and keep the vectors. To answer a question, embed the question, then find the chunk vectors closest to it.

**Cosine similarity, and why the dot product is enough.** "Closest" means the angle between two vectors is small. The cosine of that angle is `a . b / (|a| * |b|)`. If you first scale every vector to length 1 (**L2 normalisation**: divide by its own length), then `|a| = |b| = 1` and the cosine is just the **dot product** `a . b`. A dot product of a whole matrix with one vector is a single fast operation. Tiny example in 2-D: a vector at 0 degrees `(1, 0)` and one at 60 degrees `(0.5, 0.87)` have dot product `0.5 = cos(60 degrees)`; at 90 degrees the dot product is 0; opposite directions give -1. Scores run from -1 to 1, higher is more similar.

**Zero vectors.** A text with no usable content can come out as all zeros, and dividing 0 by its length 0 gives `nan`, which poisons every comparison. Normalisation must leave a zero vector as zero.

**Queries and documents are different.** Many models were trained to match a *short question* to a *long passage*, and were told which is which with a fixed phrase put in front of the question (`bge-small` uses `"Represent this sentence for searching relevant passages: "`). Leave the phrase out and the vectors still come out, only slightly worse, and nothing tells you. So the embedder must know the **query prefix** and the **document prefix** of each model. You will measure what dropping the query prefix costs.

**Truncation.** A model reads at most `max_tokens` tokens (512 for `bge-small`, 256 for `minilm`) and silently ignores the rest. Phase 2 found 76 chunks over 512 tokens, one of 7,124. For those, the model sees only the beginning. The embedder must be able to tell you *how many* texts it cannot fully read, using the model's **real tokenizer** (your Phase 2 `count_tokens` was only an approximation, and this is where that matters).

**Batching and padding, and a 2x speed-up.** The model processes texts in batches, and a batch is padded to the length of its longest text, so one 500-token chunk makes its 15 short neighbours cost as much as long ones. Sort the texts by length first so that short ones travel together, embed in that order, and put the results back in the original order. On this corpus that alone took bge-small from 3.8 to 8.2 chunks per second on a 12-core laptop CPU.

**The cache.** Embedding is the slowest thing in the project (minutes), and you will do it again and again while experimenting. So every vector is stored on disk the first time it is computed and looked up afterwards. The whole question is the **cache key**: it must name *everything that changes the vector* and nothing else.
- the exact text the model reads: your Phase 2 `content_hash` covers only the chunk's text, but what you embed is the heading path plus the text, so two chunks with the same text under different headings are different vectors
- the model (`bge-small` and `minilm` give different vectors for the same text)
- the document prefix (change it and every vector changes)
- *not* the query prefix, because queries are never cached

A key that forgets one of these is a silent bug: it serves old vectors after you change the model or the prefix, and your experiment compares nothing with nothing.

**The file format has no pickle.** In Phase 3 you saved the index with `pickle` and saw that loading a pickle can run code. Here, vectors and keys go into a numpy `.npz` file (plain arrays), loaded with `allow_pickle=False`, so a file from anywhere cannot run code.

**The flat index.** "Flat" means no tricks: store all vectors as one matrix `(n_chunks, 384)` and score every chunk with one matrix-vector product. For 3,267 chunks that takes well under a millisecond. Phase 5 builds approximate indexes for when brute force becomes too slow; first you need the exact answer to compare them against. Picking the best `k` of `n` scores is done with `np.argpartition`, which finds them in about `n` steps without sorting everything.

**A trap hidden in the top-k.** `argpartition` makes no promise about *which* of several equal scores it picks at the cut. If three chunks tie for places 2 to 4 and you want 3 results, a plain `argpartition` may return the higher-numbered ones, and results change from run to run. You need a deterministic rule (equal scores: the lower chunk number first, as in BM25), and a test checks it against a plain full sort on data full of ties.

**Memory.** `n x d x 4` bytes for `float32`. For 100,000 chunks of 384 numbers that is 153.6 MB. You will do this arithmetic in the checkpoint.

**The offline stand-in model.** `HashingModel` (provided) "embeds" a text by counting its words into 256 slots with a hash. Texts that share words get similar vectors. It has no weights and needs no download, so the tests and scripts run instantly, and it doubles as a **floor**: embeddings that only count words, and it is worth one run to see how it does.

---

## Suggested order
```powershell
cd path\to\cairn
git switch main
git pull
git switch -c phase-4-dense
```
Commit after each step. Run `ruff format .`, `ruff check .` and the tests before each commit. (`ruff check` reports import-order warnings in tests until the module they import exists: expected, and don't use `--fix` on tests.)

New files you will create:
```
src/cairn/dense/__init__.py     (empty)
src/cairn/dense/embedder.py
src/cairn/dense/cache.py
src/cairn/dense/flat_index.py
src/cairn/dense/retriever.py
scripts/embed_chunks.py
```
and you extend `scripts/run_eval.py`, `.gitignore` and `pyproject.toml`. **Provided** (already in the repository): `src/cairn/dense/models.py` (the model registry, the real-model wrapper and the offline `HashingModel`), and three scripts, `scripts/ablate_dense.py`, `scripts/bench_embed.py` and `scripts/compare_retrievers.py`.

Do Step 0, then create the package folder with an empty `__init__.py` and commit it, so the tests can import `cairn.dense`.

### Step 0. Settings
- `.gitignore`: add `data/*.npz` (the vectors and the cache are rebuilt from `chunks.jsonl`).
- `pyproject.toml`: add an optional dependency group `dense = ["sentence-transformers>=3"]` under `[project.optional-dependencies]`, next to `dev`. It is optional so the tests and BM25 do not need PyTorch installed. (Your environment already has it; to reinstall use `C:\mrv\Scripts\python -m pip install -e ".[dev,dense]"`.)

Tests: `tests/test_embed_chunks_script.py::test_generated_vector_files_are_not_tracked_by_git` and `::test_the_model_library_is_an_optional_dependency`.

### Step 1. `src/cairn/dense/embedder.py`
- `l2_normalize(x)`: scale every row (or a single vector) to length 1, return `float32`, leave the input untouched, and leave an all-zero row as zero **without a warning** (the test turns warnings into errors). Hint: `np.linalg.norm(x, axis=-1, keepdims=True)` gives the lengths; `np.divide(x, norms, out=..., where=...)` lets you divide only where the length is positive.
- `Embedder(model, name, query_prefix="", doc_prefix="", batch_size=32)`: `ValueError` if `batch_size < 1`. A *model* has `dim`, `max_tokens`, `encode(texts)` and `token_lengths(texts)` (see `models.py`).
  - properties `dim` and `max_tokens` (from the model) and `cache_key`: a string made of the model name and the document prefix (nothing else: not the query prefix, not the batch size)
  - `embed_documents(texts)`: put `doc_prefix` in front of every text; send them to the model **shortest first (by characters), in batches of `batch_size`**; normalise; return an `(n, dim)` `float32` array in the **original order of `texts`**. Equal lengths keep their input order. No texts gives shape `(0, dim)`. Hint: `sorted(range(n), key=...)` gives the order, and `result[order] = vectors` puts rows back.
  - `embed_query(text)`: `query_prefix` in front, returns a `(dim,)` array of length 1
  - `n_truncated(texts)`: how many texts (with the doc prefix) the model's own `token_lengths` says are **longer than** `max_tokens`

Test: `tests/test_embedder.py` (24 tests).

### Step 2. `src/cairn/dense/cache.py`
- `cache_key(model_key, text)`: the SHA-256 hex digest of the model key and the text together (hash a string that cannot be confused: put a separator between them).
- `EmbeddingCache(path=None)`: loads the file if it exists (an empty cache if not). `get(key)` (returns `None` for an unknown key), `put(key, vector)` (keeps **its own copy** as `float32`), `key in cache`, `len(cache)`, `save()` (creates folders; `ValueError` for a cache without a path). The file is a `.npz` with two arrays, `keys` and `vectors`; write it through `open(path, "wb")` and load it with `allow_pickle=False`.
- `embed_cached(embedder, texts, cache) -> (matrix, computed)`: for each text compute its key from `embedder.cache_key`; collect the texts whose key is not in the cache, **each only once**; embed those in one call; store them; return the matrix in the order of `texts` and the number of texts the model actually had to embed.

Test: `tests/test_embedding_cache.py` (19 tests).

### Step 3. `src/cairn/dense/flat_index.py`
- `FlatIndex(vectors, doc_ids)`: stores the vectors as `float32` (do not change the input); `ValueError` if the vectors are not 2-D or there is not one id per row. Properties `n_docs`, `dim`, `nbytes`.
- `search(query, k) -> [(row, score), ...]`: the dot product of every row with the query; the best `k` with `np.argpartition`; best first; **equal scores: the lower row first, even when the tie falls on the edge of the top `k`**. `k <= 0` or an empty index gives `[]`; `k` larger than the index returns every row; a query that is not a vector of length `dim` is a `ValueError`. Rows are plain `int`, scores plain `float`.
- `save(path)` / `FlatIndex.load(path)`: a `.npz` with `version` (the number 1), `vectors` and `doc_ids`; loaded with `allow_pickle=False`; `ValueError` for a file without the right version.
- Hint for the tie rule: after `argpartition` finds the `k` best, find the **smallest score among them**; every row with a score at least that big is a candidate (there can be more than `k`); then order the candidates by score descending and row ascending and keep `k` (`np.lexsort` sorts by its **last** key first).
- **Trap, on purpose:** a Python loop over the rows is far too slow. The speed test (30,000 vectors, 50 searches) fails fast if you write one.

Test: `tests/test_flat_index.py` (39 tests, including 20 random cases checked against a plain full sort).

### Step 4. `src/cairn/dense/retriever.py`
- `build_dense_index(chunks, embedder, cache=None) -> (FlatIndex, computed)`: embed each chunk's `index_text` (heading path plus text, as in BM25) with `embed_cached`, and give the index each chunk's `id`. No cache means a throwaway empty one.
- `DenseRetriever(embedder, index, chunks)`: `ValueError` if the chunks' ids are not exactly the index's `doc_ids`. `search(query, k)` embeds the query (with the query prefix), asks the index, and returns `Hit(path, start_line, end_line, score, text)` best first. `k <= 0` gives `[]`.

Test: `tests/test_dense_retriever.py` (14 tests).

### Step 5. Scripts
**5a. `scripts/embed_chunks.py`** (same shape as `build_index.py`). Options: `--chunks` (default `<data_dir>/chunks.jsonl`), `--model` (a key of `MODELS`, default `bge-small`), `--out` (default `<data_dir>/dense-<model>.npz`), `--cache` (default `<data_dir>/embedding_cache.npz`), `--batch-size` (default 16). A batch size below 1, a missing chunks file (mention `scripts/ingest.py`) or an empty one: one `error: ...` line on stderr, return 1, create nothing. Otherwise: load the embedder with `load_embedder(model, batch_size=...)`, count how many chunks are too long with `n_truncated`, build the index with the cache, save the cache and the index, and print
```
embedded 3267 chunks with bge-small (dim 384): 2802 computed, 465 from cache, 340.6s
83 chunks are longer than the 512-token limit (truncated)
wrote <path>
```
(the tests look for `N computed, M from cache` and `N chunks are longer than the 512-token limit`). Do not be surprised that a brand-new cache already reports "from cache": 465 of the 3,267 chunks are **exact duplicates** of another chunk (the same example code, such as `app = FastAPI()`, appears in many tutorial files), and identical texts are embedded only once. The model's own tokenizer also counts 83 chunks over 512 tokens, not the 76 your Phase 2 regex counter found: this is where the real tokenizer matters. Test: `tests/test_embed_chunks_script.py` (10 tests).

**5b. `scripts/run_eval.py --retriever dense`.** Add `dense` to the choices and the options `--model` (a key of `MODELS`, default `bge-small`), `--dense-index` (default `<data_dir>/dense-<model>.npz`) and `--no-query-prefix` (build the embedder with `use_query_prefix=False`). A missing dense index (mention `scripts/embed_chunks.py`), a missing chunks file, chunks that do not match the index, or a file that is not a dense index: one `error:` line, return 1. The test split stays frozen. Test: `tests/test_run_eval_dense.py` (8 tests).

### Step 6. Your first real dense score
```powershell
C:\mrv\Scripts\python scripts\embed_chunks.py
C:\mrv\Scripts\python scripts\embed_chunks.py          # run it twice: the second time should take no time
C:\mrv\Scripts\python scripts\run_eval.py --retriever dense
```
The first run takes about 5-6 minutes on a CPU (watch the seconds in its output), and the second should print `0 computed`. Record the dense row next to BM25 in `docs/results.md` (date, commit, commands, the per-category table, search latency). Compare it with BM25 in two sentences: which categories moved, which did not?

### Step 7. Experiments, on the dev questions only
```powershell
C:\mrv\Scripts\python scripts\ablate_dense.py --by-category
C:\mrv\Scripts\python scripts\bench_embed.py --model bge-small
C:\mrv\Scripts\python scripts\ablate_dense.py --models bge-small --chunkers all --by-category
```
The first compares `bge-small` and `minilm` on `structure-300`, with `bge-small`'s query prefix on and off. The second measures embedding speed (including what length-sorting buys) and memory. The third embeds the corpus with every chunker (**slow the first time, about 10-15 minutes; everything is cached afterwards**). Put the tables in `docs/results.md` and answer, with numbers:
1. **What does the query prefix buy?** How much does MRR change when you leave it out, and in which categories?
2. **Which model is better, and is the difference bigger than noise?** (With 40 questions one question is worth 0.025.) `minilm` reads only 256 tokens: how many chunks does it truncate (`embed_chunks.py --model minilm` prints it)? Does that show in the results?
3. **Does the chunker matter for dense retrieval the way it did for BM25?** Dense is said to like chunks that are one coherent idea. Is that what you see? Be careful about what the comparison measures: only the `structure-300` chunks carry a heading path in the text that gets embedded (`Chunk.index_text`); fixed and recursive chunks have none. So part of the gap is the heading path, not only where the chunks are cut. How would you separate the two?
4. **What are the speed and the memory?** Chunks per second, seconds for the whole corpus, MB for your corpus and for 100,000 chunks. How much does sorting by length help?
5. **Optional, cheap:** run `run_eval.py --retriever dense --model hashing` (needs `embed_chunks.py --model hashing` first). How does the word-counting floor compare with BM25 and with `bge-small`?

### Step 8. Failure analysis: `eval/failures/phase4.md`
```powershell
C:\mrv\Scripts\python scripts\compare_retrievers.py
```
It prints, for every dev question, the rank at which BM25 and dense first return the answer, a summary (both at rank 1, only BM25, only dense, neither), and then for each question where either one is *not* at rank 1: the question, the gold span and each retriever's top result. Use it as a worksheet. Create `eval/failures/phase4.md` and classify **at least 15** failures (a failure is one retriever, one question, where the answer is not at rank 1; you may mix BM25 and dense). For each: the question id, the retriever, the rank, the **cause**, one line of evidence, and what would fix it. Use these causes (add your own if none fits):
- **wrong chunk**: the right passage exists, but a different chunk about the same topic ranks higher
- **exact-token miss**: the question names an identifier, option or error text that the model blurred into a general topic (typical for dense) or that is split by the tokenizer (typical for BM25)
- **vocabulary mismatch**: the question and the passage mean the same in other words (typical for BM25)
- **too generic**: the question is vague and fits many chunks
- **duplicate chunk**: the same text exists in other files (465 chunks are copies), so identical results crowd the top or the "right" copy is not the one the answer key names
- **answer cut across chunks**: the gold span is split by the chunker (Phase 2)
- **label error**: the gold span itself is wrong or too narrow, i.e. a mistake in your answer key

Naming failure modes precisely is the skill this exercise practises. End the file with a short summary: which causes each retriever suffers from, and which phase fixes which.

### Step 9. ADR, notes and pull request
- `docs/adr/0003-dense-retrieval.md` (Context, Decision, Consequences, Alternatives considered): the model, the prefix handling, the batching, the cache key, the index, the chunks it is used with; what it cannot do; what you would revisit.
- Answer the checkpoint questions below in `docs/notes/phase4.md`.
- Commit, tell me, and I'll check what is about to be published before you push and open the PR.


---

## Reference numbers (look only AFTER you have done Steps 6 to 8)
From the reference solution on the 40 dev questions, `structure-300` chunks, `k = 10`. Yours should be very close; a difference of a question or two can come from tie-breaking, a big one is a bug worth chasing. Times depend on your machine (a 12-core laptop CPU here, no GPU).

**Step 6, the first dense score (`bge-small`, query prefix on), next to BM25:**
```
                    hit@1  hit@5  hit@10  recall@10   ndcg@10    mrr
BM25 overall        0.600  0.750   0.800      0.750     0.658  0.660
dense overall       0.600  0.825   0.925      0.863     0.723  0.711

MRR by category     lexical  semantic  multi_chunk   code
BM25                  1.000     0.432        0.658  0.550
dense                 0.817     0.724        0.700  0.602
```
dense misses at k = 10: q028, q044, q050 (3 questions; BM25 missed 8). Search time: p50 86 ms, p95 101 ms (every question is embedded on the CPU first; BM25 took 2.6 ms). First `embed_chunks.py`: 340 s, `2802 computed, 465 from cache`; the second run: `0 computed, 3267 from cache`. 83 chunks over 512 tokens for `bge-small`, 519 over 256 for `minilm`.

**Step 7, models and the query prefix (`structure-300`):**
```
model      q-prefix   hit@1  hit@10    mrr  ndcg@10   mrr lexical  semantic  multi_chunk   code
bge-small  on         0.600   0.925  0.711    0.723         0.817     0.724        0.700  0.602
bge-small  off        0.450   0.825  0.605    0.634         0.767     0.600        0.533  0.520
minilm     none       0.350   0.800  0.492    0.529         0.526     0.467        0.468  0.508
```
**Dense x chunker (`bge-small`):**
```
chunker         q-prefix  chunks   hit@1  hit@10    mrr  ndcg@10
fixed-200       on          2576   0.325   0.675  0.431    0.453
fixed-200       off         2576   0.325   0.700  0.443    0.470
fixed-400       on          1477   0.250   0.750  0.448    0.496
fixed-400       off         1477   0.250   0.775  0.440    0.494
recursive-400   on          1561   0.300   0.750  0.457    0.501
recursive-400   off         1561   0.300   0.750  0.450    0.492
structure-300   on          3267   0.600   0.925  0.711    0.723
structure-300   off         3267   0.450   0.825  0.605    0.634
```
(The first run with every chunker took about 15 minutes; a second run takes seconds because of the cache.)

**Speed and memory (`bench_embed.py --model bge-small`):**
```
variant              batch  chunks/s
original order          32       4.7
sorted by length         8       9.4
sorted by length        16       9.5
sorted by length        32       7.6
whole corpus (3,267 chunks): about 346 s;  vectors in memory: 5.0 MB;  per 100,000 chunks: 153.6 MB
```

**Step 8, `compare_retrievers.py` (rank 1 of the answer):** both retrievers 17 questions, only BM25 7, only dense 7, neither 9. 23 questions have at least one retriever below rank 1, so the worksheet has more than enough failures for the 15 you need.

---

## Checkpoint questions (answer in your own words in `docs/notes/phase4.md`)
1. Why does the dot product equal the cosine similarity here? What goes wrong if a vector is not normalised, and why must a zero vector stay zero?
2. What did dropping the query prefix cost, and why does a model care about it?
3. What is the memory for 100,000 chunks with 384 numbers each? Show the arithmetic. What would it be for 768 numbers, or for `float16`?
4. Why is the cache key built from the text the model saw plus the model and the document prefix, and not from `chunk.content_hash`? Give a concrete bug the simpler key would cause.
5. Where did dense beat BM25 on your questions, and where did BM25 beat dense? Name one question of each.
6. Why sort texts by length before batching, and why must the output be put back in the original order?
7. A chunk of 7,124 tokens goes into a model that reads 512. What does the model see? What would you do about the 76 chunks like it?

## Hint ladder reminder
Stuck? Tell me which step and paste the failing test. I'll nudge first, then give pseudocode, then a short snippet.
