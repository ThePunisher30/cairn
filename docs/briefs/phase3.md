# Phase 3 brief: BM25 from scratch

**Goal:** build a real keyword search engine from the formula up: a tokenizer that understands code, an inverted index, BM25 scoring with a heap for the top results, a saved index file, and the first real score on your evaluation questions.

**Time:** about 5-6 days at 8-10 hours a week. The code is small (about 250 lines). The time goes into understanding the formula and into the experiments in Step 8.

**You write the code. 106 tests are already in `tests/` and currently fail** (they were checked against a reference solution and against 36 deliberately broken versions of it, so they are strict but fair). The scores in the BM25 tests were worked out by hand: the working is in the test files.

**Done when**
- the full test suite is green and `ruff check .` and `ruff format --check .` are clean
- `python scripts/run_eval.py --retriever bm25` gives you your first real baseline row in `docs/results.md`
- `python scripts/bench_search.py` reports a p95 search time under 20 ms on the full corpus
- the tokenizer-variant x chunker table and the k1 x b grid are in `docs/results.md`, with your analysis (Step 8)
- `docs/adr/0002-bm25-settings.md` records which tokenizer, chunker and k1/b the next phases use, and why
- merged through a pull request on `phase-3-bm25`

**Constraints:** no `rank_bm25`, no `scikit-learn`, no Whoosh. Standard library only (`re`, `math`, `heapq`, `pickle`, `collections`, `dataclasses`).

---

## The ideas, in plain words

**Keyword search.** You type a question. The engine looks at the *words* in it, finds the chunks that contain those words, and ranks them. No meaning, only words. That sounds crude, but it is the backbone of most search engines, it is very fast, and for exact terms (a function name, an error message, a parameter) it beats everything else. Embeddings (Phase 4) will handle the other case: same meaning, different words.

**Step 1 of the idea: count the words (term frequency, "tf").** A chunk that mentions `add_task` five times is probably more about it than one that mentions it once.

**Step 2: rare words matter more (inverse document frequency, "idf").** The word `the` is in nearly every chunk, so finding it tells you nothing. The word `compare_digest` is in two chunks, so finding it tells you a lot. A word's **idf** is high when few chunks contain it and close to zero when almost all do.

**TF-IDF** multiplies the two: `tf * idf`. It has two problems.
1. **Repeating a word forever keeps helping.** Fifty mentions score ten times a few mentions. In reality, after a few mentions you already know the chunk is about it.
2. **Long chunks win by accident.** A long chunk simply contains more words, so more of the query words.

**BM25 fixes both with two knobs:**
- **`k1` (saturation).** The credit for repeating a word flattens out. With `k1 = 1.2`, the first mention is worth 1.0, the second 1.375, and a thousand mentions only about 2.2. A bigger `k1` lets repetition count for longer; `k1 = 0` means "only whether the word is there".
- **`b` (length normalisation).** A chunk longer than average gets its word credit shrunk; a shorter one gets it boosted. `b = 1` is full normalisation, `b = 0` switches it off.

**The formula** (this is what you will implement):

```
score(chunk, query) = sum over each distinct query word t of   idf(t) * tf_weight(t, chunk)

idf(t)        = ln(1 + (N - df + 0.5) / (df + 0.5))
                 N  = number of chunks, df = number of chunks containing t

                          tf * (k1 + 1)
tf_weight  =  ------------------------------------
               tf + k1 * (1 - b + b * dl / avgdl)
                 tf = occurrences of t in the chunk, dl = the chunk's length in tokens,
                 avgdl = the average chunk length
```

**A worked example.** Three tiny chunks: `A = "apple banana apple"` (3 words), `B = "banana cherry"` (2), `C = "apple apple apple dog"` (4). So N = 3 and avgdl = 3. Query: `apple`, with `k1 = 1.2`, `b = 0.75`.
- `apple` is in 2 chunks (A and C): `idf = ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln 1.6 = 0.470`.
- Chunk A: tf = 2, dl = 3, so the length factor is `1 - 0.75 + 0.75 * 3/3 = 1`. `tf_weight = 2 * 2.2 / (2 + 1.2 * 1) = 1.375`. Score `0.470 * 1.375 = 0.646`.
- Chunk C: tf = 3, dl = 4, so the length factor is `0.25 + 0.75 * 4/3 = 1.25`. `tf_weight = 3 * 2.2 / (3 + 1.2 * 1.25) = 1.467`. Score `0.470 * 1.467 = 0.689`.
- Chunk B has no `apple`: not returned.

So C beats A: three mentions beat two, even though C is a little longer. This exact example is in `tests/test_bm25_retriever.py`.

**A term in every chunk.** If `df = N`, then `idf = ln(1 + 0.5 / (N + 0.5))`, which is about `0.5 / N`: tiny but still positive. (The older textbook form without the `1 +` goes *negative* for very common words, which would make a chunk score *worse* for containing them. The `1 +` version avoids that.)

**The inverted index.** To answer a query you must not read every chunk. Instead, build once a dictionary from each word to the chunks that contain it:

```
apple  -> [(chunk 0, 2 times), (chunk 2, 3 times)]
banana -> [(chunk 0, 1),       (chunk 1, 1)]
cherry -> [(chunk 1, 1)]
dog    -> [(chunk 2, 1)]
```
This list is called a **postings list**. A query for `apple` reads two postings, not three chunks. On the real corpus a query touches a few thousand postings instead of 3,267 chunks times their words. You also store each chunk's length (needed for `dl`) and the average.

**Top-k with a heap.** A query might match 2,000 chunks but you want the best 10. Sorting all 2,000 scores costs about `n log n`. A **heap** keeps only the 10 best seen so far and costs about `n log k`. Python's `heapq` module does it. (This is a small saving at our size and a big one at millions of chunks: learn the habit now.)

**Tokenization decides code-search quality.** The question says "background tasks"; the code says `BackgroundTasks` and `add_task`. A tokenizer that only splits on spaces will never connect them. Ours is **code-aware**: it keeps the whole identifier (so an exact search for `get_request_handler` still works) *and* adds its parts (`get`, `request`, `handler`). Whether that helps or hurts is something you will measure in Step 8.

**Why the index remembers its tokenizer.** If you build the index with one tokenizer and then tokenize queries with another, words stop matching and nothing tells you why. So the tokenizer is a small immutable value stored *inside* the index, and the retriever always uses the index's own.

**Why the retriever checks chunk IDs.** An index stores chunk *numbers* ("chunk 17"), not the chunks. If you re-run ingest and the chunks change, number 17 is now a different chunk. The index remembers each chunk's ID (from Phase 2: a hash of place and content), and the retriever refuses chunks that do not match. A loud error beats silently returning the wrong text.

---

## Suggested order
```powershell
cd path\to\cairn
git switch main
git pull
git switch -c phase-3-bm25
```
Commit after each step. Run `ruff format .`, `ruff check .` and the tests before each commit. (`ruff check` reports import-order warnings in tests until the module they import exists: expected, and don't use `--fix` on tests.)

New files you will create:
```
src/cairn/retrieval/__init__.py     (empty)
src/cairn/retrieval/tokenizer.py
src/cairn/retrieval/index.py
src/cairn/retrieval/bm25.py
scripts/build_index.py
```
and you extend `scripts/run_eval.py` and `.gitignore`. Two scripts are **provided**: `scripts/bench_search.py` (search time) and `scripts/ablate_bm25.py` (the experiments of Step 8).

### Step 0. Don't track the generated index
Add `data/bm25.pkl` to `.gitignore` (the index is rebuilt from `chunks.jsonl` any time). Test: `tests/test_build_index_script.py::test_the_generated_index_is_not_tracked_by_git`.

### Step 1. `src/cairn/retrieval/tokenizer.py`
- `Tokenizer`: a **frozen dataclass** with `split_identifiers: bool = True` and `stopwords: frozenset = frozenset()`, and a method `tokenize(self, text) -> list[str]`.
- `ENGLISH_STOPWORDS`: a `frozenset` of at least 20 lowercase English function words (`the`, `a`, `of`, `to`, `how`, `what`, ...). The tests list the ones that must be in it, **and** the words that must not be: Python keywords and API words (`not`, `none`, `true`, `false`, `get`, `set`, `from`, `import`, `class`, `def`) matter in code search.
- The rules are in the docstring of `tests/test_tokenizer.py`. In short: a *word* is a run of letters, digits and underscores (a run of only underscores is not a word); lowercase it; if `split_identifiers` is on, also emit its parts after it (whole word first, then the parts); a word with no parts of its own is emitted once; keep repeats and the text order; remove stopwords from everything emitted.
- Hints: finding the words is one `re.findall`. Finding the *parts* of `HTTPException` is the same job your Phase 2 `count_tokens` regex already does. Compare the parts with the lowercased whole word to know whether to emit them.

Test: `tests/test_tokenizer.py` (24 tests).

### Step 2. `src/cairn/retrieval/index.py`
`InvertedIndex` with:
- attributes `postings` (dict: word -> list of `(chunk number, tf)`, sorted by chunk number), `doc_lengths` (list), `doc_ids` (list of chunk ids), `tokenizer`
- `InvertedIndex.build(chunks, tokenizer=None)` (a classmethod): index `chunk.index_text` (heading path plus text), so section names are searchable. The default tokenizer is `Tokenizer()`. A chunk's length is the number of tokens the tokenizer produced, repeats included.
- `n_docs`, `avg_length` (properties; `0.0` for an empty index), `doc_freq(term)` (the number of chunks containing it, `0` for an unknown word), `postings_for(term)` (the list, `[]` for an unknown word).

Test: `tests/test_inverted_index.py` (16 tests). Trace the toy corpus in its docstring by hand first.

### Step 3. `src/cairn/retrieval/bm25.py`, the formula
- `idf(n_docs, doc_freq)` and `tf_weight(tf, doc_length, avg_length, k1=1.2, b=0.75)`: the two halves of the formula above, as plain functions.

Test: `tests/test_bm25_formula.py` (13 tests). Every expected number has its hand calculation in a comment: redo two of them on paper.

### Step 4. `BM25Retriever` in the same file
- `BM25Retriever(index, chunks, k1=1.2, b=0.75)`: raises `ValueError` for `k1 < 0` or `b` outside `[0, 1]`, and `ValueError` if `[c.id for c in chunks]` is not exactly `index.doc_ids`.
- `search(query, k) -> list[Hit]`: tokenize the query **with the index's tokenizer**; use each *distinct* query word once; for each, walk its postings and add `idf * tf_weight` to that chunk's score (a dict of scores, only for chunks that contain a query word); keep the best `k` with a heap (`heapq.nlargest` is fine, or keep a size-`k` min-heap with `heappush`/`heappushpop`: both are `n log k`); return `Hit(path, start_line, end_line, score, text)` best first. **Equal scores: the lower chunk number first.** `k <= 0` or no matching word gives `[]`.
- **Trap, on purpose:** `index.avg_length` is a *property that sums every chunk length each time it is read*. If you read it inside the loop over postings, every search becomes a thousand times slower. The speed test (`20,000` chunks, `20` searches, under 4 s) fails fast if you do. Compute things once, outside the loop.

Test: `tests/test_bm25_retriever.py` (22 tests).

### Step 5. Save and load the index (in `index.py`)
- `index.save(path)` and `InvertedIndex.load(path)`, using `pickle`. The file must hold a **plain dict** (numbers, strings, lists, tuples, dicts only) with the key `"version"` set to `1`: never pickle your own class, or the file breaks the day you rename it. Store the tokenizer's two settings as plain values and rebuild the `Tokenizer` on load.
- `save` creates missing folders and overwrites. `load` raises `ValueError` for a file that is not a version-1 index file (a wrong version, or not a dict at all).
- **Security note:** unpickling can run arbitrary code. Only load index files you made yourself. (A compact, safe on-disk format is a possible later exercise: delta-encoded postings as variable-length integers.)

Test: `tests/test_index_persistence.py` (8 tests).

### Step 6. Scripts
**6a. `scripts/build_index.py`** (same shape as `scripts/ingest.py`). Options: `--chunks` (default `<data_dir>/chunks.jsonl`), `--out` (default `<data_dir>/bm25.pkl`), `--no-split-identifiers`, `--stopwords` (use `ENGLISH_STOPWORDS`). Missing or empty chunks file: one line `error: ...` on stderr (for the missing file, mention `scripts/ingest.py`), return 1, create nothing. Otherwise build, save, print how many chunks and distinct terms and the mean length, then the output path. Test: `tests/test_build_index_script.py` (7 tests).

**6b. `scripts/run_eval.py --retriever bm25`.** Add `bm25` to the choices and the options `--index` (default `<data_dir>/bm25.pkl`), `--chunks` (default `<data_dir>/chunks.jsonl`), `--k1` (1.2), `--b` (0.75). A missing index file: `error: ...` mentioning `scripts/build_index.py`, return 1. A missing chunks file, a chunks file that does not match the index, or a `k1`/`b` out of range: also one `error:` line and return 1. The test split stays frozen exactly as before. Test: `tests/test_run_eval_bm25.py` (7 tests).

### Step 7. Your first real score
```powershell
C:\mrv\Scripts\python scripts\ingest.py
C:\mrv\Scripts\python scripts\build_index.py
C:\mrv\Scripts\python scripts\run_eval.py --retriever bm25
C:\mrv\Scripts\python scripts\bench_search.py
```
Record in `docs/results.md` (date, commit, commands): the overall and per-category numbers next to the Phase 1 baselines, the index size (terms, file size), and the p50/p95 search time. Then look at the **misses** the report lists: pick three, find each question's gold span (`python scripts\show_span.py ...`), and say *why* BM25 missed it (the question uses different words than the passage? the answer is in a chunk that is too long? something else?). Write one sentence per miss.

### Step 8. Experiments, on the dev questions only
```powershell
C:\mrv\Scripts\python scripts\ablate_bm25.py --by-category
C:\mrv\Scripts\python scripts\ablate_bm25.py --tune structure-300 --tokenizer split
```
The first prints **4 chunkers x 4 tokenizer variants** (`plain`, `split`, `plain+stop`, `split+stop`) with hit@1, hit@10, MRR, nDCG@10 and MRR per category. The second prints a **k1 x b grid** for one configuration. Put both tables in `docs/results.md`, then answer in writing, with numbers:
1. **Does splitting `snake_case` and `camelCase` help?** Overall, and in which question categories? Why does it help (or hurt) those?
2. **Do stopwords help?** Is the difference bigger than noise? (With 40 questions one question is worth 0.025; a change of one or two questions is not evidence.)
3. **Does the chunker matter for BM25, and does it agree with Phase 2's integrity numbers?**
4. **What do `k1` and `b` do on your data?** Look at the `b = 0` column and at `k1 = 0`. Are the defaults far from the best cell? If the best cell is only 0.01 better, would you change the defaults on 40 questions?
5. **Which configuration do Phases 4-6 use as the keyword baseline?**

You tune on the dev questions only. The test split stays frozen until after Phase 6.

### Step 9. ADR, notes and pull request
- `docs/adr/0002-bm25-settings.md` (Context, Decision, Consequences, Alternatives considered): the tokenizer variant, chunker, `k1` and `b` that Phases 4-6 will use, and why; what BM25 cannot do (see your misses); what you would revisit.
- Answer the checkpoint questions below in `docs/notes/phase3.md`.
- Commit, tell me, and I'll check what is about to be published before you push and open the PR.

---

## Reference numbers (look only AFTER you have done Steps 7 and 8)
From the reference solution on the 40 dev questions, `structure-300` chunks, `split` tokenizer, `k1 = 1.2`, `b = 0.75`. Yours should be very close; a difference of a question or two can come from tokenizer details, a big one is a bug worth chasing.

```
                n    hit@1  hit@5  hit@10  recall@10   ndcg@10    mrr
overall        40    0.600  0.750   0.800      0.750     0.658  0.660
lexical        10    1.000  1.000   1.000      1.000     1.000  1.000
semantic       10    0.300  0.600   0.800      0.800     0.518  0.432
multi_chunk    10    0.600  0.800   0.800      0.600     0.551  0.658
code           10    0.500  0.600   0.600      0.600     0.563  0.550
misses (8): q017, q026, q031, q041, q043, q044, q058, q059
index: 3,267 chunks, 7,184 distinct terms, mean length 86.9 tokens, pickle file about 1.1 MB
search time: p50 about 2.7 ms, p95 about 4 ms
```

Tokenizer variants x chunkers (MRR and hit@10 shown; run the script for the rest):
```
chunker         tokenizer    terms   hit@10    mrr
fixed-200       plain         6606    0.675   0.459
fixed-200       split         7184    0.725   0.514
fixed-400       plain         6606    0.800   0.502
fixed-400       split         7184    0.850   0.597
recursive-400   plain         6606    0.725   0.467
recursive-400   split         7184    0.750   0.562
structure-300   plain         6606    0.750   0.552
structure-300   split         7184    0.800   0.660
structure-300   split+stop    7151    0.825   0.654
```

k1 x b grid, MRR, `structure-300 + split`:
```
          b=0    b=0.25   b=0.5   b=0.75   b=1
k1=0     0.314    0.314   0.314    0.314   0.314
k1=0.6   0.331    0.577   0.628    0.633   0.638
k1=0.9   0.329    0.604   0.644    0.654   0.637
k1=1.2   0.320    0.603   0.649    0.660   0.619
k1=1.5   0.280    0.592   0.642    0.651   0.558
k1=2     0.280    0.590   0.633    0.633   0.518
k1=3     0.242    0.586   0.611    0.584   0.465
```

---

## Checkpoint questions (answer in your own words in `docs/notes/phase3.md`)
1. What does `b = 0` do? What happened to your score when you set it to 0, and why do you think it hurt so much on this corpus? (Think about your biggest chunks from Phase 2.)
2. Why does a word that is in every chunk contribute almost nothing? Why does the formula use `ln(1 + ...)` instead of the textbook `ln(...)`?
3. Why did splitting `camelCase` and `snake_case` help or hurt on your data? Give one question where it made the difference.
4. Why keep a heap of size `k` instead of sorting all the scores? What is the cost of building the index, and what is the cost of one search?
5. Look at three misses. What can keyword search never do, and which phase fixes it?
6. Why does the index store its tokenizer, and why does the retriever check chunk IDs? What bug does each prevent?

## Hint ladder reminder
Stuck? Tell me which step and paste the failing test. I'll nudge first, then give pseudocode, then a short snippet.
