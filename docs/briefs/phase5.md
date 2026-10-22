# Phase 5 brief: approximate nearest-neighbour indexes

**Goal:** understand why vector databases exist by building the index inside them. In Phase 4 you found the nearest chunks by comparing the question with *every* chunk (the flat index). That is exact, and it is fast enough for 3,267 chunks. It stops being fast enough somewhere around a million. This phase builds the two classic ways to avoid the full scan, **IVF** (cluster the vectors, search only a few clusters) and **HNSW** (link each vector to its neighbours, then walk the links), and measures what they buy and what they cost.

**Time:** about 7-10 days at 8-10 hours a week. This is the hardest phase. The code is about 350 lines, but HNSW takes real thinking, and the benchmark runs take minutes to an hour on a CPU.

**You write the code. 91 tests are already in `tests/` and currently fail** (checked against a reference solution and against 55 deliberately broken versions of it; the five that the tests cannot tell apart are listed in the reference-numbers section). They run in about a minute with no download.

**Done when**
- the full test suite is green and `ruff check .` and `ruff format --check .` are clean
- `python scripts/make_scale_set.py` has made a 100,000-vector set, and `python bench/ann.py --scale data/scale-100000.npz` prints recall, speed and memory for flat, IVF and HNSW (this is provided; your code makes it work)
- HNSW reaches recall@10 of at least 0.95 at some `ef`, and at 100,000 vectors it answers more queries per second than the flat index at that recall (if it does not, that is a finding to explain, not a failure)
- `docs/results.md` has the recall-against-speed tables, the build times and memory, the experiments below, and your analysis
- `docs/adr/0004-ann-indexes.md` records what you chose and why, and `docs/notes/phase5.md` answers the checkpoint questions in your own words
- merged through a pull request on `phase-5-ann`

**Constraints:** `numpy` only for the maths, and Python's `heapq` and `math`. No `faiss`, no `scikit-learn` (so you write your own k-means), no `hnswlib`, no `annoy`. `matplotlib` is allowed, only for the picture, and is optional.

---

## The ideas, in plain words

**Why not always scan everything?** One search of the flat index multiplies the query by an `n x 384` matrix. The cost grows in step with `n`: 3,267 chunks take well under a millisecond, 100,000 take a few milliseconds, 100 million take seconds, and the matrix no longer fits in memory (100 million x 384 x 4 bytes is 154 GB). If you give up the guarantee of finding the *exact* nearest vectors and accept "almost always the right ones", you can touch a tiny fraction of the data instead. That trade is **approximate nearest-neighbour (ANN)** search.

**Recall is how you score the trade.** For one query, take the exact top 10 from the flat index. `recall@10` is the share of those 10 that the approximate index also returned: 0.9 means it found 9 of the 10. Every ANN result is a pair: recall and speed. There is no single "best" index; there is a **curve**, and a better index gives a better curve (more speed at the same recall).

**There is no free perfect index in high dimensions.** In 2 or 3 dimensions you can cut space into boxes and find exact neighbours quickly (a k-d tree). In 384 dimensions almost every vector is about equally far from almost every other, boxes stop excluding anything, and the exact method degrades to a full scan. That is the "curse of dimensionality", and it is why approximate methods exist. You will see its effect on your own numbers.

### IVF: cluster first, search a few clusters

1. **k-means** cuts the vectors into `nlist` groups. Start with `nlist` random vectors as centres; give every vector to its nearest centre; move each centre to the average of its vectors; repeat until nothing moves. (Here the vectors have length 1 and "near" means a large dot product, so each centre is the average direction scaled back to length 1: *spherical* k-means.)
2. **k-means++** starts better. Pick the first centre at random, then pick each next centre at random but with probability proportional to its squared distance from the nearest centre already chosen, so new centres tend to land far from old ones. Random starts often put two centres in one real group and leave another group with none.
3. **Empty groups.** A centre can lose all its vectors. Decide what to do: moving it to the vector that fits its current centre worst is a common choice.
4. **The index** is just: the `nlist` centres, and for each centre the list of rows that belong to it (an "inverted file": centre to list of rows).
5. **A search** scores the query against the `nlist` centres, takes the best `nprobe` of them, and scans only the vectors in those lists. Cost: `nlist` + (about `nprobe / nlist` of the data). With `nlist` near the square root of `n`, that is about `2 sqrt(n)` per probe. The miss: the true nearest vector sits just across a cluster boundary, in a list you did not probe. `nprobe` is the knob: more lists, more recall, less speed. With `nprobe = nlist` you scan everything and get the exact answer, which the tests use as a check.

### HNSW: a graph you walk

1. **Idea.** Link every vector to some of its near neighbours. To search, start at some vector and keep moving to whichever linked vector is closest to the query; stop when no neighbour is better. The walk reaches the neighbourhood after a few dozen steps, touching a tiny part of the data.
2. **A plain graph is not enough.** A single walk can get stuck in a local dead end (a vector whose neighbours are all worse than it but that is not near the query), and long distances take many small steps. So HNSW keeps a **beam**: instead of following only the best candidate, it keeps the `ef` best candidates found so far and keeps expanding the best unexplored one until none can improve the list.
3. **Layers.** To make the early steps big, some vectors are also placed in sparser upper layers. Each vector's top layer is drawn at random: `floor(-ln(U) / ln(M))` for `U` uniform in (0, 1], so about `1/M` of the vectors reach layer 1, `1/M^2` reach layer 2, and so on. A search starts at the single entry vector on the top layer, walks greedily (a beam of 1) down each upper layer, and uses the result as the starting point of the next layer down. At layer 0, where every vector lives, it runs the beam of size `ef`.
4. **Insertion** builds the graph one vector at a time, in row order. For a new vector with top layer `L`: walk down from the entry vector to layer `L + 1` greedily; then for each layer from `L` down to 0, find its `ef_construction` nearest vectors in that layer (the same beam search), choose `M` of them as its neighbours, and link both ways. A neighbour that now has too many links (more than `M`, or `2M` on layer 0) must drop some. If the new vector's top layer is above the current top, it becomes the entry vector.
5. **Which neighbours to keep.** The simple choice is the `M` closest. That fails on clustered data: the `M` closest are all in the same tight cluster, the cluster links only to itself, and nothing leads to other clusters. The **heuristic** (from the HNSW paper) goes through the candidates from closest to farthest and keeps a candidate only if it is closer to the new vector than to every neighbour already kept; skipped candidates fill up any spare slots at the end. The kept neighbours point in *different directions*. You will measure the difference on clustered data.
6. **The knobs.** `M` (links per vector): more links mean better recall and more memory. `ef_construction`: a wider beam while building, a better graph and a slower build. `ef_search`: a wider beam while searching, more recall and less speed. `ef` is never below `k`.
7. **Pure Python is slow to build.** Every insertion runs a beam search that computes hundreds of dot products. You batch the dot products of a node's neighbours into one numpy call (`vectors[neighbours] @ query`), but the loop around it is Python. Expect minutes for 20,000 vectors and a long run for 100,000. That slowness is a finding in itself (real libraries do this in C++).

### The memory side

HNSW stores about `M` links (`2M` on layer 0) for every vector. With `M = 16`, 32 links x 4 bytes = 128 bytes per vector for the graph, on top of the 1,536 bytes of the vector itself. IVF stores one row number per vector (8 bytes with `int64`) and the centres. You will print both.

### Why a scale set

Your real corpus has 3,267 chunks, where the flat index needs under a millisecond and no index can win. To give ANN something to prove, `scripts/make_scale_set.py` (provided) makes 100,000 vectors from your real ones: each is a mix of two real vectors plus some noise, scaled to length 1, and so are the queries. It keeps the shape of real embeddings (384 numbers, unit length, similar neighbourhoods) but it is **synthetic**: say so whenever you quote its numbers, and run the real-corpus check at the end as well.

---

## What the provided files do

- `tests/test_kmeans.py`, `test_ivf_index.py`, `test_hnsw_index.py`: the specification. Read the docstring at the top of each: it lists the exact names and behaviours.
- `scripts/make_scale_set.py`: makes `data/scale-100000.npz` (the vectors and 1,000 queries).
- `bench/ann.py`: builds (or loads) the indexes, and for each setting prints recall@10, queries per second, distance computations per query, build time and memory. It saves built indexes under `data/ann/` so a rerun is fast. `--plot` draws the recall-against-speed picture (needs `pip install matplotlib`).
- `tests/test_ann_scripts.py`: tests of those two scripts.

---

## Suggested order

```powershell
cd path\to\cairn
git switch main
git pull
git switch -c phase-5-ann
```
Commit after each step. Run `ruff format .`, `ruff check .` and the tests before each commit. (`ruff check` reports import-order warnings in tests until the module they import exists: expected, and don't use `--fix` on tests.)

New files you will create:
```
src/cairn/ann/__init__.py     (empty)
src/cairn/ann/kmeans.py
src/cairn/ann/ivf.py
src/cairn/ann/hnsw.py
```
and you extend `.gitignore` and `pyproject.toml`. **Provided** (already in the repository): `scripts/make_scale_set.py`, `bench/ann.py`, and the tests.

### Step 0. Settings
- `.gitignore`: add `data/ann/` (the built indexes; `data/*.npz` already covers the scale set).
- `pyproject.toml`: add `plot = ["matplotlib>=3.8"]` under `[project.optional-dependencies]`.
- Create the package folder with an empty `__init__.py`, and commit it.

### Step 1. `kmeans.py`
`kmeans(vectors, k, *, seed=0, max_iter=25) -> (centroids, labels)`: spherical k-means with k-means++ starts, as described above. Details the tests check:
- the centroids come back as `float32`, shape `(k, d)`, each of length 1; `labels[i]` is the centroid with the largest dot product with vector `i`, **for the centroids you return** (so do a last assignment after the loop)
- `k` must be between 1 and `n` (`ValueError` otherwise); the input array is not modified
- the same `seed` gives the same answer (`np.random.default_rng(seed)`)
- a centroid that loses all its vectors does not become a zero vector
- do the work with matrix products on blocks of vectors, not Python loops: 20,000 vectors must cluster in seconds

Hints: for unit vectors, squared distance = `2 - 2 * dot`. To average the vectors of each cluster without a loop over clusters, look at `np.add.at`, or sort by label and use `np.add.reduceat`.

### Step 2. `ivf.py`
`IVFIndex(vectors, doc_ids, nlist, *, nprobe=8, seed=0, max_iter=25)` trains k-means and fills the lists. The tests' docstring lists the names: `search(query, k, nprobe=None)`, `list_rows(j)`, `last_n_distances` (the centroid scores plus the vectors scanned: this is the machine-independent cost), `extra_nbytes`, `save`, `load`. Rules:
- same score and tie rule as `FlatIndex` (best first; equal scores, lower row first), so with `nprobe = nlist` the answer is **exactly** the flat index's
- a search that finds fewer than `k` vectors in the probed lists returns fewer than `k` results
- a per-call `nprobe` does not change the index's own `nprobe`
- the file is plain numpy arrays with a version number and no pickle, like the flat index

It is built to be a drop-in for `FlatIndex` in `DenseRetriever`: it has `doc_ids` and `search(query, k)`.

### Step 3. `hnsw.py`
`HNSWIndex(vectors, doc_ids, *, M=16, ef_construction=100, ef_search=50, seed=0, heuristic=True)` builds the graph in the constructor by inserting row 0, 1, 2, ... in order. Build it in stages, running the tests as you go:
- **3a. Layers.** Draw every vector's top layer from the seed, create empty link lists, set the entry vector. (`levels`, `entry_point`, `max_level`.)
- **3b. The beam search on one layer.** `search_layer(query, entry_points, ef, layer)`: two heaps, one of candidates to expand (best first), one of the best `ef` found so far (worst on top so you can throw it out), and a `visited` set. Stop when the best unexpanded candidate is worse than the worst of the `ef` kept. Count every dot product in `last_n_distances`.
- **3c. Insertion with the simple choice** (the `M` closest), links both ways, shrinking a neighbour that has too many links (cap `2M` on layer 0, `M` above). With the simple choice the graph is valid and a search with a huge `ef` finds everything, so the structure tests and the `heuristic=False` test pass; the recall tests on the default settings need step 3e.
- **3d. `search`.** Greedy through the upper layers, beam of `ef` (raised to `k` if smaller) on layer 0, return the best `k` sorted best first, equal scores lower row first.
- **3e. The heuristic selection**, then `heuristic=True` as the default.
- **3f. `save` / `load`**: design the format yourself (a good one: the links of all nodes as one flat integer array plus an array of start positions, like the lists of IVF), plain arrays, version number, no pickle.
- `neighbors(row, layer)`, `n_links` and `extra_nbytes` (4 bytes per link plus a little per node) are listed in the tests' docstring.

Things that bite:
- **Do not compute one dot product per neighbour in a Python loop.** Gather the unvisited neighbours of a node, then `self.vectors[neighbours] @ query` once.
- **The two heaps are on opposite sides.** `heapq` is a min-heap: for "best first" push the negated similarity.
- **Ties in heaps compare the next tuple element**: push `(-similarity, row)` so ties are decided by the row number and the build is deterministic.
- All-identical vectors are a weak spot of HNSW (nothing separates their links). The test for them only asks for a clean, valid answer.

### Step 4. Run it
```powershell
C:\mrv\Scripts\python scripts\make_scale_set.py
C:\mrv\Scripts\python bench\ann.py --scale data\scale-100000.npz --skip hnsw
```
Start with IVF only; it builds in under a minute. Then try HNSW on a smaller set to find out how long a build takes on your machine before committing to 100,000:
```powershell
C:\mrv\Scripts\python scripts\make_scale_set.py --n 20000 --out data\scale-20000.npz
C:\mrv\Scripts\python bench\ann.py --scale data\scale-20000.npz
```
Then the full set (leave it running; it saves the built index):
```powershell
C:\mrv\Scripts\python bench\ann.py --scale data\scale-100000.npz --plot data\ann-100000.png
```
If the 100,000 build is more than you want to wait for, run it at 50,000 and say so in the write-up.

### Step 5. Experiments (all on the scale set unless noted)
1. **IVF: `nlist` against `nprobe`.** Try `nlist` of 100, 316 (about the square root of n) and 1,000. For each, find the `nprobe` that gives recall@10 of 0.95 and 0.99, and write down the queries per second and the distance computations at that point. How does the right `nprobe` move with `nlist`?
2. **HNSW: `M` and `ef_search`.** Build `M = 8` and `M = 16` (and 32 if you have the patience) with `ef_construction` 100. Find the `ef_search` for recall 0.95 and 0.99. Build time and memory for each.
3. **HNSW: `ef_construction`.** Build with `ef_construction` 40 and 200 at `M = 16`. What does a better graph cost and what does it buy?
4. **The heuristic.** Build with `heuristic=False` on the same data. The tests show it matters on clustered data; how much does it matter on this set?
5. **The curve.** One picture (`--plot`) with flat, IVF and HNSW, recall on the x axis and queries per second on the y axis. Say which index you would pick for a target recall of 0.95, and why.
6. **The real corpus.** `python bench\ann.py --real` embeds the 40 dev questions and compares the indexes on your 3,267 real chunk vectors. Does ANN help there? What is the best the flat index can do, and why is the answer to "does your corpus need ANN?" what it is?
7. **Does the approximation change the answers that matter?** On the real corpus, the metric that matters is not recall@10 against the flat index but the retrieval metrics against the gold passages. Take the real-corpus index of your choice and compare its hit@10 and MRR with the flat index's 0.925 and 0.711 (you can plug an `IVFIndex` or `HNSWIndex` built from `flat.vectors` into `DenseRetriever`, as the tests do, and run the harness).
8. **The layers.** Temporarily make your search skip the upper layers (start the layer-0 beam at the entry vector) and compare recall and distance computations at the same `ef`, on the scale set and on a small set (a few thousand vectors). The tests cannot tell the two apart, so this is yours to find out: where do the layers start to matter, and what does that say about answering "what would a search be like without them"?

### Step 6. Write-up, merge
- Put the tables and your analysis in `docs/results.md`, a "Phase 5" section in the style of the others: what you did, the numbers, and what they say, claims only where the data supports them. Say clearly that the scale set is synthetic.
- Write `docs/adr/0004-ann-indexes.md`: the defaults you would use (`nlist`, `nprobe`, `M`, `ef_construction`, `ef_search`), the file formats, the trade-offs, the alternatives you measured.
- Answer the checkpoint questions in `docs/notes/phase5.md`.
- Commit, tell me, and I'll check what is about to be published before you push and open the PR.

---

## Reference numbers (look only AFTER you have done Steps 4 and 5)

From the reference solution on the 100,000-vector scale set (made with the default settings of `make_scale_set.py`), 200 queries, k = 10, a 12-core laptop CPU, no GPU, **while other programs were running**. Recall and the number of distance computations do not depend on the machine, so yours should be very close (a seed or tie-breaking can move recall by a point or two). Queries per second and build times do depend on it, and mine are rough: some rows are not even in order. Use them for the shape of the curve, not for the digits.

**The exact answer (flat index):** 100,000 distance computations per query, about 147 queries/s.

**IVF, `nlist = 316` (the square root of n), build 38 s:**
```
nprobe   recall@10   queries/s   distances per query
     1       0.868        1163          831
     2       0.953         539        1,592
     4       0.981         265        3,193
     8       0.995         194        4,689
    16       0.998          97        9,394
    32       0.999          56       15,109
    64       1.000          27       26,718
```

**HNSW, `M = 16`, `ef_construction = 100`, build 1,511 s (about 25 minutes, in pure Python):**
```
ef_search   recall@10   queries/s   distances per query
       10       0.921         996          199
       20       0.943         466          221
       40       0.973         292          447
       80       0.993         306          833
      160       0.999         175        1,324
      320       0.999          87        1,827
```
2,645,462 links, top layer reached in this build: 4. Memory on top of the 153.6 MB of vectors: IVF about 1.3 MB (centres, `int64` row numbers, offsets), HNSW about 11.4 MB (`4 x n_links + 8 x n`).

What to compare, in numbers that do not depend on the machine: for recall of about 0.95 to 0.98, flat computes 100,000 distances, IVF 1,592 to 3,193 and HNSW 221 to 447, so HNSW looks at 220 to 450 times fewer vectors than flat. In time the gap is much smaller (HNSW about 2 times the flat index's queries per second here, IVF 2 to 4 times) because the walk is a Python loop, not one matrix product. Building costs the other way round: IVF 38 s, HNSW 1,511 s.

**A smaller set, 20,000 vectors, for planning your time** (`--n 20000`, `nlist = 141`): flat 496 queries/s; IVF reached recall 0.911 at `nprobe = 4` and 0.958 at `nprobe = 16`; HNSW built in 286 s and reached recall 0.982 at `ef = 40`.

**The heuristic and the layers, on clustered data** (the test fixture: 1,500 vectors of 32 numbers in 25 clusters, `M = 8`, `ef_construction = 60`, 60 queries, three seeds): recall@10 at `ef = 10` is 0.98 to 0.99 with the heuristic and 0.69 to 0.80 with the simple choice of the `M` closest, and at `ef = 50` it is 1.00 against 0.84 to 0.87. Searching everything (`ef = n`) finds everything in both. Taking the upper layers away from a search changed almost nothing at that size: recall 0.992 against 0.973 at `ef = 10`, and 99 against 104 distance computations (on 3,000 points on a circle: 761 against 729). The layers pay off at much larger sizes, and **the tests cannot tell a search with layers from one without**, so measure it yourself (Step 5, experiment 8).

**What the tests cannot see.** Of the 55 deliberately broken versions of the reference, 50 are caught. The other five do not change what a user can observe: a k-means that does not guard against a total distance of exactly zero (floating-point noise makes that case unreachable), one that refills an empty cluster with a different point, one that accepts empty input (k-means rejects it anyway through the check on `k`), an `nlist` that is not validated in `IVFIndex` (k-means rejects it anyway), and a search that skips the upper layers (see above). Several others survived my first tests and led to new ones: k-means++ was not distinguished from random starts, stale labels were returned when k-means stopped early, empty clusters became zero vectors, a missing version check on loading, and upper layers without links.

**Not in the reference numbers:** the real-corpus check (Step 5, experiments 6 and 7). I did not run it; those numbers will be yours.

---

## Checkpoint questions (answer in your own words in `docs/notes/phase5.md`)
1. Why can't you build a perfect index for nearest-neighbour search in high dimensions? What does your own data say about it?
2. In IVF, what does raising `nprobe` do to recall, speed and memory? What does raising `nlist` do? Why is about the square root of `n` a sensible `nlist`?
3. What does k-means++ do that random starting centres do not, and what goes wrong with an empty cluster?
4. In HNSW, what does raising `efSearch` do? What does raising `M` do? What does raising `efConstruction` do, and who pays for it?
5. Why are there layers? What would a search be like without the upper layers?
6. Why does choosing "the `M` closest" as neighbours go wrong on clustered data, and what does the heuristic change?
7. Why does your corpus not need ANN? At what size would you start to need it, from your own numbers?
8. What did your pure-Python HNSW build cost, and what does that tell you about why real libraries are written in C++?

## Hint ladder reminder
Stuck? Tell me which step and paste the failing test. I'll nudge first, then give pseudocode, then a short snippet.
