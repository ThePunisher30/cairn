# Phase 6 brief: hybrid retrieval and reranking

**Goal:** combine the two retrievers you built and measure what each addition actually buys. BM25 (Phase 3) is good at exact words, dense retrieval (Phase 4) is good at meaning, and they fail on *different* questions: at rank 1 each alone is right on 24 of 40 dev questions, and at least one of them is right on 31. This phase fuses them (**hybrid retrieval**), adds a slow but accurate second look (**a cross-encoder reranker**), builds **small-to-big expansion** (retrieve a small chunk, hand the reader its parent), and ends with the first of the two runs of the frozen test set.

**Time:** about 5-7 days at 8-10 hours a week. The code is about 250 lines and is easier than Phase 5; the thinking is in the experiments, and in being strict with the 40 dev questions.

**You write the code. 110 tests (plus 10 for a provided script) are already in `tests/` and currently fail.** They were checked against a reference solution and against 84 deliberately broken versions of it (the three that the tests cannot tell apart do not change what a user can observe). They run in seconds with no download.

**Done when**
- the full test suite is green and `ruff check .` and `ruff format --check .` are clean
- `python scripts/ablate_hybrid.py --by-category` prints the ablation table on the dev split (the first run downloads the reranker, about 90 MB)
- hybrid + rerank beats the best single retriever on dev hit@5, **or** you have written down, with numbers, why it did not (that is an acceptable result if you can defend it)
- the test split has been run **once**, for the whole table, and the dev-against-test gap is in `docs/results.md`
- `docs/results.md` has the tables and your analysis, `docs/adr/0005-hybrid-retrieval.md` records what you chose and why, `eval/failures/phase6.md` classifies the questions that still fail, and `docs/notes/phase6.md` answers the checkpoint questions in your own words
- merged through a pull request on `phase-6-hybrid`

**Constraints:** `numpy` and the standard library for everything you write. The reranking *model* comes from `sentence-transformers` (as the embedding model did), through the provided `rerank_models.py`. No `langchain`, `llama-index`, `scikit-learn`, `rank_bm25`, and no ready-made fusion or reranking library.

---

## The ideas, in plain words

**Why not just add the scores?** BM25 returns scores like 11.3 or 24.8 (unbounded, and different for every query). Cosine similarity returns numbers between -1 and 1. Add them and BM25 decides everything, whatever the dense retriever thinks. There are two honest ways round it.

### Reciprocal Rank Fusion (RRF): ignore the scores, use the ranks

A chunk at rank `r` (1 = best) in a list earns `1 / (k + r)` points from that list, and the points from all lists are added. A chunk that both retrievers like earns twice; a chunk only one likes earns once. Because only *ranks* are used, the two scales never meet, and nothing needs normalising. `k` (60 in the paper) flattens the curve: with `k = 0` the first place is worth twice the second, with `k = 60` it is worth only 1.6% more, so agreement between lists counts for more than one list's first place.

### Weighted score fusion: put the scores on one scale first

Rescale each list's scores on its own, then take a weighted sum. **Min-max** maps the best score of the list to 1 and the worst to 0. **z-score** subtracts the list's mean and divides by its standard deviation. Both keep the order inside each list and fix the scale, and both have a weakness you can think about (what does min-max do to a list in which one chunk is far better than the rest? what do the "worst of 50" mean, when the list is only the top 50?). The weight says whose opinion counts more. It is a number you can *tune*, which is a danger on 40 questions (below).

### Retrieve, then rerank

The retrievers so far compare the question with a chunk through two *separate* vectors or word counts. A **cross-encoder** reads the question and one chunk *together*, so it can notice that "how do I read a header" is answered by this chunk and not that one. It is much more accurate, and much slower: one model run per (question, chunk) pair, so you cannot run it over 3,267 chunks. The pattern: a cheap retriever finds, say, the top 30, the cross-encoder re-orders only those, and you return the top 5-8. Two consequences you will measure: reranking can **never find a chunk that the first stage missed** (its ceiling is the recall of the first stage at that depth), and it adds latency in proportion to the depth.

### Small-to-big expansion

Small chunks are easy to match precisely, but a reader (the LLM of Phase 7) often needs the paragraphs around the answer. So retrieve small, then hand over the parent: here, the chunks of the same file that share the heading of the one you found (mode `section`), or the whole section one heading up (mode `parent`), growing outward until a token budget is used. Note what this does to the *measuring*: a bigger context overlaps the gold span more easily, so `hit@5` of an expanded context is not comparable with `hit@5` of chunks. The provided script therefore reports the *tokens* next to the hit rate, and the share of the gold lines that the context actually contains.

### The dev set is small, and the test set is spent twice

There are 40 dev questions: **one question is 0.025**, so differences below about 0.05 are noise. Choosing among many settings on 40 questions finds the setting that suits *those* 40 (overfitting). The 20 test questions exist to measure that. They may be run **exactly twice in the whole project**: now, and at the end. Today's run is for the whole ablation table at once, with the configuration already decided on dev. Rules: decide first, write your prediction in `docs/results.md`, run once, record the gap, change nothing afterwards. On 20 questions one question is 0.05, so only read gaps of about 0.1 or more as meaningful, and trust the *order* of the rows more than the digits.

---

## What the provided files do

- `tests/test_fusion.py`, `test_hybrid_retriever.py`, `test_reranker.py`, `test_context_expander.py`: the specification. **Read the docstring at the top of each** before you write the module: it lists the exact names, arguments, tie rules and error cases.
- `src/cairn/retrieval/rerank_models.py`: wraps the cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, 22 million parameters, runs on your CPU), adds a tiny offline `overlap` model for tests, and `load_reranker(key)`, which builds *your* `Reranker` around a model.
- `scripts/ablate_hybrid.py` and `tests/test_hybrid_scripts.py`: the ablation table. It builds BM25, dense, and the hybrids from your modules and prints hit@1/5/10, MRR, nDCG and median latency per row. Options: `--by-category`, `--sweep` (RRF `k` and fusion weight, dev only), `--expansion`, `--no-rerank`, `--rerank-depth`, `--rrf-k`, `--weight`, `--split test --use-test-set`. It refuses the test split without the flag, writes a line to `eval/test_set_runs.log`, and refuses a third run (`run_eval.py` writes to the same log).

---

## Suggested order

```powershell
cd path\to\cairn
git switch main
git pull
git switch -c phase-6-hybrid
```
Commit after each step. Run `ruff format .`, `ruff check .` and the tests before each commit. (`ruff check` reports import-order warnings in the new tests until the module they import exists: expected, and do not use `--fix` on tests.)

New files you will create:
```
src/cairn/retrieval/fusion.py
src/cairn/retrieval/hybrid.py
src/cairn/retrieval/rerank.py
src/cairn/retrieval/context.py
```

### Step 0. Check two labels first
`eval/failures/phase4.md` flagged q005 and q007: the chunk dense put first also seems to answer the question, so the gold may not be the only good passage. Read both questions, the gold span (`python scripts/show_span.py`), and what dense returned, and decide: keep, or add the second passage as another gold span. Rules: decide on the **passage**, not on what would make a retriever look better; write the decision and the reason in `eval/failures/phase4.md`; if you change a label, re-run the BM25 and dense baselines and record the before and after in `docs/results.md` (the Phase 3-4 numbers change). Do this *before* Step 5, and never after the test run.

### Step 1. `fusion.py`
`rrf`, `normalize`, `weighted_fusion`. The docstring of `test_fusion.py` has the exact behaviour, including the tie rule (the key met first wins) and what a constant list becomes. Hints: a dict from key to score; `sorted` with a key that returns a tuple; for the tie rule, record the order in which each key was first seen.

### Step 2. `hybrid.py`
`HybridRetriever`. It asks every retriever for `max(depth, k)` hits, turns each list into keys (the place of a chunk is `(path, start_line, end_line)`), calls your fusion, and builds `Hit`s with the fused score. Before you write it, decide what to do when the same chunk appears in both lists (the test says which hit's text to keep).

### Step 3. `rerank.py`
`Reranker` and `RerankingRetriever`. The `Reranker` is the embedder of Phase 4 again, with a different payload: cut long passages, keep a cache so the same (query, passage) is never scored twice, send the missing pairs to the model in batches sorted by length, put the scores back in order. The key of the cache must change whenever the answer could: what are the three things the score depends on? (Compare ADR 0003.)

### Step 4. `context.py`
`ContextExpander`. Read the docstring of `test_context_expander.py` slowly: the two modes, the growth rule (previous chunk first, then next), and the de-duplication in `expand`.

### Step 5. Run and tune (dev only)
```powershell
C:\mrv\Scripts\python scripts\ablate_hybrid.py --by-category
C:\mrv\Scripts\python scripts\ablate_hybrid.py --sweep
C:\mrv\Scripts\python scripts\ablate_hybrid.py --expansion
```
The reranking rows take a few minutes (30 model runs per question per row). The first run downloads the model.

### Step 6. Experiments (all on the dev split)
1. **Why not add raw scores?** Compare the "raw sum" row with the normalised ones. Then look at the numbers behind it: print the top five BM25 and dense scores for three questions. What is the range of each?
2. **RRF `k` and the weight.** Read the `--sweep` output. Which settings are within noise of the best (0.05)? Which one would you choose, and what does choosing the *highest* cell cost you later? (Think of the test split.)
3. **Fusion against the single retrievers, per category.** Which categories does fusion help, and which does it hurt? Compare with the Phase 4 table of who wins where.
4. **Reranking.** Which row gains most from the reranker, and which category? Which category gets *worse*, and can you say why by looking at the chunks (`scripts/compare_retrievers.py` shows how)?
5. **The depth.** Run hybrid + rerank with `--rerank-depth` 10, 20, 30, 50. What is hit@30 of the hybrid *without* reranking (`--k 30`: the hit@k column then shows hit@30)? That number is the ceiling of what a reranker can reach. What does each depth cost in latency?
6. **Where does the time go?** For one question, time BM25, the question's embedding, the fusion and the reranking separately (the table's p50 is the whole). What would you do to make hybrid + rerank under 300 ms on this CPU?
7. **Small-to-big.** Read the `--expansion` table: what do the two modes and the budgets buy (hit rate, share of the gold lines contained) and what do they cost (tokens)? Why is "expanded hit@5" the wrong number to compare with the chunk table?
8. **Failures.** Take the best configuration at k = 5. Classify every dev question that still has no gold chunk in the top 5 in `eval/failures/phase6.md` (like `phase4.md`: question, category, rank in each retriever, what was returned, what kind of failure). Which of the nine "neither retriever finds it" questions did hybrid + rerank rescue, if any?

### Step 7. The test run (once)
1. Choose the final configuration **from the dev tables**, and write it, together with your prediction of its test numbers (dev numbers, and the gap you expect), in `docs/results.md`, **before** running.
2. Run it once, for the whole table:
```powershell
C:\mrv\Scripts\python scripts\ablate_hybrid.py --split test --use-test-set --by-category
```
3. Record the table and the dev-against-test gap *per row*. Does the order of the rows survive? Which rows gap the most, and is that what you would expect from the way each was chosen? Do **not** change anything after seeing it. One run of the test split is left for Phase 11.

### Step 8. Write-up, merge
- `docs/results.md`, a "Phase 6" section in the style of the others: what you did, the tables, what they say, claims only where the data supports them.
- `docs/adr/0005-hybrid-retrieval.md`: the configuration (fusion method and its parameter, depth, rerank depth, final `k`, expansion mode and budget), the trade-offs (accuracy against latency), the alternatives you measured, what the test run showed.
- `docs/notes/phase6.md`: the checkpoint questions below.
- Commit, tell me, and I'll check what is about to be published before you push and open the PR.

---

## Reference numbers (look only AFTER you have done Steps 5 and 6)

From the reference solution on the **dev** split (40 questions, 3,267 chunks, k = 10, `bge-small`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, rerank depth 30, fetch depth 50), default settings, no tuning. Recall, MRR and the other accuracy numbers do not depend on the machine (two runs gave identical digits), so yours should match closely (a different tie rule could move one question). **Latency does depend on the machine, and mine is rough**: the machine was busy, and the same row varied by up to 1.5 times between two runs.

```
retriever                       hit@1  hit@5 hit@10    mrr ndcg@10   p50 ms
bm25                            0.600  0.750  0.800  0.660   0.658     4-6
dense                           0.600  0.825  0.925  0.711   0.723    43-57
hybrid rrf (k=60)               0.650  0.800  0.900  0.732   0.739    49-106
hybrid weighted minmax (w=0.5)  0.700  0.850  0.925  0.766   0.769    48-71
hybrid weighted zscore (w=0.5)  0.650  0.850  0.925  0.750   0.753    47-90
hybrid raw sum (w=0.5)          0.600  0.800  0.800  0.672   0.668    47-56
bm25 + rerank                   0.700  0.825  0.850  0.758   0.744   2.0-3.5 s
dense + rerank                  0.700  0.850  0.875  0.763   0.765   2.1-3.1 s
hybrid rrf + rerank             0.725  0.875  0.900  0.789   0.789   2.6-3.3 s
```
MRR by category (lexical / semantic / multi_chunk / code): BM25 1.000 / 0.432 / 0.658 / 0.550; dense 0.817 / 0.724 / 0.700 / 0.602; hybrid RRF 0.950 / 0.667 / 0.695 / 0.617; hybrid weighted min-max 0.950 / 0.775 / 0.717 / 0.624; BM25 + rerank 0.950 / 0.800 / 0.733 / 0.548; dense + rerank 0.950 / 0.750 / 0.803 / 0.548; hybrid RRF + rerank 0.950 / 0.850 / 0.808 / 0.548.

The `--sweep` (MRR, hit@5): RRF `k` = 1: 0.761, 0.850; 10: 0.748, 0.850; 30: 0.735, 0.850; 60, 100 and 1000: 0.732, 0.800. Weighted min-max by dense weight: 0.0: 0.660, 0.750; 0.25: 0.705, 0.800; **0.5: 0.766, 0.850**; 0.75: 0.752, 0.850; 1.0: 0.711, 0.825.

The `--expansion` table, on hybrid RRF + rerank, top 5 contexts (hit@5, share of gold lines contained, mean tokens): chunks as retrieved 0.875, 0.802, 1,351; `section` up to 400, 800, 1,600: 0.875, 0.807, 1,415 to 1,496; `parent` up to 400: 0.875, 0.807, 1,745; up to 800: 0.900, 0.832, 2,889; up to 1,600: 0.900, 0.841, 4,002.

What I read in it (check it against yours, and say where you disagree):
- **The raw sum is BM25 in disguise** (MRR 0.672 against 0.660; hit@10 identical): the larger scale decides. Normalising is what makes the sum a fusion.
- **Fusion helps the top of the list more than the recall.** hit@1 rises from 0.600 to 0.650 (RRF) or 0.700 (min-max) and MRR to 0.732 or 0.766, but hit@10 stays at 0.900 to 0.925, no better than dense alone (0.925).
- **Lexical is almost fully recovered** (0.950 against dense 0.817) and `semantic` stays at 0.667 (RRF) to 0.775 (min-max), against 0.724 for dense and 0.432 for BM25: that is what fusion was for. Differences between RRF `k` values (0.732 to 0.761) and between weights 0.5 and 0.75 are within the 0.05 noise; the cell with the best MRR on 40 questions is not safely the best setting.
- **The reranker beats the best single retriever on hit@5**: 0.875 against 0.825 (dense), +0.05, which is two questions. The "done when" condition is met, but only just, so say so with its size.
- **It gains most on `semantic`** (hybrid 0.667 to 0.850 with RRF) and on `multi_chunk`, **and it loses on `code`**: all three reranked rows have code MRR 0.548, below dense (0.602) and hybrid (0.617). I did not find out why; look at the chunks it pushes down (the model was trained on web passages, and a code chunk has few natural-language words).
- **It does not raise hit@10 of the hybrid** (0.900 with RRF, the same with or without; 0.925 for min-max without reranking): it can only re-order the candidates, and with `--k 10` out of 30 it can also push a gold chunk out of the top 10 (dense alone falls from 0.925 to 0.875, BM25 alone rises from 0.800 to 0.850).
- **It costs a lot of time**: about 2 to 3.5 s per question against 0.05 to 0.1 s for the hybrid, 30 model runs of about 70 to 110 ms each, 20 to 70 times the hybrid.
- **Expansion buys little here.** Chunks are already 133 tokens on average; `section` mode adds nothing (0.802 to 0.807), `parent` up to 800 tokens adds 0.030 of gold coverage and one question of hit@5 for 2.1 times the tokens, up to 1,600 adds 0.039 for 3.0 times.

---

## Checkpoint questions (answer in your own words in `docs/notes/phase6.md`)
1. Why can't BM25 scores and cosine similarities be added directly? Show it with numbers from your own data.
2. Why does RRF need no score normalisation? What does `k` do, and what would `k = 0` and `k = 1000` do to the result?
3. What is the weakness of min-max normalisation on a top-50 list? What did z-score change in your table?
4. Why can a reranker never find a chunk that the first stage missed? What does that make the first stage's `depth` responsible for, and what did your depth experiment show?
5. Why is a cross-encoder more accurate than comparing two vectors, and why is it too slow to run over the whole corpus? What did it cost in latency here?
6. Which question category improved most from reranking, and which got worse? Why?
7. What is small-to-big for, and why is `hit@5` of an expanded context not comparable with `hit@5` of chunks? What did expansion cost in tokens for what it gained?
8. What was your dev-against-test gap? Which rows gapped most, and how does that relate to how you chose them? Why is it a mistake to change anything after the test run?

## Hint ladder reminder
Stuck? Tell me which step and paste the failing test. I'll nudge first, then give pseudocode, then a short snippet.
