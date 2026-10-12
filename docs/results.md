# Results

Lab notebook. Every number in the README must trace back to a row here.

## Corpus (FastAPI 0.143.0)

| Part | Location | Files |
|---|---|---|
| Markdown docs | docs/en/docs | 156 |
| Python source | fastapi/ | 52 |

## Baselines on the dev questions

40 dev questions, k = 10. Date: 2026-10-13. Commit: 2e69929.

Candidates are 40-line windows of **every** `.md` and `.py` file under `data/raw/fastapi`: 10,608 windows. That is a bigger and noisier pool than the corpus I actually mean (`docs/en/docs`, `fastapi/` and `docs_src/`): it also contains the translated docs (for example `docs/de/...`) and the repository's own `tests/` folder. Narrowing the corpus is a decision for Phase 2.

| Retriever | hit@1 | hit@5 | hit@10 | recall@1 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|
| oracle | 1.000 | 1.000 | 1.000 | 0.875 | 1.000 | 1.000 | 1.000 |
| first-N | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| random, seed 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| random, seed 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| random, seed 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Oracle, `multi_chunk` questions only: recall@1 = 0.500 (all other categories: 1.000).

Commands: `python scripts/run_eval.py --retriever oracle` (and `first`, `random --seed N`).
The test split was not used.

### What this shows

1. If the oracle returned anything other than 1.000, I would suspect a bug in the loader, the metrics or the harness. It returns exactly the gold spans, so it must score perfectly. It does, so the three agree with each other.
2. Random and first-N both score about 0, so the questions cannot be answered by luck or by position. Random picks 10 windows out of 10,608, which is roughly a 0.1-0.3% chance per question, so guessing does not work. First-N returns the same windows for every question (the alphabetically first files, a German translation page `docs/de/docs/_llm-test.md`), so it measures what you get from ignoring the question completely: nothing.
3. The oracle's recall@1 is 0.875, not 1.0, on purpose. The 10 `multi_chunk` questions each have **two** gold spans, and at k = 1 only one result fits, so recall@1 is 1/2 = 0.500 for them. The other 30 questions have one gold span, so they score 1.0: (30 x 1.0 + 10 x 0.5) / 40 = 0.875.
