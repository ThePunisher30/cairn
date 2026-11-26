# ADR 0010: Comparing against LlamaIndex and FAISS with the protocol written first, the dev rows chosen by a stated rule, and the last test run behind a committed freeze

Status: accepted, 2026-11-26. Evidence: `docs/results.md`, section "Phase 11"; `docs/briefs/phase11.md`; `bench/frameworks/`.
The comparison is one framework (LlamaIndex) with one vector library (FAISS), one embedding model, one machine, and a test split of 20 questions, so the claims are narrow. Section "What I would not trust" says where.

## Context

Phase 11 asks what a framework does by default, what that costs or gains on the same questions, and whether each gap can be explained with something built earlier. The test split could be run exactly twice in the project; the first run was spent in Phase 6, so this is the last one, and it cannot be repeated.

**Who wrote it.** The AI assistant wrote the code, the protocol and the write-ups at the author's request.

## Decision

- **The protocol and the predictions were committed before any framework number existed** (`docs/briefs/phase11.md`): what is compared, what is held equal, eight numbered predictions, and the rules for tuning and for the test run. The results score each prediction as right or wrong (4 right, 3 wrong, 1 mostly wrong).
- **Held equal:** the corpus files, the embedding model with its query instruction, k = 10, the harness and its hit rule. **Not equal, on purpose:** the reader, the chunker, the retrieval code.
- **An adapter, not a rewrite of the metric.** A framework node carries a character range in the text it read; `cairn.evaluation.adapters` turns it into a file and a line range, so a framework row is scored by exactly the same code as a Cairn chunk. A node without offsets is an error unless the text is located in the document (counted and reported), never a guess.
- **One process per row** in one shared environment, so each row's peak memory is its own and every row runs the same library versions. Each row is built cold (no embedding cache).
- **The LlamaIndex tuning candidates were listed before they were run** (`candidates.json`, eight of them) and chosen on dev by a stated rule (highest hit@5, ties by MRR). Every candidate is reported, not only the winner.
- **Diagnostic rows are allowed after seeing results, and labelled** (the inner-product fusion rows, the line-ending control). They are not candidates and are not chosen by search.
- **A freeze before the test run, enforced by code.** `bench/frameworks/frozen.json` lists the nine rows with their dev numbers and is committed first. `run_test.py` refuses unless the rows requested are exactly the frozen ones, the file is committed and unchanged, and `eval/test_set_runs.log` shows one earlier run (`cairn.evaluation.freeze`, tested). It writes the log line before running, so a crash cannot give a free retry, and `run_row.py` refuses the test split unless `run_test.py` has allowed it. `--check-only` reports whether the run would be allowed without running anything.
- **The test run was run once, and the results are reported whatever they were.** The log now shows both allowed runs.
- **Paired statistics are in the repository** (`paired.py`): the questions where only one row hit, an exact sign test, and a bootstrap interval, so a difference of one question is shown as one question.

## Why

- **Writing predictions first** makes the comparison falsifiable: 3 of 8 were wrong and 1 mostly wrong, including the cause I gave for the default's weakness (it keeps every gold span whole; its chunks are too big for the model, which is a different problem).
- **Listing candidates first and reporting all of them** limits the damage of choosing the best of eight on 40 questions. It did not remove it: the selected row lost 0.125 hit@5 on test.
- **A committed freeze with a refusing runner** turns "I will not tune on the test set" from a promise into a check that fails.
- **Character offsets, not file-level matching,** because the harness's hit rule is about lines and a file-level match would hide the difference between finding the file and finding the passage (the default finds the right file for 67.5% of questions and the right lines for 52.5%).

## Trade-offs

- **One framework and one vector library** is cheap and clean, and it means "the frameworks" is not established. LangChain, Chroma and pgvector were not tried.
- **The hit rule favours big chunks;** each row reports its chunk size and the share of chunks over the model's window to make that visible, and the integrity measure shows the default keeps every gold span whole.
- **Holding the embedding model equal** isolates the pipeline from the model and says nothing about a framework's other default models.

## What I would not trust

- **Any claim that Cairn beats LlamaIndex.** On dev the default LlamaIndex pipeline was 0.30 hit@5 below Cairn dense (an interval from +0.12 to +0.47), but Cairn was tuned on those questions, so is the tuned LlamaIndex row; on the 20 test questions no pair of rows is distinguishable (the smallest p is 0.25) and the gap fell to 0.10.
- **The test numbers as a ranking.** One question is 0.05; the sign tests have almost no power; nine rows and two metrics make a few intervals miss zero by chance.
- **The explanations marked "candidate cause".** The chunking ladder changes several things at once; the mixed effect of metadata, tokenisation for BM25, rank against score fusion, batch size for build time and BLAS for FAISS's exact search were not isolated.
- **Hybrid over dense as the default.** It is better on dev (hit@5 0.850 against 0.825, MRR 0.766 against 0.711, neither significant) and worse on MRR on test (0.533 against 0.659, interval -0.24 to -0.02) for the second test run in a row. The decision of ADR 0005 was made on dev and is not changed by test numbers; the evidence for it is weak.
- **Timing and memory to better than about 10%,** and **any build time measured while other work ran.**

## Consequences

- The test split is closed. Any later claim about retrieval quality rests on dev numbers (tuned on) and these two test runs (not tunable); a new held-out set would be needed for anything stronger.
- A future change to the chunkers, tokenizer, scoring or fusion is guarded by the CI gate of ADR 0009, not by this comparison.
- If LlamaIndex is compared again, the dev rows can be rerun deterministically (`bench/frameworks/run_all.py`), and the framework's tuned configuration should be re-chosen on a new dev set, not on these 40 questions.
