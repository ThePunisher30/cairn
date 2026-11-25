# Phase 10: checkpoint notes

The numbers come from `docs/results.md`, section "Phase 10". The CI workflow ran on GitHub (the first run failed on a YAML error, the second passed both jobs), and the public demo was not deployed.

> **Draft:** these answers were drafted with the assistant from the measurements, and in this phase the assistant also wrote the page, the Docker and CI files and the gate. The brief asks for answers in my own words, so each one should be rewritten in my words (keeping the numbers) before the notes count as mine.

## 1. What does your CI gate catch that unit tests don't?

It catches "every piece still does what its test says, and the system finds worse answers". The unit tests ask whether each function does what was specified; the gate asks the question the project exists for: on a fixed set of 40 questions, is the right passage still in the top five?

I tried seven plausible regressions against both (the tokenizer no longer splitting `snake_case` and `camelCase`; prose chunks shrunk from 300 to 40 tokens; BM25's length normalisation turned off in two ways; repeated query words counted twice; the heading path left out of the indexed text; min-max normalisation replaced by a division). The gate caught all seven, and so did the unit suite, because this repository's tests pin defaults and behaviours closely. So for those the gate was not needed to find the bug.

The difference showed up in one of them. Shrinking prose chunks to 40 tokens failed exactly one unit test, and that test was one of my own that hard-coded a chunk count (`12`, "3 files x 4 chunks"), which is a number that changes when the chunk size changes for a reason unrelated to quality. I made that test independent of chunk size. Then the change passed **all 1,738 unit tests** and the gate failed: BM25 hit@5 fell from 0.775 to 0.700 and MRR from 0.703 to 0.560 (hybrid hit@5 0.775 to 0.700). That is the case the gate is for: a tuning change that looks harmless, breaks no stated behaviour, and makes retrieval worse. The gate also makes a deliberate loss visible: to accept it someone has to edit `eval/ci_gate.json` in the same pull request.

What it does not catch: it uses the 40 dev questions the project was tuned on, 693 chunks instead of 3,267, and a word-hashing stand-in for the embedding model (hit@5 0.550 against BM25's 0.775), so a change that only hurts the real embedding model, or a question type not among the 40, passes. It guards the code (chunkers, tokenizer, scoring, fusion), not the real model's quality.

## 2. What did the deploy force you to change?

I did not deploy (it needs the author's hosting account), so this is what preparing for it, and building the container, forced:

- **A mode with no language model.** A free host cannot run the model I used (about 40 s per answer on a 12-core laptop), and an open page with an API key behind it would let anyone spend it. `/ask` now has a retrieval-only mode that returns the passages and says why, and the page shows them. That meant separating "no model attached" from "the model failed", so it is not counted as a failure in the metrics.
- **A mode with no embedding model.** The hybrid container used 454 to 742 MiB (two runs) with the model loaded, and its image is 2.65 GB; the BM25-only container used 93 MiB and its image is 457 MB. `--retriever bm25` exists because a 512 MB host cannot hold the other.
- **Data inside the image.** The index files are gitignored and generated, so a container cannot start from a clone. The Dockerfile downloads the pinned corpus and builds the indexes while building, which also made the image start in about 10 s, with no network at run time.
- **Paths I had assumed.** The cost estimate reads `eval/llm_prices.json` relative to the source tree (`config.PROJECT_ROOT`), which by reading the code would point somewhere else for a normal (non-editable) install. I did not test that case; the image installs the package in editable mode so the paths hold.
- **A source viewer that is a file-reading endpoint.** A citation chip has to show lines, so the server now reads files for a browser. I reused the path checks from Phase 8 and added the rule that only indexed files can be read, with tests that try `..`, absolute paths and a file outside the indexed folders.
- **The rate limiter's assumption.** It limits per client address, which behind a hosting proxy is the proxy's address, so for all visitors it is one shared limit. I chose to keep that (and say so) rather than trust `X-Forwarded-For`, which a visitor could set to anything.
- **Two things the browser showed that tests had not:** the API-key box was visible when no key was needed (a CSS `display: flex` overrode the `hidden` attribute), and the metrics labelled page files as "unmatched". Both were fixed and the second has a test.
- **A plan B for the host.** The documentation lists the free CPU tier as 2 vCPU and 16 GB, while 2026 forum posts report new accounts getting a quota error; I could not check which is true for this account, so the deploy notes say how to run the same image anywhere.
