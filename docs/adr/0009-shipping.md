# ADR 0009: A page served by the API, a retrieval-only public demo, an image built from the pinned corpus, and a CI gate on a deterministic fixture

Status: accepted, 2026-11-26. Evidence: `docs/results.md` (section "Phase 10"), `docs/deploy.md`.
The gate and the image were run on this machine; **the CI workflow ran on GitHub once after a fix** (the first run failed on a YAML error, the second passed both jobs), and **the public demo was not deployed** (it needs the author's hosting account). Section "What I would not trust" says where that matters.

## Context

Phase 10 makes the project something a stranger can run and trust: a page to use it, a container, CI that guards the quality, and a place to host a demo. The constraint is unchanged: spend nothing. That rules out a paid model behind a public URL and any paid host.

**Who wrote it.** In this phase the AI assistant wrote the page, the Docker and CI files, the gate and the deployment preparation at the author's request, with tests.

## Decision

- **The API serves the page** (`serve/static`: one HTML file, one script, one stylesheet, no framework, no build step). It reads the server-sent events with `fetch` (a POST cannot use `EventSource`), shows tokens as they come, replaces them with the final text on `done`, turns `[n]` markers into buttons, and opens `GET /source` in a dialog with the cited lines highlighted. Everything from the server (model text, file text) goes in with `textContent`; a test fails if the script uses `innerHTML` and friends, and the page is served with a policy (`default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'`, no framing) that forbids inline script and style.
- **`GET /source` is a path from the browser, so it is treated like a path from the model:** the Phase 8 checks (no absolute path, no `..`, nothing that resolves outside the corpus), plus only files that were indexed (the corpus definition itself), at most 400 lines, and the same key and rate limit as the other routes.
- **Answers carry structured `references`** (number, path, lines, GitHub URL) so the page does not parse the Markdown `sources` strings.
- **The public demo is retrieval only** (`--llm none`): `/ask` returns the retrieved passages with `degraded: true` and the reason, and the page says so. It costs nothing and still shows the retrieval work. It is a separate mode, not a failure: it is not counted as one in the metrics.
- **A BM25-only mode** (`--retriever bm25`): no torch, no embedding model, no dense index. In containers: 93 MiB and a 457 MB image, against 454 to 742 MiB and 2.65 GB with the model (measured on one machine), so it fits hosts with 512 MB. It is the Docker default.
- **One Dockerfile, three stages.** `data` clones the corpus at the pinned tag, chunks it and (with `WITH_DENSE=1`) embeds it; `runtime` holds only the service and the data; `demo` (the last stage, so the default target) adds the public settings. The data is baked in at build time, so `docker build` is the whole setup and the image needs no network at run time. The service runs as a non-root user.
- **Public settings:** no key, retrieval only, a shared rate limit of 10 requests per second with bursts of 30. Behind a hosting proxy all visitors share the proxy's address, so this is one global limit; trusting `X-Forwarded-For` would let a visitor pick their own address.
- **CI** (`.github/workflows/ci.yml`): format and lint, unit tests, the retrieval gate, then a build of the demo image that is started and asked a question. Read-only token, a concurrency group that cancels superseded runs, and every action pinned to a commit (looked up on 2026-11-26; the comment names the version).
- **The retrieval gate** (`scripts/ci_gate.py`): the real chunkers, BM25 and fusion over `tests/fixtures/ci_corpus` (every file a dev question points into plus 40 seeded distractors, copied unchanged so the line numbers hold) with the word-hashing model as the dense half. Deterministic, about 2 s, no network. The thresholds (`eval/ci_gate.json`) are the measured values: hit@5 may not drop at all (one question is 0.025), MRR may drop by 0.01.
- **Deployment is prepared, not done:** `scripts/make_space.py` writes the folder for a Hugging Face Space and `docs/deploy.md` gives the steps and the caveats.

## Why

- **No framework and no build step for the page:** the page is about 300 lines, it is read by someone checking the project, and a toolchain would add a lot to review. The cost is that the script has no automated tests of its own; it was checked in a real browser and by the string checks above.
- **Retrieval only:** a free host cannot run a model like `gemma4:e4b` (about 40 s per answer on a 12-core laptop), and a key behind an open page makes every visitor a cost. The same page answers in full when run locally with Ollama.
- **Baking the data into the image:** a container that downloads and indexes at start is slow to start and fails if the network does. The price is that the image holds one version of the corpus.
- **A fixture and a hashing model for the gate:** the real model needs a download and its numbers shift slightly with library versions; a gate that flakes gets ignored. The hashing model makes the dense half weak (hit@5 0.550 against BM25's 0.775), so the gate mostly guards the chunkers, the tokenizer, the scoring and the fusion, which is what a code change can break.
- **Exact thresholds:** the run is deterministic, so there is no noise to allow for. A change that loses one question must be a visible, reviewed edit of `eval/ci_gate.json` (`--update` prints the new numbers).

## Trade-offs

- **The gate is not the real system.** The fixture has 693 chunks against 3,267, and the dense half is a stand-in, so its numbers are not the ones in `docs/results.md`. It answers "did this change make ranking worse on a fixed set", not "how good is retrieval".
- **An open demo with a shared limit** can be used up by one visitor; it is a brake, not fairness.
- **BM25-only gives up the dense half** (hit@5 0.750 against hybrid on the dev set, see `docs/results.md` for the rest).

## What I would not trust

- **That CI stays green.** It passed once, on pull request 14, after a first run failed on a YAML mistake that my local checks had not caught (they ran the commands, not the workflow file). On the Linux runner 1,750 tests passed (6 skipped, against 1,749 passed and 7 skipped in my Windows run: platform-dependent tests) and the gate printed the same numbers as on Windows, which supports its determinism across the two systems.
- **The gate as a measure of quality.** It uses the 40 dev questions the project was tuned on, a small fixture and a weak dense model.
- **The page beyond one browser and one session.** It was checked in Chrome only, by one person's tooling, with no automated script tests and no accessibility audit.
- **The demo's availability.** It is not deployed, and the free hosting tier was reported inconsistent when checked (see `docs/deploy.md`).
- **The memory figures.** They come from `docker stats` on one machine after a few requests (the hybrid container read 454 MiB in one run and 742 MiB in another); a different host will differ.

## Consequences

- Pull requests now run the workflow; a red run blocks nothing automatically (no branch protection is configured), so the merge decision is still a person's.
- Changing a chunker, the tokenizer, BM25 or fusion now needs either no change in the gate numbers or an `--update` in the same pull request, with the reason written down.
- Phase 11 (the comparison with the frameworks) can start the same containers for a fair timing of end-to-end requests.
