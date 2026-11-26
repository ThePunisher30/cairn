# Cairn: building a retrieval engine from scratch, and checking it against LlamaIndex

*Draft for the author to rewrite in their own voice. The numbers are from `docs/results.md`; the caveats are part of the numbers.*

## What this is

Cairn answers questions about the FastAPI docs and source (pinned to tag 0.143.0) with citations. I built each layer myself in order to understand it, and measured every choice: chunking, BM25, embeddings, approximate nearest-neighbour indexes (IVF and HNSW), hybrid fusion, reranking, answer generation with citations and abstention, an agent loop with tools, an HTTP service with a live index, a web page, Docker and CI. The last step was to rebuild the same pipeline in LlamaIndex with FAISS and compare.

One thing to know before anything else: **an AI assistant wrote most of the later code** (Phases 8 to 11, the tests, the write-ups) at my request, and drafted the evaluation questions. The README says so. What I can claim is the decisions, the review, and being able to explain it.

## How it is judged

60 questions about FastAPI in four kinds (exact-word, paraphrased, needing several passages, about code), split into 40 for development and 20 held out. An answer is a file and a range of lines, not a chunk id, so it survives any change of chunking. A retrieved passage is a hit if its lines overlap the answer's. The held-out 20 were run exactly twice in the whole project, once after hybrid retrieval and once at the end.

## What the measurements showed

- **Chunking mattered most.** Cutting Markdown at its headings and Python at its functions kept 88% of the answers whole in 133-token chunks, against 78% for fixed 400-token chunks.
- **Small details were worth a lot.** Splitting `snake_case` and `camelCase` identifiers lifted BM25's MRR from 0.552 to 0.660; leaving off bge's query prefix cost dense retrieval 0.106.
- **Hybrid retrieval helped on the development questions** (hit@5 0.850 against 0.825 for dense alone) **and did not reproduce on the held-out ones**, twice. I kept it as the default because the decision was made on development data, and I say plainly that the evidence is weak.
- **The agent loop did not beat plain retrieval-then-answer** with the small local model I could afford: 23 of 40 against 29, not a significant gap, at 1.8 to 2.7 times the tokens. It skipped its own tools in 5 of 50 runs.
- **The service taught me about threads, not databases.** The first bottleneck was the embedding library using every core for one tiny query, so simultaneous requests fought each other; one thread per call gave 1.5 to 1.7 times the throughput. A route that does CPU work on the event loop froze the health check for 622 ms instead of 5 ms. A cache that matches questions by meaning could not separate "enable" from "disable" at any threshold, so it is off.
- **CI caught something the unit tests did not,** but only once I fixed a test of my own that counted chunks: shrinking chunks from 300 to 40 tokens then passed all 1,738 unit tests and failed the retrieval gate (hit@5 0.775 to 0.700).

## The comparison with LlamaIndex and FAISS

I wrote the protocol and eight predictions before measuring, listed the tuning candidates before running them, chose the best by a fixed rule on the development questions, then froze the nine rows in a committed file before the final held-out run. The runner refuses to start unless that freeze is in order.

- **On the development questions the default LlamaIndex pipeline was far worse** than Cairn: hit@5 0.525 against 0.825. The reason was not splitting answers (every answer sat whole inside one of its chunks) but chunks too big for the embedding model: 46% were longer than its 512-token window. Making chunks smaller, splitting Markdown at headings and Python with a code-aware splitter brought it to 0.775 and the same MRR as Cairn's dense retrieval.
- **A trap in the documented setup:** with the FAISS index its documentation shows, LlamaIndex returns distances as scores (smaller is better), and its rank fusion sorts largest first, so it silently reverses the dense half: MRR 0.391. With an inner-product index the same fusion reached 0.614, still below Cairn's hybrid.
- **FAISS's HNSW was 4.2 to 5.4 times faster than mine at the same recall and built 21 times faster; its exact search was slower than my numpy version.** I predicted at least 5 times and 2 times faster respectively, so I was wrong on both counts.
- **The framework pipeline is 18 to 28 lines against 922 for mine,** which is what you get for letting a library do the work; it does not say mine is worse code or better code.
- **On the 20 held-out questions I cannot tell any of the systems apart.** The gap between Cairn and the default shrank from 0.30 to 0.10 and is not significant; no pair of rows has a p value below 0.25. The rows I had tuned on the development questions tended to lose on the held-out ones (the best-of-eight LlamaIndex row lost 0.125), and two untuned rows gained. That is what selecting on a small set looks like, and it means the development numbers are optimistic for anything I tuned, including Cairn.

Three of my eight predictions were wrong and one mostly wrong. Among the wrong ones was the cause I gave for the default's weakness.

## What I would not claim

That Cairn beats LlamaIndex; that hybrid retrieval is better than dense; that the explanations I marked "candidate cause" are proven; or anything about a real language model under load, another framework, or another vector store. The held-out set has 20 questions, is now used up, and one question is worth 0.05.

## Reproduce it

`docs/results.md` lists every command. Retrieval numbers come from `scripts/run_eval.py`; the comparison from `bench/frameworks/` (`run_all.py` for the development rows, `paired.py` for the statistics); the CI gate from `scripts/ci_gate.py`, which runs in about two seconds without a network.
