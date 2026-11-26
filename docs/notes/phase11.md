# Phase 11: defend every number

The brief's test is: "you can defend every number in the README without opening the code". This sheet is for that. Each line is a number, where it comes from, what limits it, and the question an interviewer is likely to ask. All numbers come from `docs/results.md` ("Phase 11" and the earlier phases).

> **Draft:** these were drafted with the assistant, and the assistant also wrote the code. Rewrite the answers in your own words; if you cannot explain one without reading it, that line is the one to study.

## The comparison

| Number | Where it comes from | The limit to state | Likely question |
|---|---|---|---|
| LlamaIndex default hit@5 0.525 against Cairn dense 0.825 (dev) | the same 40 dev questions, same model and metric, one process per row | Cairn was tuned on these questions; the interval for the difference is +0.12 to +0.47 | "Is that a fair comparison?" |
| The default's chunks: 518 tokens, 46% over bge's 512-token window | `chunk_stats` in each result file | the unit is Cairn's token counter for the mean and bge's own tokenizer for the 512 limit | "Why is the default worse?" |
| 100% of the default's gold spans are whole in one chunk, 88% for Cairn, 54% for 256-token chunks | `cairn.evaluation.integrity` on each chunker's output | it measures whether an answer fits in one chunk, not whether it is retrieved | "Does the default split code mid-function?" (No: I predicted that and was wrong.) |
| Tuned LlamaIndex 0.775 hit@5, MRR 0.711 (Cairn dense 0.825, 0.711) | best of 8 candidates chosen on dev by a rule fixed beforehand | the winner won a tie on MRR; 8 candidates on 40 questions | "Did you tune it fairly?" |
| Test: nothing distinguishable between rows (smallest p 0.25) | the second and last test run, 20 questions, nine frozen rows | one question is 0.05; the rows tuned on dev tended to lose on test | "So who is better?" (The data cannot say.) |
| Hybrid MRR on test 0.533 against dense 0.659 | the same run | interval -0.24 to -0.02; second test run in a row where hybrid did not beat dense | "Why is hybrid your default then?" |
| LlamaIndex fusion MRR 0.391 (L2) against 0.614 (inner product) | the score direction of FAISS L2 results is smaller-is-better and the fusion sorts largest first | I checked the score order directly; the better row was added after seeing the failure | "What did the framework get wrong?" |
| FAISS HNSW 4.2 to 5.4 times faster than Cairn's at the same recall; build 22.5 s against 482 s | one run, the Phase 5 synthetic set, one query at a time | synthetic vectors; the Phase 5 table measured Cairn's HNSW 2.3 times slower on another day | "Why is yours slower?" (Python against C++ and SIMD; I did not profile it.) |
| FAISS exact search 0.65 times Cairn's numpy | same | FAISS reported one thread here | "FAISS was slower than numpy?" (A single query is one matrix-vector product; untested why.) |
| 922 lines against 18 to 28 | `bench/frameworks/loc.py`, a stated counting rule | the framework count excludes the library, Cairn's includes its chunkers and BM25 | "So the framework is 40 times less code?" |
| Peak memory +7 to 14%, build time embedding-dominated, query p50 within 25% | two idle repeats | about 10% noise in time, none in memory; every row imported the embedding stack | "What does the framework cost?" |

## Earlier phases (from the README)

| Number | Limit to state |
|---|---|
| BM25 hit@5 0.750, dense 0.825, hybrid 0.850 (dev) | tuned on dev; hybrid's lead is one question and did not reproduce on test |
| Structure-aware chunking keeps 0.88 of gold spans whole against 0.78 for fixed 400-token chunks | the gold spans were drafted with AI help |
| Agent 23 of 40 against single-shot 29 (p = 0.24) | one run, a small local model, a same-model judge |
| 1.5 to 1.7 times the throughput with one torch thread | a fake model, the load generator on the same machine, 20% run-to-run spread |
| 3 of 45 chunks re-embedded after changing 3 of 671 files | one run on the real index |
| The CI gate failed a change that passed all 1,738 unit tests (BM25 hit@5 0.775 to 0.700) | after I fixed a test of my own that hard-coded a chunk count; one change out of seven tried |

## Three things worth being able to say without notes

1. **What I measured and what I did not.** I measured retrieval quality, cost and speed against LlamaIndex with FAISS. I did not measure LangChain or other vector stores, a paid model, or the real model under load, and the test split has 20 questions.
2. **Where I was wrong.** Three of eight predictions failed, including the cause I gave for the default's weakness; and the best row on dev lost 0.125 on test.
3. **How it was built.** Phases 8 to 11 were written by an AI assistant at my request, with tests, and the README says so. The questions and the gold spans were also drafted with AI help. What is mine is the decisions, the review, and understanding it.
