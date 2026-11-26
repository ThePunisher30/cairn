# Phase 11: the comparison with LlamaIndex and FAISS (protocol and predictions)

Written **before** any framework number was measured, so that the predictions can be wrong in public and the dev and test rules cannot be bent afterwards. The assistant writes the code (at the author's request); this file says what is compared, what is held equal, what is predicted, and how the one remaining test-split run is spent.

## The question

What do the frameworks do for you by default, what does that cost or gain on the same 40 dev questions, and can every gap be explained with something built in Phases 2 to 5? Not "which is better": the frameworks were built for convenience across thousands of uses, Cairn for one corpus.

## What is compared

| Row | What it is |
|---|---|
| **Cairn BM25 / dense / hybrid** | the Phase 3, 4 and 6 retrievers, re-measured with the same script for build time, memory and latency |
| **LlamaIndex default** | `SimpleDirectoryReader`, `SentenceSplitter` at its defaults, `HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")`, `VectorStoreIndex` over a FAISS flat index, retrieval `similarity_top_k` = 10 (its own default is 2) |
| **LlamaIndex tuned** | the same with chunking and reader settings chosen **on dev only** from a fixed, listed set of at most 8 candidates (every candidate tried is reported, not just the winner) |
| **LlamaIndex BM25 + fusion** | its BM25 retriever and `QueryFusionRetriever` over the dense one, default then tuned (only if the package installs cleanly) |
| **FAISS indexes alone** | `IndexFlatIP` and `IndexHNSWFlat` on the Phase 5 synthetic 100,000-vector set, against Cairn's numpy flat index and from-scratch HNSW, at the same recall |

pgvector and Chroma are not in this phase (a database server for little added insight; one store is enough to see the pattern).

## Held equal

The corpus files (the three folders in `ingest/documents.py`, release notes excluded); the embedding model `BAAI/bge-small-en-v1.5` with bge's query instruction and unit-length vectors (so inner product is cosine); k = 10 for the quality numbers; the 40 dev questions and the same hit rule (a chunk is a hit if its lines overlap a gold span); the same machine, run one at a time.

**Deliberately not equal** (it is the point): the reader, the chunker, the retrieval code. The adapter turns each framework node into a `Hit` with line numbers computed from the character offsets of the node in its document, so the harness scores it exactly like a Cairn chunk.

**A caveat that goes in the results:** the hit rule is "overlaps a gold span", so a large chunk scores a hit more easily than a small one. Every row therefore also reports the mean chunk size in tokens and the tokens in the top 5, and a framework row with bigger chunks must not be read as better retrieval without that.

## Measured for every row

hit@1, hit@5, hit@10, recall@10, MRR, nDCG@10 (the existing harness, overall and per category); chunks and mean/max chunk tokens; **build time** (read, chunk, embed, index; a cold embedding cache for every row); **peak memory** of the process (peak working set); **query latency** p50 and p95 for retrieval only, one query at a time; **lines of code** of what you write for the pipeline (non-blank, non-comment lines, counted by one script, with the rule stated: Cairn's count includes its chunkers and BM25, the framework's does not include the library).

## Predictions (to be marked right or wrong in `docs/results.md`)

1. **LlamaIndex default has hit@5 between 0.70 and 0.85 and MRR between 0.55 and 0.72** (Cairn dense: 0.825 and 0.711). Reason: chunks of about 1,000 tokens against Cairn's 133, so more answers sit inside a retrieved chunk (helps hit@10) but the embedder reads only the first 512 tokens (bge's limit) and the splitter ignores Markdown headings and Python structure (hurts the rank of the right chunk).
2. **The weakest category for the default is `code`**, because the splitter cuts Python files mid-function and mid-line-range.
3. **Tuned LlamaIndex gets within 0.05 hit@5 of Cairn dense** by using smaller chunks and a structure-aware parser, but not above it by more than 0.05.
4. **Query latency p50 is within 25% of Cairn's** for the dense rows (about 100 ms): the cost is embedding the question on the CPU, not the scan.
5. **FAISS HNSW answers at least 5 times more queries per second than Cairn's HNSW at equal recall**, and FAISS flat at least 2 times more than Cairn's numpy flat.
6. **Peak memory of the LlamaIndex row is more than 30% above Cairn's dense row** (the import alone loads a lot).
7. **Build time is dominated by embedding for every row**; the framework row with fewer, longer chunks is not faster per token.
8. **The framework pipeline is shorter than Cairn's, by roughly an order of magnitude, in lines you write** (the library does the rest).

## Rules for tuning and for the test split

- **Dev only until the freeze.** Candidates are listed in `bench/frameworks/candidates.json` before they are run; the selection rule is highest hit@5, ties broken by MRR. With 40 questions and up to 8 candidates the best one is partly luck (as in Phase 6), and the results say so.
- **The freeze.** The chosen configurations, and the Cairn ones (the Phase 6 default), are written to `bench/frameworks/frozen.json` and committed. Nothing is tuned after that commit.
- **The test run is exactly one run of all frozen rows together** on the 20 test questions (5 per category, so one question is 0.05 and a gap under about 0.1 is noise), appended to `eval/test_set_runs.log` as run 2 of 2. It is not run until the author says go. Dev and test are reported side by side.
- **The 5 held-out multi-hop questions (m11 to m15) need a language model and the agent**; whether to spend them is a separate decision for the author, not part of this retrieval comparison.

## Not part of this phase

Generation or the agent through the frameworks; a paid model; pgvector, Chroma, LangChain; tuning Cairn to the test set.
