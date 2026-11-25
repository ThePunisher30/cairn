# Phase 9: checkpoint notes

The numbers come from `docs/results.md`, section "Phase 9". Everything was measured with a **fake language model** on one laptop with the load generator on the same machine, so none of it says how the service behaves with a real model under load.

> **Draft:** these answers were drafted with the assistant from the measurements, and in this phase the assistant also wrote the implementation. The brief asks for answers in my own words, so each one should be rewritten in my words (keeping the numbers) before the notes count as mine.

## 1. Why does your event loop stall during embedding, and how did you fix it?

The event loop is one thread that runs every request's code in turns; it can only switch to another request when the current one says `await` and waits for something. Embedding a query is a CPU-bound call that never waits: it holds the thread until it is done (about 35 ms for one query, more under load). If a route calls it directly inside an `async def`, nothing else runs in that time, not even `GET /health`.

I measured it instead of assuming it. With 20 users on the same retrieval, `/health` (called every 100 ms by another client) answered in a median of 5.4 ms (max 38 ms) when the search was a plain `def` route, and in a median of 622 ms (max 1,422 ms) when it was an `async def` route that called the search directly (`/debug/search-blocking`, which exists only to show this). A unit test shows the same with one request and a 0.4 s search.

The fix is that the routes are plain `def`: FastAPI runs them in a thread pool, and the event loop stays free to answer other requests. This works because the heavy work (a torch matrix product, a numpy dot product) lets other threads run while it computes. It did not remove the next limit: the pure-Python part (BM25 scoring, tokenizing, fusion) shares one interpreter lock, so one process tops out at about 20 to 29 searches per second here. And there was a second, separate stall I did not expect: torch used all 12 cores for each query, so concurrent requests fought each other (12 requests/s from 5 to 20 users). Setting torch to one thread per call gave 1.5 to 1.7 times the throughput.

## 2. What is a semantic cache's failure mode?

It answers a different question with the answer to a similar one. A semantic cache stores questions as vectors and serves a stored answer if a new question is close enough (cosine similarity above a threshold). The vectors measure "about the same topic", not "wants the same answer", so questions that differ in one word that flips the meaning ("true" against "false", "enable" against "disable", "has a body" against "has no body") sit very close together.

I measured it with 15 hand-written paraphrases and 18 near-misses on dev questions. The paraphrases' median similarity was 0.93 and the near-misses' was 0.94, so no threshold separates them: at 0.92 the cache would answer 10 of 15 paraphrases and also 13 of 18 near-misses with the wrong stored answer; "accepted as 'true'" and "accepted as 'false'" scored 0.985. Only at 0.98 do the false hits drop (2 of 18) but then only 5 of 15 paraphrases hit. A higher threshold is a trade, not a fix. So I built it, tested it and left it off by default; the exact cache (the same question after lowercasing and trimming) is on. The second failure mode, staleness (a cached answer citing text that was since changed), is handled by putting the index version in the cache key. The caveat is that the samples are small and the pairs were written by one person who knew what a near-miss was.

## 3. What does a tombstone cost you?

A tombstone marks a chunk dead instead of removing it, so nothing has to be renumbered. What I measured on the real index (3,267 chunks) is that search does not get slower, and does not get faster either: at 75% dead, BM25 took 1.18 ms against 0.76 ms for an index compacted to the same documents (the dead postings are still walked), and the dense scan took 0.30 ms against 0.05 ms (it multiplies every slot, dead or not; at 0 to 50% dead it stayed at 0.2 ms while documents disappeared). So the cost is wasted scanning and the memory of the dead slots, and a rebuild to get them back: compaction took 42 ms under the write lock, and searches wait for it.

The rebuild threshold is a choice: compact too often and the pauses add up, too rarely and the wasted scan grows. I set it at 25% dead slots (and at least 50 slots). The scale matters: with 3,267 chunks everything here is milliseconds, so I would not trust the threshold for a corpus a thousand times larger. I also did not meet HNSW's version of this problem (a deleted node is still a node in the graph that other nodes point to): the service uses a flat matrix and the Python postings, not the approximate indexes of Phase 4.
