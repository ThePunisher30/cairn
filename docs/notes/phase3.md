# Phase 3: checkpoint notes

The numbers quoted here come from running the real code on the real corpus (see `docs/results.md`).

## 1. What does `b = 0` do? What happened to your score when you set it to 0, and why?

`b` controls length normalisation. In the formula the term frequency is divided by
`1 - b + b * dl / avgdl`. With `b = 0` this is always 1, so a chunk's length no longer matters at all; with
`b = 1` a chunk twice the average length has its word credit shrunk as if it were twice as hard to impress.

On the real corpus, MRR dropped from 0.660 (`k1 = 1.2`, `b = 0.75`) to 0.320 at `b = 0`, and it is about 0.24
to 0.33 for every `k1` in that column. My explanation: the Phase 2 chunks are very uneven (mean 87 tokens, but
76 code chunks over 512 tokens and one of 7,124). A very long chunk contains many different words, so it
collects a little credit from nearly every word of a query. Without length normalisation those long chunks
outrank the short, focused chunks that actually answer the question. I have not run a test that shows
this chunk by chunk, so it is the most likely cause, not a proven one.

## 2. Why does a word that is in every chunk contribute almost nothing? Why `ln(1 + ...)` and not `ln(...)`?

If every chunk contains a word, finding it tells you nothing about which chunk is relevant. The idf says so:
with `df = N` it is `ln(1 + 0.5 / (N + 0.5))`, about `0.5 / N`. On my 3,267 chunks that is 0.0002, while
`the` (in 1,424 chunks) is worth 0.83 and `add_task` (in 8 chunks) is worth 5.95, about seven times more.

The textbook form `ln((N - df + 0.5) / (df + 0.5))` turns negative once a word is in more than half of the
chunks: for N = 10, a word in 6 chunks gives -0.37 and a word in all 10 gives -3.04. Then containing a common
word would lower a chunk's score. Adding the 1 inside the logarithm keeps the value above 0 (for N = 10 and
`df = 10` it gives +0.047), so common words are simply worth very little, and never worth less than nothing.

## 3. Why did splitting `camelCase` and `snake_case` help or hurt? Give one question where it made the difference.

It helped. The code and docs are full of identifiers, but the questions use plain words. With the plain
tokenizer, `generate_unique_id` is one word that only a search for exactly that string can match. Keeping the
whole identifier and also its parts (`generate`, `unique`, `id`) lets "unique ID" in a question match it, and
the exact name still works.

Question q032: "How is the default unique operation ID of a route assembled?". The answer is the function
`generate_unique_id`, which builds `operation_id` from the route name and path. With the plain tokenizer only
`route` matched, and the answer ranked 226th; with splitting `id`, `operation`, `route` and `unique` match,
and it ranks 2nd. Over all questions, splitting was better on 15, worse on 7 and equal on 18. For the chunking
we chose, MRR went from 0.552 to 0.660, with lexical going from 0.820 to 1.000 and code from 0.334 to 0.550.
It did not help the semantic questions (0.439 to 0.432): those fail because the passage uses different
words, which no tokenizer can fix.

## 4. Why keep a heap of size `k` instead of sorting all the scores? What does building the index cost, and one search?

A query can give a score to about 2,000 chunks but I only want the best 10. Sorting all of them costs about
`n log n` comparisons (roughly 22,000 for n = 2,000); a heap that only remembers the best `k` costs about
`n log k` (roughly 6,600). At this size the saving is small, because most of a search is walking the postings
in Python; it matters when the corpus is a thousand times bigger, and it is a habit worth having.

Building the index is one pass over every token of every chunk: 283,772 tokens here, into 133,862 postings
entries, and it took 0.29 seconds. It is done once and saved. A search costs the postings of the query's
distinct words (the sum of their document frequencies) plus the heap work. Median search time was 2.6 ms and
p95 4.1 ms over 800 searches.

## 5. Look at three misses. What can keyword search never do, and which phase fixes it?

- **q026** ("How can I turn off the interactive documentation pages in production without editing the
  code?"): the gold passage says "disable OpenAPI ... set the environment variable `OPENAPI_URL`". It shares
  only `can` and `the` with the question, and ranked 1217th. Different words, same meaning.
- **q031** ("For which HTTP status codes does FastAPI decide that a response must have no body?"): the answer
  is the function `is_body_allowed_for_status_code`, but the tutorial page about status codes uses the
  question's words more densely (it scores 36 and is the top hit; the answer ranked 188th), and the question
  says `codes` where the code says `code`.
- **q043** (what the HTTPException handler returns in an example): the gold span is a whole example file that
  `structure-300` cut into 4 chunks, so no single chunk matches the question's words as well as a docs page.

Keyword search can never match a passage that says the same thing in other words, because without a shared
word there is nothing to look up. Embeddings (Phase 4) compare meaning and should fix that. It also cannot
know that a question wants source code and not the documentation page, which hybrid retrieval and
reranking (Phase 6) should help with; stemming is a cheap tokenizer experiment for the `codes`/`code` case, and
the cut-up example file is the chunking problem from Phase 2.

## 6. Why does the index store its tokenizer, and why does the retriever check chunk IDs? What bug does each prevent?

The query must be cut into words exactly the way the chunks were. If the index was built with splitting and
queries were tokenized without it (or the other way round), words quietly stop matching: searches return
fewer or worse results and nothing reports an error. Storing the tokenizer inside the index, and always using
it for queries, makes that mismatch impossible.

The postings only contain chunk numbers ("chunk 17"). If I re-run ingest after a change, chunk 17 can be a
different chunk, and the search would return the wrong text and the wrong file and line numbers with nothing
to show it. The index remembers each chunk's ID (a hash of its place and content from Phase 2), and the
retriever raises an error if the IDs of the chunks it is given are not exactly the same list, so the
fix ("rebuild the index") is stated instead of a silent wrong answer.
