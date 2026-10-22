# Phase 4: checkpoint notes

The numbers quoted here come from running the real code on the real corpus (see `docs/results.md`).

## 1. Why does the dot product equal the cosine similarity here? What goes wrong if a vector is not normalised, and why must a zero vector stay zero?

The cosine similarity of `a` and `b` is `a . b / (|a| |b|)`: it compares only the direction of two vectors. If
both have length 1, the bottom is `1 x 1` and the cosine is just `a . b`. So I scale every vector to length 1
once, and then a search is one multiplication, `vectors @ query`.

Without that, the dot product also grows with the length of the vectors. Example: `a = (3, 4)` and
`b = (6, 8)` point the same way, and against the query `(1, 0)` they score 3 and 6, so `b` wins only because it
is longer. After scaling both are `(0.6, 0.8)` and both score 0.6.

A zero vector has length 0, and dividing by 0 gives `nan` (not a number), which would poison every score and the
sort. So its length is replaced by "leave it zero": the vector stays all zeros and its dot product with
anything is 0, so an empty text matches nothing, and nothing crashes. A test checks this.

## 2. What did dropping the query prefix cost, and why does a model care about it?

On `structure-300` with `bge-small`: MRR 0.711 to 0.605, hit@1 0.600 to 0.450, hit@10 0.925 to 0.825, and 4 more
questions missed at k = 10 (q016, q026, q031, q034). Every category got worse.

bge was trained to match a short question to a longer passage, and the questions in its training carried the
sentence "Represent this sentence for searching relevant passages: " in front. That tells the model "this text
is a search question, not a passage", and it moves the question's vector toward the passages that answer it.
Without the sentence the question is treated like any other text, and it lands nearer to texts that merely look
like it. Documents are embedded without a prefix.

It is not a rule that holds everywhere in my results: for the other three chunkers the prefix changes MRR by at
most 0.012 in either direction. I did not find out why only `structure-300` gains so much.

## 3. What is the memory for 100,000 chunks with 384 numbers each? Show the arithmetic. What would it be for 768 numbers, or for `float16`?

Each number is a `float32`, which is 4 bytes. `100,000 x 384 x 4 = 153,600,000` bytes, which is 153.6 MB. With
768 numbers it doubles: `100,000 x 768 x 4 = 307.2 MB`. With `float16` (2 bytes) the 384-number version halves:
`100,000 x 384 x 2 = 76.8 MB`. My real index, 3,267 chunks, is `3,267 x 384 x 4 = 5.0 MB`.

At this size memory is not a problem. It would become one at tens of millions of chunks (about 15 GB at 10
million chunks of 384 numbers), and that, together with search time, is what the ANN indexes of Phase 5 and
smaller number formats are for.

## 4. Why is the cache key built from the text the model saw plus the model and the document prefix, and not from `chunk.content_hash`? Give a concrete bug the simpler key would cause.

The vector depends on three things: the exact string given to the model, the model, and the prefix put in front
of documents. The key has to change whenever any of them changes, and only then. `content_hash` is a hash of
the chunk's text only (`chunkers.py`), so it misses all three.

Concrete bugs:
- **The heading is not in it.** The model sees `heading path + "\n" + text` (`index_text`). Two chunks with the
  same text under different headings have the same `content_hash`, so the second would be given the first's
  vector, although the model would have seen different strings.
- **The model is not in it.** I embed with `bge-small`, later run `--model minilm` on the same cache file, and
  every lookup finds the old `bge-small` vector. Both models have 384 numbers, so nothing even crashes: the
  index is full of vectors from the wrong model and the scores look normal.
- **The prefix is not in it.** The same thing happens if I change the document prefix.

The query prefix and the batch size are left out on purpose: they do not change a document's vector (queries are
not cached, and batching does not change the result).

## 5. Where did dense beat BM25 on your questions, and where did BM25 beat dense? Name one question of each.

Dense beat BM25 on q017, "Which spellings in the URL are accepted as 'true' for a boolean option?": BM25 ranked
the answer 212th (the passage has `true` but none of the other content words of the question), dense ranked it
1st. It is the clearest of the 7 questions that dense gets at rank 1 and BM25 does not; in general semantic MRR
went from 0.432 to 0.724.

BM25 beat dense on q028, "How do I stop the API description from being rebuilt every time someone opens the
docs?": BM25 ranked the answer 1st, dense 38th. The gold passage ("Cache the OpenAPI schema") says the schema is
generated "every time a user opens your API docs", so the question's own words match. Dense returned
`conditional-openapi.md`, a passage about hiding the docs: it matched the topic (the docs) and not the intent
(caching). Across the lexical questions BM25 has MRR 1.000 and dense 0.817. At rank 1 each wins on 7 questions
the other loses, which is why Phase 6 combines them.

## 6. Why sort texts by length before batching, and why must the output be put back in the original order?

A batch is padded to the length of its longest text, and the padding is computed like real text. If a short
text sits in a batch with a long one, most of its work is wasted. Sorting by length puts texts of similar length
together, so little is wasted. On a sample of 160 chunks: 5.4 chunks/s in the original order (batch 32) and 8.9
chunks/s sorted (batch 16), 1.65 times faster.

Sorting changes the order of the results: row 0 of the output is the shortest text, not the first chunk. The
index says "row 17 is chunk 17", so if the rows are not put back, a search returns the right vector with the
wrong chunk's text, file and line numbers, and nothing reports an error. A test checks that the rows come back in
the order of the texts.

## 7. A chunk of 7,124 tokens goes into a model that reads 512. What does the model see? What would you do about the 76 chunks like it?

It sees the first 512 tokens and silently ignores the other 6,612: the vector describes only the start of the
chunk, so a question about something later in the chunk cannot match it. (The 7,124 and the 76 come from the
ingest tokenizer; `bge-small`'s own tokenizer counts 83 chunks over 512, and the two numbers are not
comparable.)

I have not tested any fix, so these are options. (1) Cut the very large chunks into pieces under the limit
(for code, at statement or line boundaries) and index each piece. I have not looked at why the Phase 2
chunker left these 83 uncut. (2) Keep one chunk but embed
several windows of it and score it by the best window. (3) Leave them: I did not check whether any of the 16
dev failures involves a truncated chunk, so I cannot say it costs anything on these questions. The first check
would be to list the 83 chunks and see whether any dev question's gold span lies in the cut-off tail.
