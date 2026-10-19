# Phase 2: checkpoint notes

The numbers quoted here come from running the real code on the real corpus (see `docs/results.md`).

## 1. Why must a chunk ID be computed from the content and position instead of a counter? What does the content hash add?

A counter gives "chunk 17" whatever happens to be 17th. Add a file, change a chunker setting or sort
differently and every number shifts, so the same chunk gets a different ID from run to run. My ID is the
hash of `path:start_line:end_line:content_hash`. The same text at the same place always gets the same ID,
so running ingest twice gives a byte-identical `chunks.jsonl` (a test checks this), and two chunks with
identical text in different files still get different IDs because the position is part of the recipe.

The content hash is a fingerprint of the text alone. It adds two things the position cannot: I can tell
whether a chunk's text changed (later phases can re-embed only the chunks whose hash changed), and I can
spot the same text appearing in several places.

## 2. Why do chunk sizes use a token count and not a character count? What is wrong with our counter, and where would the real tokenizer matter?

Models read and limit text in tokens, not characters. The embedding model I plan to use has a limit of
512 tokens, and the same number of characters can be very different numbers of tokens: `x = 1` is a few
characters and 3 tokens, while a long word is many characters and one or two tokens. Sizing by characters
would give chunks that are fine for the model in one place and truncated in another.

My counter is a regular expression that splits words at camelCase boundaries and counts numbers and each
punctuation mark. It only approximates a real tokenizer (WordPiece or BPE), which splits rare words and
identifiers into sub-word pieces in its own way. I have not measured how far off it is. It matters most
where the limit is hard: an embedding model silently ignores everything past its limit, so a chunk I count
as 500 tokens could really be 560 and lose its tail. I should compare the two on the real chunks in
Phase 4, when I pick the embedding model.

## 3. Why does the heading path matter for retrieval? Give an example of a chunk that is ambiguous without it.

A chunk from the middle of a page often does not say what it is about. In the FastAPI docs there are 9
chunks whose section is called "Check it". Two of them:

- `advanced/security/oauth2-scopes.md`: heading path `('OAuth2 scopes', 'Check it')`, text starts
  "If you open the API docs, you can authenticate and specify which scopes..."
- `advanced/wsgi.md`: heading path `('Including WSGI - Flask, Django, others', 'Check it')`, text starts
  "Now, every request under the path `/v1/` will be handled by the..."

Without the path, the search text of the second one is about `/v1/` and requests and never mentions WSGI,
so a question like "how do I mount a Flask app?" has nothing to match. `Chunk.index_text` puts
`OAuth2 scopes > Check it` in front of the text, so the page and section names become searchable words and
the chunk carries its own context.

## 4. Why does `RecursiveChunker` try paragraphs before lines? What does overlap buy, and what does it cost?

A paragraph is the author's own unit of thought, so a cut between two paragraphs loses the least. A cut
between two lines can land in the middle of a sentence or a list. The chunker only falls back to lines when
a single paragraph is too big for the budget.

Overlap repeats the last few lines of a chunk at the start of the next one, so something that straddles a
boundary appears whole in at least one of the two. It helped on the real questions: recursive-200 went from
0.620 to 0.680 intact with 3 lines of overlap (code 0.600 to 0.700, lexical 0.700 to 0.800). The cost is
duplicated text: the number of chunks stays 2,767 but the mean size goes from 157.5 to 185.6 tokens
(about 18% more text to store and embed), and the same passage can come back twice in the results, which
wastes space when the chunks are given to an LLM.

## 5. Why is "gold integrity" an upper bound on retrieval quality? What can it not tell you?

Gold integrity asks whether some chunk contains the whole gold span. If none does, the answer is cut
across chunks, and no retriever, however good, can hand back the whole answer in one piece. It does not
depend on the retriever at all, so it is a ceiling: a strategy with 0.70 intact on code questions cannot
return a whole answer for more than 70% of them.

It cannot tell me three things. Whether the retriever finds the chunk (it only measures the chunker).
Whether the chunk is useful: a chunk holding a whole file scores 1.0 and is a poor search result, which
is why I also look at chunk counts and sizes. And how much a cut answer still helps: the Phase 1 metrics
give credit when a result shares at least one line with a gold span, so a cut span can still produce hits.
Integrity is a strict ceiling on getting the answer in one chunk, not on every metric.

## 6. What did you find about the oversized chunks, and what would you do about them in a later phase?

76 chunks are over 512 tokens, all of them code and none of them prose (prose sections are split by
paragraph when they are too long; the AST chunker never splits a function). The largest is
`FastAPI.__init__` in `fastapi/applications.py`, lines 64-1052, with 7,124 tokens: about 60 parameters,
each documented inline with a long `Doc("...")` string. The next largest, 2,800-3,400 tokens, are
`get_request_handler` and the near-identical HTTP-method functions `APIRouter.put/post/head/patch/trace`
and `FastAPI.put`. They cluster in `routing.py` (21), `applications.py` (13), `param_functions.py` (9),
`params.py` (8) and `dependencies/utils.py` (6).

They do not hurt BM25 much, but an embedding model with a 512-token limit would see only the start of each
one, which for these functions is mostly the parameter documentation, and would never see the body. Two
fixes to try in a later phase: split a function that is over the limit into its signature plus docstring
and its body, each with the function name as the heading path; or embed a shortened form of the chunk (the
name, the first docstring line and the signature) while keeping the full text for display. A separate
problem is that the AST chunker splits whole classes and whole example files into pieces (code intact stays
0.700 at every size), which parent-child retrieval (find a small chunk, return its parent) would address.
