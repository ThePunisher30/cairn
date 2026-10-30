# Phase 4 failures: dense retrieval (bge-small, structure-300, trained query prefix)

Dev questions only. A failure here is a question where the first chunk overlapping a gold span is not at rank 1 (16 of 40). Source: `python scripts/compare_retrievers.py --model bge-small --fail-rank 1`, then the gold span and the chunk dense ranked first were read in the corpus. "BM25" is the rank BM25 gave the same question in Phase 3.

Causes used: **wrong chunk** (a chunk that does not hold the answer outranks the gold chunk), **exact-token miss**, **vocabulary mismatch**, **too generic**, **answer cut across chunks**, **duplicate chunk**, **label error**. Where a cause is a judgement from reading one passage, it says so.

## The 16 failures

| Q | Category | Dense | BM25 | Cause | What the text shows |
|---|---|---|---|---|---|
| q004 | lexical | 3 | 1 | exact-token miss | The question names `OAuth2PasswordRequestForm` and `grant_type`; the gold is a 7-line tip in `simple-oauth2.md` that names `OAuth2PasswordRequestFormStrict`. Dense ranked the source of `oauth2.py` (59-159, a long chunk with `grant_type` in it) first. BM25 gets the tip at rank 1 because both names are rare. |
| q005 | lexical | 2 | 1 | wrong chunk, right page (label possibly debatable) | Dense's first result is a one-line chunk on the same page (`http-basic-auth.md:53`) that says `secrets.compare_digest()` protects against timing attacks. That is arguably also an answer to the question, so the gold (95-102) may not be the only good passage. Not verified further. |
| q007 | lexical | 3 | 1 | wrong chunk, right page (label possibly debatable) | Dense's first result is `cors.md:56-60`, the list of `allow_*` parameters, which also answers "what can CORSMiddleware be told to allow". The gold (35-47) lists credentials, methods and headers. Same doubt as q005. |
| q019 | semantic | 7 | 5 | wrong chunk, right page | The question paraphrases the "Caveat" section (use a queue like Celery instead of `BackgroundTasks`). Dense's first result is the page's introduction (`background-tasks.md:1-12`). |
| q026 | semantic | 10 | 1217 | wrong chunk, other page | Gold: set `OPENAPI_URL` to an empty string, no code change. Dense's first result (`custom-docs-ui-assets.md:15-21`) says to set the docs URLs to `None` when creating the app, which does require editing code. The question's "without editing the code" was ignored. |
| q028 | semantic | 38 | 1 | wrong chunk, other page | Gold (`extending-openapi.md:68-76`, "Cache the OpenAPI schema") says the schema is generated "every time a user opens your API docs". Dense's first result is `conditional-openapi.md:5-24`, about hiding the docs. Dense matched the topic (the docs) and missed the intent (caching). |
| q031 | code | 10 | 188 | wrong chunk, docs outrank code | Gold is `is_body_allowed_for_status_code` in `fastapi/utils.py`. Dense's first result is `response-status-code.md:32-40`, which says some status codes have no body but does not list them. |
| q037 | code | 2 | 1 | wrong chunk, sibling code | Gold is `make_not_authenticated_error` in `oauth2.py`; dense's first result is the method with the same name in `api_key.py:31-45`, which has a similar shape and docstring. |
| q041 | code | 6 | 136 | wrong chunk, tiny chunk | Gold is `_solve_generator` in `fastapi/dependencies/utils.py`. Dense's first result is `dependencies-with-yield.md:238-238`, a chunk that is only the heading `## Context Managers`. Check how many chunks are only a heading. |
| q043 | code | 4 | 16 | wrong chunk, prose outranks example | The question asks about "the example that overrides both exception handlers". Gold is the example file `tutorial004_py310.py`; dense's first result is the prose that introduces it (`handling-errors.md:112-118`). |
| q044 | code | 52 | 972 | wrong chunk, prose outranks example | Same pattern as q043, worse: the gold is the 20-line example `tutorial005_py310.py`, and dense's first result is the prose of `sub-dependencies.md`. Both retrievers fail. |
| q047 | multi_chunk | 2 | 1 | wrong chunk, right file | Two gold spans (the tutorial section and `exception_handlers.py`). Dense's first result is the first six lines of the example `tutorial004_py310.py` (imports only), not the part that answers. |
| q049 | multi_chunk | 2 | 1 | wrong chunk, right page | Dense's first result is "What are Context Managers" (`dependencies-with-yield.md:240-256`), a general explanation of Python's `with`; the gold on that page is 268-288. |
| q050 | multi_chunk | 17 | 4 | wrong chunk, too generic | The first gold chunk (`websockets.md:101-110`) is a bare list (`Depends`, `Security`, `Cookie`, ...). Dense's first result is an unrelated page about validators that mentions "validation" and "Dependencies" (`query-params-str-validations.md:391-397`). |
| q058 | multi_chunk | 2 | 20 | wrong chunk, right page | Dense's first result is `response-model.md:147-149`, next to the gold (133-141) on the same page. |
| q059 | multi_chunk | 2 | 47 | wrong chunk, right page | Dense's first result is `bigger-applications.md:130-154`; both gold spans are on the same page (90-100 and 393-395). |

## Counts

| Cause | Questions |
|---|---|
| wrong chunk, right page or file | q005, q007, q019, q047, q049, q058, q059 (7) |
| wrong chunk, other page | q026, q028, q050 (3) |
| wrong chunk, prose outranks the example or code | q031, q043, q044 (3) |
| wrong chunk, sibling code or a tiny chunk | q037, q041 (2) |
| exact-token miss | q004 (1) |
| answer cut across chunks, duplicate chunk, vocabulary mismatch, label error | none confirmed (see below) |

What the table says:

1. **Dense rarely misses the right page; it picks a neighbouring chunk.** In 7 of 16 failures the first result is on the same page or file as the gold. The chunks of one page share vocabulary, so the embeddings are close, and the gold chunk has no clear lead over its neighbours.
2. **7 of the 16 are questions BM25 gets at rank 1** (q004, q005, q007, q028, q037, q047, q049). The other 9 are failures for both (q019, q026, q031, q041, q043, q044, q050, q058, q059). Those 9 are what hybrid search cannot fix by itself.
3. **Two failures have a debatable label** (q005, q007): the chunk dense put first also answers the question. Before Phase 6, look at the other labelled spans for the same problem. If it is real, the labels, not dense, are wrong for these two.
4. **Not confirmed:** I did not check whether the gold chunk of q028 is too short to be matched, or how many chunks are only a heading (q041), or why prose beats the example in q043 and q044. These are ideas to test, not findings.

## Where dense fixed a BM25 failure (not failures)

BM25 ranks the answer below 1 and dense ranks it at 1 for q016, q017, q020, q022, q023, q032 and q053 (7 questions, `semantic` for the first five). q017 is the clearest case: BM25 rank 212, dense rank 1, because (Phase 3) the passage contains `true` but none of the content words of the question. In q053 dense's first chunk (`applications.py:4747-4791`) contains the gold lines 4787-4791 inside a larger chunk, so it counts as a hit.

## Label check before Phase 6 (q005, q007)
Kept both labels unchanged. I read the gold span and the chunk dense ranked first for each.
- q005: the gold explains the fix; dense's one-line chunk only names the attack. The gold is the fuller answer, so I kept it.
- q007: dense's chunk is the argument list, which also answers it, but the gold is the passage the question was written from.