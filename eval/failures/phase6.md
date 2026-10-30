# Phase 6 failures: hybrid rrf (k=60) + cross-encoder rerank

Dev questions only. A failure here is a question where the first chunk overlapping a gold span is not at rank 1 for `rrf+rerank` (11 of 40; hit@1 = 0.725). Source: `python scripts/list_failures.py --rerank --focus rrf+rerank --markdown`, searching 30 deep, rerank depth 30. A `-` means the gold is not in the top 30. For a question with two gold spans, the rank is that of the first hit on either span.

Causes use the Phase 4 words where they fit: **wrong chunk, right page**, **wrong chunk, other page**, **prose outranks code**, **sibling code**, **vocabulary mismatch**, **tiny chunk**. A new one: **label debatable**, a passage other than the gold also answers the question.

## The 11 failures

| Q | Category | bm25 | dense | rrf | minmax | rrf+rerank | Cause | What the text shows |
|---|---|---|---|---|---|---|---|---|
| q004 | lexical | 1 | 3 | 1 | 1 | 2 | label debatable | Gold is the tip in `simple-oauth2.md:64-70` that names `OAuth2PasswordRequestFormStrict`. The reranker's first result is `oauth2.py:59-159`, the `__init__` of `OAuth2PasswordRequestForm`. Its `grant_type` docstring says "If you want to enforce it, use instead the `OAuth2PasswordRequestFormStrict` dependency", which is the same answer. It is the chunk dense ranked first in Phase 4, where this question was filed as an exact-token miss. That reading is revised: the chunk contains the answer sentence. BM25 and RRF had the gold at rank 1; the reranker moved it to 2. |
| q022 | semantic | 7 | 1 | 2 | 2 | 2 | vocabulary mismatch | The gold (`sub-dependencies.md:57-72`) says "dependency", "cache" and `use_cache=False`. The question says "helper", "run once or repeatedly" and "force a fresh run". The reranker's first result is `deployment/concepts.md:185-189`, about running several processes (workers). Why the model scored it above the gold is not verified. |
| q026 | semantic | - | 10 | 26 | - | 16 | wrong chunk, right page | Gold is the section that sets `OPENAPI_URL` to an empty string (chunk `conditional-openapi.md:26-56`, gold 34-50). The first result is the first section of the same page (`5-24`), "Hiding your documentation user interfaces in production shouldn't be the way to protect your API". It matches the topic (docs, production) and not the intent ("without editing the code"). Same failure as Phase 4, where dense chose a different page. The reranker gives it -0.96 and the next best passage -5.0. The gold scores -8.8, rank 16 of 30. |
| q031 | code | - | 10 | 29 | 17 | 16 | prose outranks code | Gold is `is_body_allowed_for_status_code` in `fastapi/utils.py:26-40`, which lists the cases (`< 200`, 204, 205, 304). The first result is a tutorial note: "Some response codes (see the next section) indicate that the response does not have a body." It states the topic and does not say which codes. The code scores -2.43 and the note 6.39. BM25's top 30 does not contain the gold. |
| q037 | code | 1 | 2 | 1 | 1 | 3 | sibling code | Gold is `make_not_authenticated_error` (`oauth2.py:401-421`): 401, "Not authenticated", `WWW-Authenticate: Bearer`. The first result is `docs_src/security/tutorial003_py310.py:56-64`, the tutorial's own `get_current_user`, which raises an `HTTPException` with the same three values for an unknown user. It is the tutorial's error and not the OAuth2 class's. BM25 and RRF had the gold at rank 1; the reranker moved it to 3. |
| q041 | code | - | 6 | 20 | 10 | 16 | prose outranks code, other page | Gold is `_solve_generator` (`fastapi/dependencies/utils.py:566-574`), which picks `asynccontextmanager` or `contextmanager` by the kind of generator. The first result is `events.md:57-81`, "Async Context Manager", about the lifespan function. It shares the phrase "async context manager" and is about another feature. The question asks about "the code". |
| q043 | code | 16 | 4 | 6 | 7 | 7 | label debatable | Gold is the whole example `docs_src/handling_errors/tutorial004_py310.py:1-26`, where the `HTTPException` handler returns `PlainTextResponse`. The first result, `handling-errors.md:156-178`, is the docs section that embeds that example and says "you could want to return a plain text response instead of JSON for these errors". It answers the question in words. |
| q044 | code | - | - | - | - | - | label debatable, tiny chunk | The gold file is cut into 4 chunks of 15, 16, 46 and 53 tokens, one per function; none is in the top 30 of any row. The answer ("return `last_query`", from a cookie) is in lines 10-15. The docs prose that describes the example, `sub-dependencies.md:19-30`, says "If the user didn't provide any query `q`, we use the last query used, which we saved to a cookie before". That prose is minmax's first result. It describes `tutorial005_an_py310.py`, the Annotated version, and not the gold file `tutorial005_py310.py`. The reranker row's first result is `57-80`, about `use_cache`, which does not answer. |
| q049 | multi_chunk | 1 | 2 | 1 | 1 | 2 | wrong chunk, right page | Gold: the "context managers" section of `dependencies-with-yield.md:268-288` and `contextmanager_in_threadpool` in `concurrency.py:17-41`. The first result is the introduction of the same page (`1-24`), which says FastAPI uses `contextlib.contextmanager` and `asynccontextmanager` internally. It covers the first half of the question and not how the enter and exit of a sync one are run. The rank is that of the first gold span, so the second span is not measured here. |
| q055 | multi_chunk | 1 | 1 | 1 | 1 | 3 | wrong chunk, right page | Gold: `oauth2-jwt.md:149-159` (the function that creates the token) and `get-current-user.md:19-28` (how `get_current_user` receives the token from `oauth2_scheme`). The first result is the next section of the second page (`31-35`), about the utility function that turns a token into a user. It is adjacent to the gold and does not say how the token arrives. BM25, dense and RRF had a gold chunk at rank 1; the reranker moved it to 3. |
| q058 | multi_chunk | 20 | 2 | 5 | 4 | 4 | wrong chunk, other page | Gold: `response-model.md:133-141` (why FastAPI filters) and `routing.py:308-335` (`serialize_response`, which raises `ResponseValidationError`). The first result is `custom-response.md:19-27`, "JSON Responses": the data "will be automatically converted (and filtered) with any Pydantic `response_model`". It shares the topic and does not give the reason or say what happens when validation fails. |

## What the table says

| Cause | Questions |
|---|---|
| label debatable (another passage also answers) | q004, q043, q044 (3) |
| wrong chunk, right page | q026, q049, q055 (3) |
| prose outranks code | q031, q041 (2) |
| sibling code | q037 (1) |
| vocabulary mismatch | q022 (1) |
| wrong chunk, other page | q058 (1) |

1. **Only one gold is out of reach of the first stage.** In 10 of 11 failures the gold is inside the 30 passages the reranker sees. Only q044 is in no pool. For q026, q031 and q041 the gold sits at rank 16 of 30. The scores are close to their neighbours' (-8.6, -8.8, -9.0 for q026), and I checked that these ranks are real and not ties or a cap.
2. **Four of the five code questions lose to a docs page.** In q031, q041, q043 and q044 the first result is a tutorial `.md` page and the gold is a `.py` file. q037's first result is another code chunk.
3. **The reranker took rank 1 away from four questions RRF had right:** q004, q037, q049, q055. In each, the passage it preferred is related to the gold: a chunk with the same sentence (q004), a sibling (q037), the introduction of the same page (q049), the adjacent section (q055). Four examples, so this is a pattern to test, not a result. RRF has rank 1 on 26 questions and the reranker on 29, so the net is +3.
4. **Three labels have a second passage that answers the question** (q004, q043, q044). All three have the answer in the first result of some row. If any label is changed, every row's baseline changes and has to be re-run (the rule of Step 0). I have not changed any label.

## Decision on the three label candidates

q004, q043 and q044 are kept unchanged, so no baseline is re-run. This leaves a known limit: for these three the first result of some row also answers the question, so hit@1 on the dev set understates every retriever by up to three questions (0.075). The Phase 6 write-up says so next to the numbers.

## Not confirmed

- Why the model preferred each first result. I read the passages and the scores; I did not test any explanation.
- That the cross-encoder reads code badly. In q031 and q041 the gold code chunk scores below zero (-2.43, -1.18) and the docs pages score 6.39 and 4.06. In q026 the gold is prose and scores -8.8. Two code examples do not separate "bad at code" from "prefers the passage that matches the question's words".
- Whether the tiny function-sized chunks of q044 are the reason it is missed. They are 15 to 53 tokens each, against a mean of 133.
- Whether the reranker's lost questions (point 3) are one failure or four.
