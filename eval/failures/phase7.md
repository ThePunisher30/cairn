# Phase 7 failures: generation (gemma4:e4b, hybrid retriever, k=5, 1500-token budget)

Dev questions only. Run: `python scripts/gen_eval.py --llm ollama --model gemma4:e4b --name gemma4-hybrid --k 5 --max-context-tokens 1500`. Judge: `judge_v1` (the baseline), re-judged with `judge_v2` by `scripts/rejudge.py`. A question counts as a failure below when **both** judge prompts called the answer incorrect (9 of 40 answerable questions). "Gold in context" is whether a packed chunk overlaps a gold span (line-range overlap, `eval/questions.jsonl`). Causes that are a judgement from reading one answer say so.

Both judges are the same small model that wrote the answers, and the references are drafts the author has not fully verified (15 answers were graded by hand, see the end).

## The 9 failures

| Q | Category | Gold in context | Cause | What the answer shows |
|---|---|---|---|---|
| q026 | semantic | no | retrieval missed, model refused | The five chunks come from `conditional-openapi.md`, `fastapi-cli.md`, `custom-docs-ui-assets.md` (twice) and `openapi-callbacks.md`; none overlaps the gold span. The model answered `CANNOT_ANSWER`, which is the right behaviour for that context. It counts as a false abstention. |
| q031 | code | no | retrieval missed | Phase 4 had the same failure. The context holds `response-status-code.md`, `additional-status-codes.md` and `status.md`, not `is_body_allowed_for_status_code`. The answer lists `100-199` and `204` and never mentions `205`, `304`. |
| q041 | code | no | retrieval missed, answer admits it | The context holds general pages on context managers, not `_solve_generator`. The answer says outright that the chunks "do not explicitly detail" the sync/async split. Honest, but it should have said `CANNOT_ANSWER`: the model gave a partial answer where an abstention fitted. |
| q043 | code | no | retrieval missed | The question is about "the example that overrides both exception handlers". The context holds prose from `handling-errors.md` and an `exception_handlers.py` chunk. The answer says the example returns "plain text" and names `PlainTextResponse`, but does not name the handler's response type for `HTTPException` from the example itself (judgement from the answer text; not checked against the example file). |
| q047 | multi_chunk | yes (one of two) | context covers half | The answer gets the first half (override the handler, `PlainTextResponse`) and says the chunks do not state what the default handler sends. The reference's second half comes from `exception_handlers.py`, which is not among the five chunks. |
| q049 | multi_chunk | yes | answer incomplete | The answer covers the `with` / `async with` part and not how a sync context manager is run (`contextmanager_in_threadpool`). Whether the chunks hold that part was not checked. |
| q050 | multi_chunk | yes | answer misses the question | The answer lists methods of `WebSocket` (`receive_text`, `send_text`, `accept`) instead of the helpers (`Depends`, `Security`, `Cookie`, `Header`, `Path`, `Query`). It does get `WebSocketRequestValidationError` right. |
| q055 | multi_chunk | yes | false abstention, cause not investigated | Chunks from `get-current-user.md` and `tutorial004_an_py310.py` are in the context and overlap the gold, but the model answered `CANNOT_ANSWER`. |
| q059 | multi_chunk | no | retrieval missed, wrong answer | The context holds `bigger-applications.md` (318-354, 527-547, 130-154) and `separate-openapi-schemas.md`; no gold span. The answer says to include one `APIRouter` in another with `router.include_router(...)` and does not describe `app.include_router()`. The judge `v2` reason calls this a contradiction of the reference. |

## Counts

| Cause | Questions |
|---|---|
| Gold not in the context (retrieval) | q026, q031, q041, q043, q059 (5 of 9) |
| Gold in the context, answer incomplete or off target | q047, q049, q050 (3 of 9) |
| Gold in the context, false abstention | q055 (1 of 9) |

By category: multi_chunk 5 (q047, q049, q050, q055, q059), code 3 (q031, q041, q043), semantic 1 (q026). No lexical question failed under both judges. Both false abstentions (q026, q055) are the only questions the model refused; all 10 unanswerable questions were refused.

The retrieval ceiling matters more than the prompt here: 6 of the 40 answerable questions (15%) had no gold chunk in the five that were packed, so correctness could not exceed about 0.85 with this context (`scripts/context_recall.py`, hybrid, k=5).

## Where the judges and the hand grading disagree

Answers a judge flagged and a human reading did not (from the 15 hand-graded answers, see below): `q010` (`v1` said unfaithful although its own reason admits chunk 2 states the default), `q041` and `q050` (`v1` and `v2` said unfaithful; the answers only restate or misplace what the chunks say), `q058`, `q017` (`v1` said correct, graded incorrect by hand for a missing part). Under `judge_v2`, `q040` and `q052` became "incorrect" although the answers hold the reference's key fact.

## Hand grading

15 answers were graded blind to the judge's verdicts. The grader was Claude, not the author; the author's own check of a few is still open. Agreement with `judge_v1`: correct 0.867 (kappa 0.667), faithful 0.733 (kappa 0.250). With `judge_v2`: correct 0.867 (kappa 0.727), faithful 0.800 (kappa -0.098). Only one of the 15 was unfaithful by that grading, so the faithful kappas are close to meaningless. `judge_v2` was written after seeing the `v1` disagreements on the same 15, so its agreement is optimistic.
