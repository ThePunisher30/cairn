# Phase 8 failures: the agent loop (gemma4:e4b, hybrid retriever, k=5, `agent_v1`)

Dev questions only. Runs: `python scripts/agent_eval.py --mode agent --set multihop|dev --llm ollama --model gemma4:e4b` and the same with `--mode single` (numbers in `docs/results.md`, section "Phase 8"). Judge `judge_v1`, the same small model that answers. The traces were read in full for all 10 multi-hop runs; for the 11 single-hop questions the agent lost to single-shot the queries, answers and judge reasons were read; for the other single-hop runs only the tool-call shape, the evidence seen and the start of the answer. Where a cause is a judgement from one answer it says so; a cause marked **untested** was not tried.

The judge's written reasons do not always match the answer (m09, m10 below), and the references are drafts, so a "failure" is "the judge called it wrong", checked by reading the answer against the reference.

## Multi-hop dev questions the agent got wrong (6 of 10)

| Q | Tool calls | Evidence seen | What happened |
|---|---|---|---|
| m01 | search, search | 0 of 2 | Two generic queries ("yield dependency thread cleanup") found `dependencies-with-yield.md` and tutorials, not `fastapi/dependencies/utils.py` or `fastapi/concurrency.py`. The answer says the docs do not mention the helper: honest, but it should have said `CANNOT_ANSWER`, and the second search was not built from anything the first found. |
| m04 | none | 0 of 2 | Answered `CANNOT_ANSWER` at step 1 with no search. Single-shot answered it correctly: this is the one question one search solves. |
| m07 | search, read_chunk | 0 of 2 | The chunk it read ends at `oauth2.py` line 664, thirteen lines before `scope_str` (677). The answer says the chunk "does not explicitly state what its string attribute contains". `read_file` on that file would have shown it; the tool was never used. Single-shot got this one right. |
| m08 | search, read_chunk | 1 of 2 | It found `python-multipart` and answered with the docs' command (`uv add python-multipart`). **The question is ambiguous**: it asks what FastAPI "tells you to run when it is missing", and the code's error message says `pip install python-multipart`. The agent's answer is defensible; the reference came from the code. A flaw in my question, not in the agent. |
| m09 | search, read_chunk, read_chunk, search | **2 of 2** | Both gold chunks were in what it saw. The first paragraph of the answer is about the `Response` class (wrong mechanism); the second paragraph says `http_exception_handler` uses the exception's `headers` (right). The judge's reason is garbled. It is a reading failure, not a retrieval failure. |
| m10 | search, read_chunk, search, read_chunk, read_chunk | 1 of 2 | The second search ("jsonable_encoder timedelta conversion") was built from the name found by the first, which is the intended multi-hop behaviour, and still did not surface `fastapi/encoders.py`. The answer says the snippets do not detail `timedelta`. The judge's reason says the answer "incorrectly states that it converts a `timedelta`", which the answer does not say. The agent never opened `encoders.py` with `read_file`. |

Where the agent won and single-shot did not: **m02** (a second search, "default exception handler RequestValidationError status code body", built from the exception name found first, reached `exception_handlers.py`; single-shot got the 422 but added a wrong detail) and **m03** (it read one more chunk and named `OAuth2PasswordBearer`; single-shot named `HTTPBearer` from a higher-ranked chunk).

## The single-hop questions the agent got wrong and single-shot got right (11 of 37)

| Cause | Questions | Count |
|---|---|---|
| **No tool call at all**: refused or gave up at step 1 | q022, q025, q035 (`CANNOT_ANSWER`), q019 ("I am sorry, but I cannot find ...", which the abstention check does not recognise) | 4 |
| **Invented `path_filter`**: three searches with `examples/security`, `examples`, `security`, each "No results." | q005 | 1 |
| **Searched with its own query, gold not in what it saw** (evidence seen 0.0), the answer says the results do not contain it | q016, q023, q028 | 3 |
| **Saw the gold chunk, answer incomplete or wrong** (evidence seen 1.0); one `search` and no `read_chunk` in q017 and q038, one `read_chunk` in q001 | q001, q017, q038 | 3 |

- **The queries.** For q016 the agent searched "FastAPI route matching fixed vs ID", for q028 "stop api description from rebuilding docs": its own terse rewrites of questions that deliberately avoid the docs' wording. Single-shot searches with the question text itself and saw the gold span in all 11 questions; the agent saw all of it in 3 and none in 8. **Untested:** that the rewriting is the cause (a run whose first search is the question text would test it).
- **Snippets.** In q017 (omits `on`, `yes`), q038 ("did not find ... the colon") and q040 (never states that every member must be a `BaseModel`) the judge says the missing detail is in the chunks, and each run was one `search` followed by an answer. Search results show only the first 400 characters of a chunk, so a possible cause is that the answer was written from a cut snippet. **Untested.**
- **The 4 no-tool runs** include a question (q035, `decimal_encoder`) whose name is in the question text, so a search could hardly have failed. Why the model refuses before searching, when the prompt says to search first, was not investigated.

## Questions both modes got wrong (5 of 37) and the agent-only wins (5)

- **Both wrong:** q026, q040, q041, q049, q050 (q026, q041, q049, q050 are also in `eval/failures/phase7.md`; q040 was judged correct by `judge_v1` in the Phase 7 run).
- **Agent only:** q007, q031, q043, q047, q055. q031, q043, q047 and q055 are also failures of single-shot in Phase 7 (`eval/failures/phase7.md`), and single-shot's own verdict on q007 flipped between the Phase 7 run and this one, so with two noisy runs a "win" can be a different draw of the same question.

## Counts

| Cause (agent wrong, 17 of 40 dev; see the first table for multi-hop) | Questions |
|---|---|
| No tool call | q019, q022, q025, q035 (4) |
| Invented `path_filter` | q005 (1) |
| Searched, gold not seen | q016, q023, q026, q028, q041, q050, q058 (7) |
| Saw the gold (q049 only one of its two spans), incomplete or wrong | q001, q017, q038, q040, q049 (5) |

q026, q040, q041, q049, q050 and q058 were not read as carefully as the 11 above; their row in this table is from the tool-call shape, the evidence seen and the start of the answer.

## What these failures say, and what they do not

- The agent's worst failure here was **not using its tools** (5 of 50 runs) and **not using the right query** (its rewrite replaced the question text). Neither is a failure of the loop, the budgets or error handling, which the real model never stressed (no tool errors, no repeats, no budget stops in 82 calls).
- It never used `read_file` or `list_symbols`, the two tools that would have fixed m07 and m10.
- A better prompt (`agent_v2`: search first, quote the question in the first search, use `read_file` when a chunk stops short) is the obvious next experiment. It was **not run**.
- The reasons for q017, q038, q040 (snippet length) and q016, q023, q028 (query rewriting) are guesses until tested.
