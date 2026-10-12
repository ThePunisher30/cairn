# Reading list for the 60 questions

Tick the boxes as you go. You are **not** studying FastAPI: you are reading enough pages to write questions
and to find the exact lines that answer them. Write questions *while* you read (see the rules in
`docs/briefs/phase1.md`, Step 5).

All paths are inside `data/raw/fastapi/`. Line counts are in brackets.

## Tools (use these every time)
```powershell
# the outline of a file, with the line range of every section / function:
C:\mrv\Scripts\python scripts\outline.py docs/en/docs/tutorial/background-tasks.md
C:\mrv\Scripts\python scripts\outline.py fastapi/routing.py --grep route

# look at exact lines (check a gold span before you save it):
C:\mrv\Scripts\python scripts\show_span.py docs/en/docs/tutorial/background-tasks.md 36 47

# check your question file as you go:
C:\mrv\Scripts\python -m pytest tests\test_question_set.py -q
```
The outline of a doc gives you ready-made, tight spans (one section each). The outline of a source file
gives you every function with its start and end line.

## The target: 60 questions
| Category | Count | Written in |
|---|---|---|
| `lexical` | 15 | sessions 1-5 (3 per session) |
| `semantic` | 15 | sessions 1-5 (3 per session) |
| `code` | 15 | session 6 |
| `multi_chunk` | 15 | session 7 |

Splits: in each category the 3rd, 6th, 9th, 12th and 15th question is `"test"`, the rest `"dev"`.

---

## Session 1: the basics (about 2 h) -> 3 lexical + 3 semantic
- [ ] `docs/en/docs/tutorial/first-steps.md` [427] (skim the code, read the explanations)
- [ ] `docs/en/docs/tutorial/path-params.md` [251]
- [ ] `docs/en/docs/tutorial/query-params.md` [188]
- [ ] `docs/en/docs/tutorial/query-params-str-validations.md` [450] (skim: lots of small variations)
- [ ] `docs/en/docs/tutorial/body.md` [166]

*Optional:* copy the example from `first-steps.md`, run it, open `/docs` in the browser (20 min).

## Session 2: requests and responses (about 2 h) -> 3 lexical + 3 semantic
- [ ] `docs/en/docs/tutorial/response-model.md` [344]
- [ ] `docs/en/docs/tutorial/response-status-code.md` [101]
- [ ] `docs/en/docs/tutorial/handling-errors.md` [244]
- [ ] `docs/en/docs/tutorial/background-tasks.md` [86]
- [ ] `docs/en/docs/tutorial/middleware.md` [95]
- [ ] `docs/en/docs/tutorial/cors.md` [89]

## Session 3: dependencies (about 2 h) -> 3 lexical + 3 semantic
- [ ] `docs/en/docs/tutorial/dependencies/index.md` [250]
- [ ] `docs/en/docs/tutorial/dependencies/classes-as-dependencies.md` [288]
- [ ] `docs/en/docs/tutorial/dependencies/sub-dependencies.md` [105]
- [ ] `docs/en/docs/tutorial/dependencies/dependencies-with-yield.md` [290]
- [ ] `docs/en/docs/tutorial/dependencies/dependencies-in-path-operation-decorators.md` [69]

## Session 4: security (about 2 h) -> 3 lexical + 3 semantic
- [ ] `docs/en/docs/tutorial/security/first-steps.md` [203]
- [ ] `docs/en/docs/tutorial/security/get-current-user.md` [105]
- [ ] `docs/en/docs/tutorial/security/simple-oauth2.md` [289]
- [ ] `docs/en/docs/tutorial/security/oauth2-jwt.md` [277]
- [ ] `docs/en/docs/advanced/security/http-basic-auth.md` [107]
- [ ] `docs/en/docs/advanced/security/oauth2-scopes.md` [274] (skim)

## Session 5: advanced and how-to (about 2 h) -> 3 lexical + 3 semantic
- [ ] `docs/en/docs/advanced/settings.md` [326]
- [ ] `docs/en/docs/advanced/events.md` [165]
- [ ] `docs/en/docs/advanced/websockets.md` [186]
- [ ] `docs/en/docs/advanced/custom-response.md` [273]
- [ ] `docs/en/docs/advanced/additional-responses.md` [247]
- [ ] `docs/en/docs/how-to/configure-swagger-ui.md` [70]
- [ ] `docs/en/docs/how-to/extending-openapi.md` [88]
- [ ] `docs/en/docs/how-to/custom-request-and-route.md` [109]
- [ ] `docs/en/docs/deployment/server-workers.md` [139] (skim)

## Session 6: the source code (about 3 h) -> 15 `code` questions
Use `outline.py` first, then read only the functions that look interesting.
- [ ] `fastapi/background.py` [67] (small: read it all, it pairs with the background-tasks doc)
- [ ] `fastapi/exceptions.py` [256]
- [ ] `fastapi/exception_handlers.py` [34]
- [ ] `fastapi/concurrency.py` [41]
- [ ] `fastapi/encoders.py` [366] (`jsonable_encoder`)
- [ ] `fastapi/security/oauth2.py` [693] (the classes, not every docstring)
- [ ] `fastapi/security/http.py` [417] and `fastapi/security/api_key.py` [320] (skim)
- [ ] `fastapi/dependencies/utils.py` [1053] (outline: how dependencies are resolved)
- [ ] `fastapi/openapi/utils.py` [679] (outline: how the OpenAPI schema is built)
- [ ] `fastapi/routing.py` [6527] (outline only: `APIRoute`, `APIRouter`, the request handler)
- [ ] `fastapi/applications.py` [4838] (outline only: the `FastAPI` class)
- [ ] `fastapi/param_functions.py` [2460] and `fastapi/params.py` [754] (outline only: `Query`, `Path`, `Depends`)
- [ ] a few examples in `docs_src/` (for instance `docs_src/background_tasks/`, `docs_src/dependencies/`):
      questions like "which example shows X?"

Typical `code` questions: "Where is X implemented?", "What does function Y return?", "Which class
handles Z?", "How does FastAPI decide which exception handler to use?" The gold is the function or class
(use the outline's line range, but trim it if it is longer than 60 lines).

## Session 7: questions that need two places (about 2-3 h) -> 15 `multi_chunk` questions
A `multi_chunk` question needs **two or more separate passages** to answer fully. List each passage as its
own gold span. Ways to find them as you read:
- a **docs page + the source that implements it** (for example: the background-tasks doc section "Add the
  background task" + the `add_task` method in `fastapi/background.py`)
- **two docs pages** that each hold a piece (for example: declaring a dependency, and testing it with an
  override)
- a **docs page + the example code** in `docs_src/`
- **two sections of the same long page** that are far apart (more than 60 lines apart)

Mark candidates while you read in sessions 1-6 (a note "pairs with ..." costs you nothing), then
turn them into questions here.

---

## Habits that make it faster
1. Read a section, then **close the page and write the question from memory** in your own words. That
   naturally gives `semantic` questions (your words, not the page's).
2. For `lexical` questions, keep one exact term from the page (a parameter, class or function name).
3. Write the gold span down immediately (file, start, end) from `outline.py` or `show_span.py`.
4. Save the question into `eval/questions.jsonl` right away and run the question-set tests.
5. After each session, paste me 5 questions and I will review them for ambiguity before you continue.
