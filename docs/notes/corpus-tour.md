# A tour of the corpus (FastAPI 0.143.0)

Written while reading the passages to draft the 60 questions. Everything here was observed in the files;
where I am guessing, it says so.

## What FastAPI is, in one paragraph
FastAPI is a Python library for building web APIs. You write ordinary Python functions, and the library
turns them into HTTP endpoints, checks the incoming data using the type hints, and produces interactive
documentation automatically. The documentation you are indexing is its official manual; the source is the
library itself.

## The vocabulary (so the passages make sense)
| Term | What it means |
|---|---|
| **path operation** | one endpoint: a URL path plus an HTTP method plus the function that handles it (`@app.get("/items/{id}")`) |
| **path parameter / query parameter** | a value taken from the URL path (`/items/5`) or from after the `?` (`?q=abc`) |
| **request body** | JSON sent by the client; described with a Pydantic model (a Python class that validates data) |
| **response model** | a type that filters and validates what you send back (`response_model=...`) |
| **dependency / `Depends`** | a function whose result is injected into your handler. Shared logic (database sessions, current user) lives here. "Dependency injection" |
| **`yield` dependency** | a dependency that runs setup, hands over a value, then runs cleanup after the request |
| **OAuth2 / JWT / bearer token** | login standards; the tutorial builds a login flow where the client sends a signed token in the `Authorization` header |
| **middleware** | code that wraps every request and response (timing, CORS headers...) |
| **CORS** | browser rules about web pages calling an API on a different domain; `CORSMiddleware` sets the headers |
| **lifespan** | code that runs once at app startup and once at shutdown (load a model, close a connection pool) |
| **background task** | work scheduled to run after the response has been sent |
| **WebSocket** | a long-lived two-way connection, for chat or live updates |
| **OpenAPI / Swagger UI** | the machine-readable description of your API (`/openapi.json`) and the interactive page generated from it (`/docs`) |
| **`APIRouter`** | a mini-app: lets you split endpoints into several files and attach them to the main app |

## How the repository is laid out
- `docs/en/docs/`: the English manual, 156 Markdown pages.
  - `tutorial/` (the step-by-step guide, with `dependencies/` and `security/` sub-folders),
    `advanced/` (the less common features), `how-to/` (recipes), `deployment/`, `reference/`
    (short pages listing the public classes).
  - skipped on purpose: `release-notes.md` (7,493 lines of changelog), translations, newsletter pages.
- `fastapi/`: the library, 52 Python files. The big ones are `routing.py` (6,527 lines), `applications.py`
  (4,838) and `param_functions.py` (2,460). The small, readable ones are `background.py`,
  `exception_handlers.py`, `concurrency.py`, `encoders.py`, and the `security/` package.
- `docs_src/`: 464 tiny runnable example programs (10-30 lines each) that the manual embeds.

## Things about this corpus that will matter for retrieval (observed)
These are the findings that make chunking a real design problem, not a formality.

1. **The docs do not contain their own code.** A manual page includes an example with a line like
   `{* ../../docs_src/path_params/tutorial003_py310.py hl[6,11] *}`. The code is in a separate file under
   `docs_src/`. So the words in an example (a parameter name that only appears in the code) are *not* in
   the Markdown text. A question whose key term only exists in the example file can't be answered from
   the page. I only used terms that appear in the Markdown itself for the `lexical` questions.
2. **The Markdown has its own mini-syntax.** Blocks like `/// tip ... ///`, `/// note | Technical Details`,
   tab groups (`//// tab | Python 3.10+`), `<dfn title="...">` and heading ids (`{ #add-the-background-task }`).
   A chunker that ignores this will put "tip" blocks and tab boundaries in odd places.
3. **Headings are good chunk boundaries**, and sections are mostly 10-60 lines. That is why the
   heading-aware chunker beat the fixed-size one in the demo, and why gold spans here are tight.
4. **Many near-duplicate examples.** `docs_src` has the same example in several variants (for
   instance `tutorial002_py310.py` and `tutorial002_an_py310.py`, with and without `Annotated`). The
   retriever will see several almost-identical candidates: expect ties.
5. **Source functions are long because of their docstrings.** Parameters are documented inside the
   signature with `Annotated[..., Doc("""...""")]`, so one function can be 100+ lines of mostly prose
   (`OAuth2PasswordBearer` is 112 lines). A chunker that splits by function gets long chunks that may exceed
   the embedding model's 512-token limit, and whose useful part (the `__call__` method) is only 9 lines.
6. **Probably repetitive code (not verified).** `APIRouter` is a 4,226-line class; I expect many methods
   with near-identical long docstrings (one per HTTP verb). Check when you reach the code chunker.

## How the 60 questions in `eval/questions.jsonl` were made
- **Passages** came from `docs/notes/candidate-passages.md` (a menu picked with the outline script).
- **The questions were drafted by Claude** after reading each passage, at your request, and the spans were
  checked by `tests/test_question_set.py` (file exists, lines in range, at most 60 lines, not the changelog).
- **Safeguard for bias:** `scripts/question_overlap.py` measures the share of a question's content words
  that also appear in its answer passage:

  | Category | Mean overlap |
  |---|---|
  | lexical | 0.55 |
  | semantic | 0.28 |
  | code | 0.44 |
  | multi_chunk | 0.55 |

  `semantic` questions were deliberately written in different words, and that shows in the number.
- **Limits you should state honestly:** questions written after reading a passage still tend to echo
  it somewhat; every question has exactly one "best" place chosen by the author; 60 questions is small
  (a difference of a few points between two methods is noise).
- **README wording suggestion:** "The 60 evaluation questions were drafted with AI assistance from
  passages chosen with a helper script, then reviewed and edited by me. Gold spans are verified by tests;
  question/passage word overlap is reported per category."
