# Candidate passages for the 60 questions

A menu of passages to write questions about. **I picked the passages; you write every question.** Each
entry is a file and an exact line range (checked by a script against the real files: every range exists
and is at most 60 lines). Say in your write-up: "candidate passages were suggested by a tool, and every
question was written by hand."

## How to use this list
1. Pick a passage you find interesting. Open it: `scripts\show_span.py <path> <start> <end>`.
2. **Read it**, plus a few lines around it. If you don't understand it, skip it and pick another.
3. Close it and write the question in **your own words** (see the rules in `docs/briefs/phase1.md`).
4. Use each passage for **at most one** question. Spread across files: don't take all questions from the
   same page.
5. Paths below are relative to the corpus folder (`data/raw/fastapi`) and are exactly what goes into the
   `"path"` field of `eval/questions.jsonl`. Keep the line range as given, or **tighten** it if only part
   of the passage answers your question (you may not make it longer than 60 lines).
6. The topic notes are deliberately plain labels. They are **not** question wording and they can be wrong
   about what a passage does in detail: trust the text, not my label.

## Quotas
| You write | From | Count |
|---|---|---|
| `lexical` | **D** passages: choose one where a *specific term* (a parameter, class, function or setting name in the heading) is the heart of the passage | 15 |
| `semantic` | **D** passages: choose one and ask it the way someone who *doesn't know the term* would; reword so you share few key words | 15 |
| `code` | **S** (source functions) and **X** (example files) | 15 |
| `multi_chunk` | **P** (pairs): the question must need *both* parts | 15 |

A good split for `code`: about 10 from **S** and 5 from **X**.

---

## D: documentation passages (for `lexical` and `semantic`)

#### `docs/en/docs/tutorial/first-steps.md`
- D01 `75-82`: the interactive API docs page
- D02 `91-94`: what OpenAPI is
- D03 `144-151`: what OpenAPI is used for
- D04 `314-361`: the path operation decorator
- D05 `362-389`: the path operation function

#### `docs/en/docs/tutorial/path-params.md`
- D06 `45-78`: data validation on path parameters
- D07 `111-130`: why the order of path operations matters
- D08 `135-150`: predefined values with an `Enum` class
- D09 `218-239`: path parameters that contain a path (the path convertor)

#### `docs/en/docs/tutorial/query-params.md`
- D10 `60-73`: optional query parameters
- D11 `74-112`: query parameter type conversion

#### `docs/en/docs/tutorial/query-params-str-validations.md`
- D12 `169-184`: regular expressions on a query parameter
- D13 `199-222`: required query parameters
- D14 `231-267`: a query parameter as a list (multiple values)
- D15 `327-346`: alias parameters
- D16 `347-360`: deprecating a parameter
- D17 `361-366`: excluding a parameter from OpenAPI

#### `docs/en/docs/tutorial/path-params-numeric-validations.md`
- D18 `94-101`: number validations, greater than or equal
- D19 `111-122`: number validations for floats

#### `docs/en/docs/tutorial/body.md`
- D20 `131-139`: request body together with path parameters
- D21 `140-163`: request body with path and query parameters

#### `docs/en/docs/tutorial/body-updates.md`
- D22 `29-46`: partial updates with PATCH
- D23 `47-58`: Pydantic's `exclude_unset` parameter

#### `docs/en/docs/tutorial/extra-models.md`
- D24 `159-174`: `Union` / `anyOf`

#### `docs/en/docs/tutorial/response-model.md`
- D25 `23-60`: the `response_model` parameter
- D26 `133-150`: return type and data filtering
- D27 `211-222`: disabling the response model
- D28 `237-264`: `response_model_exclude_unset`
- D29 `306-333`: `response_model_include` and `response_model_exclude`

#### `docs/en/docs/tutorial/response-status-code.md`
- D30 `42-72`: about HTTP status codes
- D31 `73-98`: a shortcut to remember the status names

#### `docs/en/docs/tutorial/handling-errors.md`
- D32 `30-43`: raising an `HTTPException`
- D33 `72-81`: custom headers on an error
- D34 `82-111`: installing custom exception handlers
- D35 `120-155`: overriding the request validation exceptions
- D36 `218-237`: FastAPI's `HTTPException` versus Starlette's

#### `docs/en/docs/tutorial/background-tasks.md`
- D37 `22-35`: creating a task function
- D38 `36-47`: adding the background task
- D39 `48-63`: background tasks and dependency injection
- D40 `76-83`: the caveat

#### `docs/en/docs/tutorial/middleware.md`
- D41 `22-51`: creating a middleware
- D42 `68-90`: the execution order of multiple middleware

#### `docs/en/docs/tutorial/cors.md`
- D43 `27-34`: wildcards
- D44 `35-68`: using `CORSMiddleware`
- D45 `69-74`: CORS preflight requests

#### `docs/en/docs/tutorial/dependencies/index.md`
- D46 `7-21`: what dependency injection is
- D47 `117-142`: sharing `Annotated` dependencies
- D48 `143-158`: `async` or not

#### `docs/en/docs/tutorial/dependencies/classes-as-dependencies.md`
- D49 `17-40`: what makes something a dependency
- D50 `41-91`: classes as dependencies

#### `docs/en/docs/tutorial/dependencies/sub-dependencies.md`
- D51 `57-88`: using the same dependency multiple times

#### `docs/en/docs/tutorial/dependencies/dependencies-with-yield.md`
- D52 `26-49`: a database dependency with `yield`
- D53 `96-115`: dependencies with `yield` and `HTTPException`
- D54 `134-186`: the execution order of dependencies with `yield`
- D55 `187-201`: early exit and `scope`

#### `docs/en/docs/tutorial/dependencies/dependencies-in-path-operation-decorators.md`
- D56 `11-38`: adding `dependencies` to the path operation decorator
- D57 `63-66`: dependencies for a group of path operations

#### `docs/en/docs/tutorial/security/first-steps.md`
- D58 `89-114`: the `password` flow
- D59 `115-168`: `OAuth2PasswordBearer`

#### `docs/en/docs/tutorial/security/get-current-user.md`
- D60 `19-30`: a `get_current_user` dependency

#### `docs/en/docs/tutorial/security/simple-oauth2.md`
- D61 `21-46`: the `scope` field
- D62 `51-86`: `OAuth2PasswordRequestForm`
- D63 `115-122`: password hashing
- D64 `153-184`: returning the token

#### `docs/en/docs/tutorial/security/oauth2-jwt.md`
- D65 `9-28`: about JWT
- D66 `97-130`: hashing and verifying passwords
- D67 `131-160`: handling JWT tokens
- D68 `179-200`: the JWT "subject" `sub`

#### `docs/en/docs/advanced/security/http-basic-auth.md`
- D69 `55-82`: timing attacks
- D70 `95-102`: the fix with `secrets.compare_digest()`

#### `docs/en/docs/advanced/settings.md`
- D71 `53-72`: creating the `Settings` object
- D72 `185-206`: reading a `.env` file
- D73 `230-256`: creating the `Settings` only once with `lru_cache`

#### `docs/en/docs/advanced/events.md`
- D74 `25-46`: lifespan
- D75 `87-102`: the deprecated alternative events

#### `docs/en/docs/advanced/websockets.md`
- D76 `43-56`: creating a websocket
- D77 `99-121`: using `Depends` with websockets
- D78 `153-180`: handling disconnections and multiple clients

#### `docs/en/docs/advanced/custom-response.md`
- D79 `29-34`: JSON performance
- D80 `149-173`: `RedirectResponse`
- D81 `174-197`: `StreamingResponse`
- D82 `198-218`: `FileResponse`
- D83 `255-270`: the default response class

#### `docs/en/docs/advanced/stream-data.md`
- D84 `21-28`: a `StreamingResponse` with `yield`

#### `docs/en/docs/tutorial/server-sent-events.md`
- D85 `72-81`: `ServerSentEvent`
- D86 `96-103`: resuming with `Last-Event-ID`

#### `docs/en/docs/advanced/additional-responses.md`
- D87 `194-211`: combining information

#### `docs/en/docs/advanced/response-directly.md`
- D88 `35-52`: using `jsonable_encoder` in a `Response`
- D89 `65-76`: how a response model works

#### `docs/en/docs/advanced/sub-applications.md`
- D90 `23-30`: mounting the sub-application

#### `docs/en/docs/how-to/configure-swagger-ui.md`
- D91 `11-26`: disabling syntax highlighting
- D92 `27-36`: changing the theme

#### `docs/en/docs/how-to/extending-openapi.md`
- D93 `68-77`: caching the OpenAPI schema

#### `docs/en/docs/how-to/custom-request-and-route.md`
- D94 `31-46`: a custom `GzipRequest` class

#### `docs/en/docs/how-to/conditional-openapi.md`
- D95 `26-56`: conditional OpenAPI from settings and env vars

#### `docs/en/docs/tutorial/testing.md`
- D96 `9-58`: using `TestClient`

#### `docs/en/docs/advanced/testing-dependencies.md`
- D97 `23-53`: the `app.dependency_overrides` attribute

#### `docs/en/docs/tutorial/bigger-applications.md`
- D98 `74-83`: `APIRouter`
- D99 `360-378`: avoiding name collisions
- D100 `519-526`: including the same router multiple times

---

## S: source-code passages (for `code`)
Read the function before you write a question about it. Questions can ask what it does, what it returns,
why it exists, or where a behaviour lives. Labels come from the function names.

#### `fastapi/background.py`
- S01 `12-67`: class `BackgroundTasks`
- S02 `41-62`: `BackgroundTasks.add_task`

#### `fastapi/exceptions.py`
- S03 `174-209`: class `ValidationException`
- S04 `212-221`: class `RequestValidationError`
- S05 `224-231`: class `WebSocketRequestValidationError`
- S06 `234-243`: class `ResponseValidationError`
- S07 `167-171`: class `DependencyScopeError`
- S08 `252-256`: class `FastAPIDeprecationWarning`

#### `fastapi/exception_handlers.py`
- S09 `11-17`: `http_exception_handler`
- S10 `20-26`: `request_validation_exception_handler`
- S11 `29-34`: `websocket_request_validation_exception_handler`

#### `fastapi/concurrency.py`
- S12 `17-41`: `contextmanager_in_threadpool`

#### `fastapi/encoders.py`
- S13 `59-81`: `decimal_encoder`

#### `fastapi/security/oauth2.py`
- S14 `401-421`: `OAuth2.make_not_authenticated_error`
- S15 `423-430`: `OAuth2.__call__`
- S16 `536-544`: `OAuth2PasswordBearer.__call__`
- S17 `642-650`: `OAuth2AuthorizationCodeBearer.__call__`
- S18 `653-693`: class `SecurityScopes`

#### `fastapi/security/api_key.py`
- S19 `11-52`: class `APIKeyBase`
- S20 `47-52`: `APIKeyBase.check_api_key`

#### `fastapi/security/http.py`
- S21 `16-26`: class `HTTPBasicCredentials`
- S22 `29-66`: class `HTTPAuthorizationCredentials`
- S23 `69-102`: class `HTTPBase`
- S24 `202-219`: `HTTPBasic.__call__`
- S25 `303-316`: `HTTPBearer.__call__`
- S26 `404-417`: `HTTPDigest.__call__`

#### `fastapi/utils.py`
- S27 `26-40`: `is_body_allowed_for_status_code`
- S28 `80-92`: `generate_operation_id_for_path`
- S29 `95-100`: `generate_unique_id`
- S30 `103-118`: `deep_dict_update`

#### `fastapi/datastructures.py`
- S31 `153-168`: class `DefaultPlaceholder`
- S32 `66-84`: `UploadFile.write`
- S33 `86-102`: `UploadFile.read`

#### `fastapi/responses.py`
- S34 `39-66`: class `UJSONResponse`
- S35 `69-98`: class `ORJSONResponse`

#### `fastapi/dependencies/utils.py`
- S36 `103-129`: `ensure_multipart_is_installed`
- S37 `132-144`: `get_parameterless_sub_dependant`
- S38 `169-195`: `get_flat_params`
- S39 `213-227`: `get_typed_signature`
- S40 `566-574`: `_solve_generator`
- S41 `869-885`: `is_union_of_base_models`
- S42 `912-948`: `_extract_form_body`
- S43 `951-998`: `request_body_to_args`
- S44 `1001-1048`: `_get_body_field`

#### `fastapi/openapi/utils.py`
- S45 `132-156`: `_get_openapi_security_definitions`
- S46 `231-263`: `get_openapi_operation_request_body`
- S47 `266-278`: `generate_operation_id`
- S48 `551-582`: `get_fields_from_routes`

#### `fastapi/applications.py`
- S49 `1054-1105`: `FastAPI.build_middleware_stack`
- S50 `1107-1140`: `FastAPI.openapi`
- S51 `1142-1195`: `FastAPI.setup`
- S52 `1229-1284`: `FastAPI.add_api_route`
- S53 `4747-4791`: `FastAPI.middleware`
- S54 `4793-4838`: `FastAPI.exception_handler`

#### `fastapi/routing.py`
- S55 `128-167`: `request_response`
- S56 `308-348`: `serialize_response`
- S57 `351-364`: `run_endpoint_function`
- S58 `787-831`: `get_websocket_app`
- S59 `6495-6527`: `APIRouter.on_event`

---

## X: example files (also for `code`)
Whole files from `docs_src/`, each one a small runnable example used by the docs. Open the file, see
what it shows, and ask something about it.

- X01 `docs_src/background_tasks/tutorial001_py310.py` `1-15`
- X02 `docs_src/background_tasks/tutorial002_py310.py` `1-24`
- X03 `docs_src/events/tutorial001_py310.py` `1-16`
- X04 `docs_src/events/tutorial003_py310.py` `1-28`
- X05 `docs_src/handling_errors/tutorial001_py310.py` `1-12`
- X06 `docs_src/handling_errors/tutorial004_py310.py` `1-26`
- X07 `docs_src/middleware/tutorial001_py310.py` `1-14`
- X08 `docs_src/dependencies/tutorial001_py310.py` `1-17`
- X09 `docs_src/dependencies/tutorial002_py310.py` `1-23`
- X10 `docs_src/dependencies/tutorial004_py310.py` `1-23`
- X11 `docs_src/dependencies/tutorial005_py310.py` `1-20`
- X12 `docs_src/security/tutorial001_py310.py` `1-11`
- X13 `docs_src/cors/tutorial001_py310.py` `1-24`
- X14 `docs_src/custom_response/tutorial004_py310.py` `1-23`
- X15 `docs_src/response_model/tutorial004_py310.py` `1-24`

---

## P: pairs for `multi_chunk`
A `multi_chunk` question must **need both passages**: write it so that one passage alone gives only half the
answer. Both spans go into the question's `"gold"` list. Where a pair's two passages share a file, they are
far enough apart to count as two places.

- P01 `docs/en/docs/tutorial/background-tasks.md:36-47` + `fastapi/background.py:41-62`
- P02 `docs/en/docs/tutorial/handling-errors.md:156-179` + `fastapi/exception_handlers.py:11-17`
- P03 `docs/en/docs/tutorial/handling-errors.md:120-155` + `fastapi/exception_handlers.py:20-26`
- P04 `docs/en/docs/tutorial/dependencies/dependencies-with-yield.md:258-290` + `fastapi/concurrency.py:17-41`
- P05 `docs/en/docs/tutorial/security/first-steps.md:115-168` + `fastapi/security/oauth2.py:536-544`
- P06 `docs/en/docs/tutorial/security/simple-oauth2.md:21-46` + `fastapi/security/oauth2.py:653-693`
- P07 `docs/en/docs/advanced/security/http-basic-auth.md:15-28` + `fastapi/security/http.py:202-219`
- P08 `docs/en/docs/advanced/security/http-basic-auth.md:29-54` + `docs/en/docs/advanced/security/http-basic-auth.md:95-102`
- P09 `docs/en/docs/tutorial/middleware.md:22-51` + `fastapi/applications.py:4747-4791`
- P10 `docs/en/docs/tutorial/handling-errors.md:82-111` + `fastapi/applications.py:4793-4838`
- P11 `docs/en/docs/how-to/extending-openapi.md:56-61` + `fastapi/applications.py:1107-1140`
- P12 `docs/en/docs/advanced/events.md:87-102` + `fastapi/routing.py:6495-6527`
- P13 `docs/en/docs/tutorial/dependencies/index.md:28-63` + `docs/en/docs/advanced/testing-dependencies.md:23-53`
- P14 `docs/en/docs/tutorial/security/oauth2-jwt.md:131-160` + `docs/en/docs/tutorial/security/get-current-user.md:19-30`
- P15 `docs/en/docs/advanced/settings.md:185-206` + `docs/en/docs/how-to/conditional-openapi.md:26-56`
- P16 `docs/en/docs/tutorial/bigger-applications.md:90-111` + `docs/en/docs/tutorial/bigger-applications.md:379-414`
- P17 `docs/en/docs/tutorial/body-updates.md:47-58` + `docs/en/docs/tutorial/response-model.md:237-264`
- P18 `docs/en/docs/tutorial/response-model.md:133-150` + `fastapi/routing.py:308-348`
- P19 `docs/en/docs/advanced/websockets.md:99-121` + `fastapi/routing.py:787-831`
- P20 `docs/en/docs/advanced/custom-response.md:174-197` + `docs/en/docs/advanced/stream-data.md:21-28`
- P21 `docs/en/docs/tutorial/cors.md:35-68` + `docs_src/cors/tutorial001_py310.py:1-24`
- P22 `docs/en/docs/tutorial/dependencies/classes-as-dependencies.md:41-91` + `docs_src/dependencies/tutorial004_py310.py:1-23`

Pairs P01-P03, P05-P07, P09-P12, P18, P19 pair a docs page with the code behind it; the rest pair two
documentation passages or a page with its example. A pair is only a *suggestion*: if you read both and a
good question doesn't come naturally, skip it.
