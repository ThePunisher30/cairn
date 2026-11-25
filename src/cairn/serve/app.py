"""The HTTP API (Phase 9).

GET  /health           liveness and readiness (no key)
GET  /metrics          Prometheus text
POST /search           retrieval only
POST /ask              an answer, as JSON or as a stream of server-sent events
POST /ingest           start a re-index job (202); GET /ingest/{id} reads its state
GET  /cache            what the caches hold now
"""

import hmac
import json
import logging
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from cairn.generation.errors import LLMError
from cairn.serve.cache import normalize_question
from cairn.serve.engine import Engine
from cairn.serve.observe import Metrics, ObservabilityMiddleware, RequestStats
from cairn.serve.protect import Overloaded, RateLimiter
from cairn.serve.schemas import (
    AskRequest,
    AskResponse,
    HitOut,
    SearchRequest,
    SearchResponse,
    answer_out,
    make_ref,
)
from cairn.serve.source import SourceError, read_source
from cairn.serve.state import FAILED

log = logging.getLogger("cairn.serve")

STATIC = Path(__file__).parent / "static"
NO_MODEL = "no language model is attached to this server, so these are the retrieved passages"
# The page loads only its own script and style and talks only to its own origin. Model output is
# shown with textContent (never as HTML), and this header is the second line of defence.
UI_CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
    "img-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


class SecurityHeaders:
    """Adds the headers a browser needs to treat the page carefully. Pure ASGI."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        is_page = scope["path"] == "/" or scope["path"].startswith("/ui")

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                headers.append((b"x-content-type-options", b"nosniff"))
                headers.append((b"referrer-policy", b"no-referrer"))
                if is_page:
                    headers.append((b"content-security-policy", UI_CSP.encode()))
                    headers.append((b"x-frame-options", b"DENY"))
            await send(message)

        await self.app(scope, receive, send_wrapper)


def sse(event, data):
    """One server-sent event. `json.dumps` puts the data on one line (newlines become \\n), which
    the format needs: a bare newline inside `data:` would end the event early."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def hit_out(h):
    return HitOut(
        path=h.path, start_line=h.start_line, end_line=h.end_line, score=h.score, text=h.text
    )


def new_metrics():
    m = Metrics()
    m.describe("cairn_requests_total", "HTTP requests by route and status")
    m.describe("cairn_request_seconds", "HTTP request duration")
    m.describe("cairn_stage_seconds", "time in each stage of answering")
    m.describe("cairn_tokens_total", "language model tokens used")
    m.describe("cairn_cost_usd_total", "estimated cost of language model calls")
    m.describe("cairn_llm_fallback_total", "answers replaced by retrieved chunks (model failed)")
    return m


def slot(engine):
    """The concurrency limit as a context manager (a no-op if the engine has none)."""
    return engine.limit if engine.limit is not None else nullcontext()


def create_app(service, *, api_key=None, rate_limit=None, metrics=None, blocking_demo=False):
    """`service` is a `Service`. `api_key` None turns authentication off (local use only).
    `rate_limit` is (requests per second, burst) per client, or None for no limit."""
    app = FastAPI(title="Cairn", version="0.9")
    metrics = metrics if metrics is not None else new_metrics()
    app.state.metrics = metrics
    limiter = RateLimiter(*rate_limit) if rate_limit else None
    app.state.limiter = limiter
    app.add_middleware(ObservabilityMiddleware, metrics=metrics)
    app.add_middleware(SecurityHeaders)
    if STATIC.is_dir():
        app.mount("/ui", StaticFiles(directory=STATIC, html=True), name="ui")

    def gauge(name, read):
        def value():
            return read(service.engine) if service.ready else 0

        metrics.gauge(name, value)

    gauge("cairn_ready", lambda e: 1)
    gauge("cairn_index_chunks", lambda e: e.live.n_alive if e.live else 0)
    gauge("cairn_index_tombstones", lambda e: e.live.n_dead if e.live else 0)
    gauge("cairn_index_generation", lambda e: e.generation)
    gauge("cairn_in_flight_retrievals", lambda e: e.limit.in_flight if e.limit else 0)
    gauge("cairn_waiting_retrievals", lambda e: e.limit.waiting if e.limit else 0)
    gauge("cairn_circuit_open", lambda e: int(e.breaker.state != "closed") if e.breaker else 0)
    gauge("cairn_exact_cache_entries", lambda e: len(e.exact_cache) if e.exact_cache else 0)

    @app.exception_handler(Overloaded)
    def overloaded(request, exc):
        metrics.inc("cairn_overloaded_total")
        return JSONResponse(
            {"detail": f"the service is busy: {exc}"}, status_code=503, headers={"Retry-After": "1"}
        )

    def require_key(x_api_key: str | None = Header(default=None)):
        if api_key is None:
            return
        given = (x_api_key or "").encode()
        if not hmac.compare_digest(given, api_key.encode()):  # constant time: no timing leak
            raise HTTPException(
                401, "missing or wrong API key", headers={"WWW-Authenticate": "ApiKey"}
            )

    KeyDep = Annotated[None, Depends(require_key)]

    def require_rate(request: Request, _: KeyDep):
        if limiter is None:
            return
        # one bucket per client address (behind a proxy, run uvicorn with --proxy-headers)
        who = request.client.host if request.client else "?"
        allowed, retry_after = limiter.check(who)
        if not allowed:
            metrics.inc("cairn_rate_limited_total")
            raise HTTPException(
                429,
                "too many requests",
                headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
            )

    def get_engine(_: Annotated[None, Depends(require_rate)]):
        if service.ready:
            return service.engine
        if service.status == FAILED:
            raise HTTPException(503, f"the service failed to start: {service.error}")
        raise HTTPException(503, f"not ready yet ({service.status})", headers={"Retry-After": "5"})

    EngineDep = Annotated[Engine, Depends(get_engine)]

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/ui/")

    @app.get("/health")
    def health():
        # no key needed: a load balancer asks this. 503 until ready, so it holds traffic back.
        body = service.describe()
        body["auth_required"] = api_key is not None
        if service.ready:
            engine = service.engine
            body["mode"] = {
                "retriever": engine.config.retriever,
                "llm": engine.config.llm,
                "model": engine.model,
            }
            if engine.live is not None:
                body["index"] = engine.live.stats()
        return JSONResponse(body, status_code=200 if service.ready else 503)

    @app.get("/metrics")
    def metrics_page(_: KeyDep):
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest, engine: EngineDep):
        stats = RequestStats()
        with stats.timed("retrieve"), slot(engine):
            hits = engine.retriever.search(req.query, req.k)
        record_stages(metrics, stats)
        return SearchResponse(hits=[hit_out(h) for h in hits])

    @app.post("/ask")
    def ask(req: AskRequest, engine: EngineDep):
        stats = RequestStats()
        if req.stream:
            return StreamingResponse(
                events(engine, req.question, stats, metrics),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        if engine.retrieval_only:
            response = fallback(engine, req.question, None, stats, metrics, message=NO_MODEL)
            record(metrics, stats, response)
            return response
        k = engine.config.k
        cached, vector = lookup(engine, req.question, k, stats)
        if cached is not None:
            record(metrics, stats, cached)
            return cached
        try:
            response = answer_out(engine.make_answerer(stats).ask(req.question))
        except LLMError as exc:
            response = fallback(engine, req.question, exc, stats, metrics)
        else:
            remember(engine, req.question, k, response, vector)
        record(metrics, stats, response)
        return response

    @app.get("/source")
    def source(
        engine: EngineDep,
        path: Annotated[str, Query(min_length=1, max_length=300)],
        start: Annotated[int, Query(ge=1)] = 1,
        end: Annotated[int | None, Query(ge=1)] = None,
    ):
        try:
            return read_source(engine.source_root, path, start, end)
        except SourceError as exc:
            raise HTTPException(exc.status, str(exc)) from exc

    @app.post("/ingest", status_code=202)
    def ingest(engine: EngineDep):
        if engine.jobs is None:
            raise HTTPException(501, "no corpus folder is configured: nothing to ingest")
        job, created = engine.jobs.submit()
        return {"created": created, "job": job.describe()}

    @app.get("/ingest/{job_id}")
    def ingest_status(job_id: str, engine: EngineDep):
        job = engine.jobs.get(job_id) if engine.jobs else None
        if job is None:
            raise HTTPException(404, "no such job (only the latest few are kept)")
        return job.describe()

    @app.get("/cache")
    def cache_stats(engine: EngineDep):
        exact, semantic = engine.exact_cache, engine.semantic_cache
        out = {"exact": None, "semantic": None}
        if exact is not None:
            out["exact"] = {"entries": len(exact), "hits": exact.hits, "misses": exact.misses}
        if semantic is not None:
            out["semantic"] = {
                "entries": len(semantic),
                "hits": semantic.hits,
                "misses": semantic.misses,
                "threshold": semantic.threshold,
                "recent_hits": [
                    {"question": q, "matched": m, "similarity": s}
                    for q, m, s in semantic.recent_hits
                ],
            }
        return out

    if blocking_demo:

        @app.post("/debug/search-blocking")
        async def search_blocking(req: SearchRequest, engine: EngineDep):
            # on purpose the wrong way: a CPU-heavy call straight on the event loop. Used by
            # bench/load_test.py to show the stall; scripts/serve.py never enables it by default.
            hits = engine.retriever.search(req.query, req.k)
            return SearchResponse(hits=[hit_out(h) for h in hits])

    return app


# --- helpers used by the routes ---------------------------------------------------------------


def record_stages(metrics, stats):
    for stage, seconds in stats.stages.items():
        metrics.observe("cairn_stage_seconds", seconds, stage=stage)


def record(metrics, stats, response):
    record_stages(metrics, stats)
    if response is None:
        return
    metrics.inc("cairn_tokens_total", response.input_tokens, kind="input")
    metrics.inc("cairn_tokens_total", response.output_tokens, kind="output")
    metrics.inc("cairn_cost_usd_total", response.cost_usd)


def lookup(engine, question, k, stats):
    """Looks in the exact cache, then the semantic one. Returns (response or None, the question's
    embedding if the semantic cache computed it, so `remember` can reuse it)."""
    scope = engine.scope(k)
    exact = engine.exact_cache
    if exact is not None:
        hit = exact.get((scope, normalize_question(question)))
        if hit is not None:
            stats.cache = "exact"
            return hit.model_copy(update={"cached": "exact"}), None
    semantic = engine.semantic_cache
    vector = None
    if semantic is not None:
        with slot(engine):
            vector = semantic.embed(question)
        found = semantic.lookup(question, scope, vector)
        if found is not None:
            stats.cache = "semantic"
            return found.value.model_copy(update={"cached": "semantic"}), vector
    return None, vector


def remember(engine, question, k, response, vector):
    if response.degraded:
        return  # never cache a failure: the model may be back a second later
    scope = engine.scope(k)
    if engine.exact_cache is not None:
        engine.exact_cache.put((scope, normalize_question(question)), response)
    if engine.semantic_cache is not None:
        engine.semantic_cache.store(question, scope, response, vector)


def fallback(engine, question, error, stats, metrics, *, message=None):
    """No answer from the model (it failed, or none is attached): reply with what retrieval
    found, and say so. A real failure is counted; "no model attached" is not a failure."""
    if message is None:
        metrics.inc("cairn_llm_fallback_total")
        message = f"the language model failed: {error}"
        log.warning("model failed, returning retrieved chunks: %s", error)
    stats.fallback = True
    with stats.timed("retrieve"), slot(engine):
        hits = engine.retriever.search(question, engine.config.k)
    return AskResponse(
        answer="",
        abstained=False,
        abstain_reason=None,
        truncated=False,
        citations=[],
        sources=[f"[{i}] {h.path}:{h.start_line}-{h.end_line}" for i, h in enumerate(hits, 1)],
        model=engine.model,
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        latency_ms=0.0,
        degraded=True,
        error=message,
        chunks=[hit_out(h) for h in hits],
        references=[make_ref(i, h) for i, h in enumerate(hits, 1)],
    )


def events(engine, question, stats, metrics):
    """The stream: `token` events, then one `done` event with the whole answer.

    Failures: if the model fails before the first token, a `fallback` event carries the retrieved
    chunks and `done` has `degraded: true`. After the first token the answer cannot be replaced,
    so it ends with one `error` event. The status line (200) was sent with the first byte, which
    is why a failure can only be an event. `done.answer` is the final text (invalid citations
    removed, or the abstain message): a client should show it in place of the tokens."""
    k = engine.config.k
    response = None
    try:
        if engine.retrieval_only:
            response = fallback(engine, question, None, stats, metrics, message=NO_MODEL)
            chunks = [c.model_dump() for c in response.chunks]
            yield sse("fallback", {"message": response.error, "chunks": chunks})
            yield sse("done", response.model_dump())
            return
        cached, vector = lookup(engine, question, k, stats)
        if cached is not None:
            response = cached
            yield sse("token", {"text": cached.answer})
            yield sse("done", cached.model_dump())
            return
        answerer = engine.make_answerer(stats)
        sent = False
        try:
            for piece in answerer.stream(question):
                sent = True
                yield sse("token", {"text": piece})
        except LLMError as exc:
            if sent:
                yield sse("error", {"message": f"the language model failed: {exc}"})
                return
            response = fallback(engine, question, exc, stats, metrics)
            chunks = [c.model_dump() for c in response.chunks]
            yield sse("fallback", {"message": response.error, "chunks": chunks})
            yield sse("done", response.model_dump())
            return
        response = answer_out(answerer.last_answer)
        remember(engine, question, k, response, vector)
        yield sse("done", response.model_dump())
    except Overloaded:
        metrics.inc("cairn_overloaded_total")
        yield sse("error", {"message": "the service is busy, try again"})
    except Exception:
        log.exception("stream failed")
        yield sse("error", {"message": "internal error"})
    finally:
        record(metrics, stats, response)
