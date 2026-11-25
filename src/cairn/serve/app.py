"""The HTTP API: /health, /search, /ask (JSON or a server-sent-event stream)."""

import hmac
import json
import logging
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from cairn.generation.errors import LLMError
from cairn.serve.engine import Engine
from cairn.serve.schemas import (
    AskRequest,
    HitOut,
    SearchRequest,
    SearchResponse,
    answer_out,
)
from cairn.serve.state import FAILED

log = logging.getLogger("cairn.serve")


def sse(event, data):
    """One server-sent event. `json.dumps` puts the data on one line (newlines become \\n), which
    the format needs: a bare newline inside `data:` would end the event early."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def create_app(service, *, api_key=None):
    """`service` is a `Service`. `api_key` None turns authentication off (local use only)."""
    app = FastAPI(title="Cairn", version="0.9")

    def require_key(x_api_key: str | None = Header(default=None)):
        if api_key is None:
            return
        given = (x_api_key or "").encode()
        if not hmac.compare_digest(given, api_key.encode()):  # constant time: no timing leak
            raise HTTPException(
                401, "missing or wrong API key", headers={"WWW-Authenticate": "ApiKey"}
            )

    def get_engine(_: Annotated[None, Depends(require_key)]):
        if service.ready:
            return service.engine
        if service.status == FAILED:
            raise HTTPException(503, f"the service failed to start: {service.error}")
        raise HTTPException(503, f"not ready yet ({service.status})", headers={"Retry-After": "5"})

    EngineDep = Annotated[Engine, Depends(get_engine)]

    @app.get("/health")
    def health():
        # no key needed: a load balancer asks this. 503 until ready, so it holds traffic back.
        return JSONResponse(service.describe(), status_code=200 if service.ready else 503)

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest, engine: EngineDep):
        hits = engine.retriever.search(req.query, req.k)
        return SearchResponse(
            hits=[
                HitOut(
                    path=h.path,
                    start_line=h.start_line,
                    end_line=h.end_line,
                    score=h.score,
                    text=h.text,
                )
                for h in hits
            ]
        )

    @app.post("/ask")
    def ask(req: AskRequest, engine: EngineDep):
        answerer = engine.make_answerer()  # one per request: it remembers `last_answer`
        if req.stream:
            return StreamingResponse(
                events(answerer, req.question),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            return answer_out(answerer.ask(req.question))
        except LLMError as exc:
            raise HTTPException(502, f"the language model failed: {exc}") from exc

    return app


def events(answerer, question):
    """The stream: `token` events, then one `done` event with the whole answer, or one `error`.

    The status line (200) is already sent when the first token goes out, so a failure after that
    can only be reported as an event. The `done` event's `answer` is the final text (invalid
    citations removed, or the abstain message); a client should show it in place of the tokens."""
    try:
        for piece in answerer.stream(question):
            yield sse("token", {"text": piece})
    except LLMError as exc:
        yield sse("error", {"message": f"the language model failed: {exc}"})
        return
    except Exception:
        log.exception("stream failed")
        yield sse("error", {"message": "internal error"})
        return
    yield sse("done", answer_out(answerer.last_answer).model_dump())
