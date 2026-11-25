import hashlib
import json
import logging
import threading
import time

import numpy as np
from fastapi.testclient import TestClient

from cairn.generation.errors import ServerError
from cairn.generation.fake import FakeLLM
from cairn.retrieval.hybrid import HybridRetriever
from cairn.serve.app import create_app
from cairn.serve.cache import LRUCache, SemanticCache
from cairn.serve.engine import Engine, EngineConfig
from cairn.serve.indexer import Indexer
from cairn.serve.jobs import JobRunner
from cairn.serve.live import LiveBM25, LiveIndex
from cairn.serve.manifest import Manifest
from cairn.serve.observe import JsonFormatter, Metrics, request_id_var
from cairn.serve.protect import CircuitBreaker, ConcurrencyLimit, GuardedLLM
from cairn.serve.state import Service
from cairn.types import Hit

DIM = 16


class Embedder:
    dim = DIM

    def _vector(self, text):
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
        v = np.random.default_rng(seed).normal(size=DIM).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_documents(self, texts):
        return np.array([self._vector(t) for t in texts], dtype=np.float32)

    def embed_query(self, text):
        return self._vector(text)


class Retriever:
    def __init__(self, delay=0.0):
        self.delay = delay
        self.calls = 0
        self.started_at = None

    def search(self, query, k):
        self.calls += 1
        self.started_at = time.perf_counter()
        time.sleep(self.delay)
        return [Hit("docs/cors.md", 1, 5, 0.9, "Use CORSMiddleware to allow other origins.")][:k]


def make_engine(llm=None, retriever=None, **kwargs):
    llm = llm if llm is not None else FakeLLM(responder=lambda s, m: "Use CORSMiddleware [1].")
    return Engine(
        retriever or Retriever(),
        llm,
        "fake-model",
        EngineConfig(data_dir="."),
        None,
        exact_cache=kwargs.pop("exact_cache", LRUCache(50)),
        **kwargs,
    )


def client_for(engine, **options):
    service = Service(lambda: engine)
    service.start()
    assert service.wait(5)
    return TestClient(create_app(service, **options))


def ask(client, question="How do I enable CORS?", **extra):
    return client.post("/ask", json={"question": question, **extra})


def events(text):
    out = []
    for block in text.strip().split("\n\n"):
        name, data = block.split("\n")
        out.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


# --- exact cache ---------------------------------------------------------------------------


def test_the_same_question_twice_is_answered_from_the_cache_without_the_model():
    llm = FakeLLM(["Use CORSMiddleware [1]."])  # one reply only: a second call would fail loudly
    retriever = Retriever()
    client = client_for(make_engine(llm, retriever))
    first = ask(client).json()
    second = ask(client, "  how do i ENABLE cors ").json()  # same question, other spelling
    assert first["cached"] is None and second["cached"] == "exact"
    assert second["answer"] == first["answer"] and second["citations"] == first["citations"]
    assert len(llm.calls) == 1 and retriever.calls == 1


def test_a_different_k_or_model_does_not_share_cache_entries():
    engine = make_engine()
    assert engine.scope(5) != engine.scope(8)


def test_a_cached_answer_is_streamed_as_one_token_and_done():
    client = client_for(make_engine())
    ask(client)
    response = ask(client, stream=True)
    names = [n for n, _ in events(response.text)]
    assert names == ["token", "done"]
    assert events(response.text)[-1][1]["cached"] == "exact"


def test_a_degraded_answer_is_never_cached():
    llm = FakeLLM([ServerError("down"), "Use CORSMiddleware [1]."])
    client = client_for(make_engine(llm))
    assert ask(client).json()["degraded"] is True
    second = ask(client).json()  # the model is back: a real answer, not the failure replayed
    assert second["degraded"] is False and second["answer"] == "Use CORSMiddleware [1]."


# --- semantic cache ------------------------------------------------------------------------


class TableEmbedder:
    TABLE = {
        "how do i enable cors?": [1, 0, 0],
        "how can i turn on cors?": [0.98, 0.2, 0],
        "what is a path parameter?": [0, 1, 0],
    }

    def embed_query(self, text):
        v = np.array(self.TABLE[text.lower()], dtype=np.float32)
        return v / np.linalg.norm(v)


def test_a_paraphrase_is_served_from_the_semantic_cache_and_recorded():
    llm = FakeLLM(["Use CORSMiddleware [1]."], responder=lambda s, m: "Another answer [1].")
    engine = make_engine(llm, semantic_cache=SemanticCache(TableEmbedder(), threshold=0.9))
    client = client_for(engine)
    ask(client, "How do I enable CORS?")
    paraphrase = ask(client, "How can I turn on CORS?").json()
    unrelated_embedder_miss = ask(client, "What is a path parameter?")  # a miss: new model call
    assert paraphrase["cached"] == "semantic"
    assert unrelated_embedder_miss.status_code in (200,)
    info = client.get("/cache").json()
    assert info["semantic"]["hits"] == 1
    assert info["semantic"]["recent_hits"][0]["question"] == "How can I turn on CORS?"


# --- the index changing --------------------------------------------------------------------


def corpus_engine(tmp_path, llm=None):
    root = tmp_path / "corpus"
    page = root / "docs/en/docs/cors.md"
    page.parent.mkdir(parents=True)
    page.write_text("# CORS\n\nUse CORSMiddleware to allow other origins. " * 10, encoding="utf-8")
    live = LiveIndex(DIM, min_dead_to_compact=10**6)
    indexer = Indexer(live, Manifest(tmp_path / "m.sqlite"), Embedder(), root)
    indexer.reindex()
    exact = LRUCache(50)
    semantic = SemanticCache(Embedder(), threshold=0.99)

    def work():
        report = indexer.reindex()
        exact.clear()
        semantic.purge_except(live.generation)
        return report.as_dict()

    engine = Engine(
        HybridRetriever([LiveBM25(live)], method="weighted"),
        llm or FakeLLM(responder=lambda s, m: "Use CORSMiddleware [1]."),
        "fake-model",
        EngineConfig(data_dir="."),
        None,
        live=live,
        indexer=indexer,
        jobs=JobRunner(work),
        exact_cache=exact,
        semantic_cache=semantic,
    )
    return engine, page


def wait_for_job(client, job_id, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/ingest/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.02)
    raise TimeoutError


def test_ingest_returns_202_at_once_and_the_job_reports_what_it_did(tmp_path):
    engine, page = corpus_engine(tmp_path)
    client = client_for(engine)
    page.write_text("# CORS\n\nrewritten text about origins. " * 10, encoding="utf-8")
    response = client.post("/ingest")
    assert response.status_code == 202 and response.json()["created"] is True
    job = wait_for_job(client, response.json()["job"]["id"])
    assert job["status"] == "done" and job["result"]["changed"] == 1
    assert client.get("/ingest/nope").status_code == 404


def test_after_a_reindex_a_cached_answer_for_old_chunks_is_not_served(tmp_path):
    llm = FakeLLM(responder=lambda s, m: "Use CORSMiddleware [1].")
    engine, page = corpus_engine(tmp_path, llm)
    client = client_for(engine)
    assert ask(client).json()["cached"] is None
    assert ask(client).json()["cached"] == "exact"
    page.write_text("# CORS\n\ncompletely different origins text. " * 10, encoding="utf-8")
    wait_for_job(client, client.post("/ingest").json()["job"]["id"])
    after = ask(client).json()
    assert after["cached"] is None  # the generation changed, so the old entry cannot be found
    assert len(llm.calls) == 2


def test_the_cache_key_carries_the_index_generation_not_just_a_clear_after_ingest(tmp_path):
    """Even if nobody clears the cache, a changed index must not serve an old answer."""
    engine, _ = corpus_engine(tmp_path)
    client = client_for(engine)
    ask(client)
    assert ask(client).json()["cached"] == "exact"
    engine.live.apply(remove_paths=["docs/en/docs/cors.md"])  # the index changes behind the cache
    assert ask(client).json()["cached"] is None


def test_health_shows_the_index_state(tmp_path):
    engine, _ = corpus_engine(tmp_path)
    index = client_for(engine).get("/health").json()["index"]
    assert index["chunks"] > 0 and index["tombstones"] == 0 and index["generation"] >= 1


def test_ingest_without_a_corpus_is_501():
    assert client_for(make_engine()).post("/ingest").status_code == 501


# --- rate limit ----------------------------------------------------------------------------


def test_past_the_burst_a_client_gets_429_with_retry_after_and_health_is_exempt():
    client = client_for(make_engine(), rate_limit=(0.5, 3))
    codes = [client.post("/search", json={"query": "cors"}).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    limited = client.post("/search", json={"query": "cors"})
    assert int(limited.headers["retry-after"]) >= 1
    assert client.get("/health").status_code == 200  # a probe is never rate limited
    assert client.app.state.metrics.counter_value("cairn_rate_limited_total") == 3


def test_each_client_address_has_its_own_rate_limit():
    app_client = client_for(make_engine(), rate_limit=(0.01, 1))
    other = TestClient(app_client.app, client=("10.0.0.9", 5555))
    body = {"query": "x"}
    assert app_client.post("/search", json=body).status_code == 200
    assert app_client.post("/search", json=body).status_code == 429
    assert other.post("/search", json=body).status_code == 200  # another address, a full bucket


# --- the model being down ------------------------------------------------------------------


def test_after_repeated_failures_the_breaker_stops_calling_the_model_but_answers_still_come():
    inner = FakeLLM([ServerError("down")] * 10)
    breaker = CircuitBreaker(failures=2, cooldown_s=3600)
    client = client_for(make_engine(GuardedLLM(inner, breaker), breaker=breaker, exact_cache=None))
    bodies = [ask(client, f"question {i}?").json() for i in range(5)]
    assert all(b["degraded"] and b["chunks"] for b in bodies)  # always something useful
    assert len(inner.calls) == 2  # the model was tried twice, then left alone
    assert "circuit open" in bodies[-1]["error"]
    assert client.app.state.metrics.counter_value("cairn_llm_fallback_total") == 5
    assert "cairn_circuit_open 1.0" in client.get("/metrics").text


# --- backpressure --------------------------------------------------------------------------


def test_a_request_that_cannot_get_a_slot_in_time_is_refused_with_503():
    engine = make_engine(retriever=Retriever(delay=0.5), limit=ConcurrencyLimit(1, max_wait_s=0.05))
    client = client_for(engine)
    codes = []

    def search():
        codes.append(client.post("/search", json={"query": "cors"}).status_code)

    threads = [threading.Thread(target=search) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert sorted(codes) == [200, 503, 503]
    refused = client.app.state.metrics.counter_value("cairn_overloaded_total")
    assert refused == 2


# --- metrics, request ids, logs ------------------------------------------------------------


def test_metrics_count_requests_stages_tokens_and_the_state():
    client = client_for(make_engine())
    ask(client)
    ask(client, "another question?")
    client.post("/search", json={"query": "cors"})
    text = client.get("/metrics").text
    assert 'cairn_requests_total{route="/ask",status="200"} 2' in text
    assert 'cairn_requests_total{route="/search",status="200"} 1' in text
    assert 'cairn_stage_seconds_count{stage="retrieve"} 3' in text
    assert 'cairn_stage_seconds_count{stage="llm_total"} 2' in text
    assert 'cairn_tokens_total{kind="input"} 20' in text
    assert "cairn_ready 1.0" in text and "# TYPE cairn_request_seconds histogram" in text
    assert 'cairn_request_seconds_bucket{le="+Inf",route="/ask"} 2' in text


def test_a_stream_records_time_to_first_token():
    client = client_for(make_engine())
    ask(client, stream=True)
    assert (
        client.app.state.metrics.histogram_count("cairn_stage_seconds", stage="llm_first_token")
        == 1
    )


def test_unknown_paths_share_one_label_so_a_scanner_cannot_blow_up_the_metrics():
    client = client_for(make_engine())
    for i in range(5):
        client.get(f"/wp-admin/{i}")
    text = client.get("/metrics").text
    assert 'route="unmatched",status="404"} 5' in text and "wp-admin" not in text


def test_metrics_need_the_key_when_one_is_configured():
    client = client_for(make_engine(), api_key="s3cret")
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_every_response_has_a_request_id_and_a_sane_incoming_one_is_kept():
    client = client_for(make_engine())
    generated = client.get("/health").headers["x-request-id"]
    assert len(generated) == 16
    assert (
        client.get("/health", headers={"X-Request-ID": "trace-abc.1"}).headers["x-request-id"]
        == "trace-abc.1"
    )
    hostile = client.get("/health", headers={"X-Request-ID": "a b\nfake log line"})
    assert hostile.headers["x-request-id"] != "a b\nfake log line"
    assert "\n" not in hostile.headers["x-request-id"]


def test_the_access_log_is_one_json_line_with_the_request_id(caplog):
    client = client_for(make_engine())
    with caplog.at_level(logging.INFO, logger="cairn.serve.access"):
        client.get("/health", headers={"X-Request-ID": "req-42"})
    record = next(r for r in caplog.records if r.name == "cairn.serve.access")
    line = json.loads(JsonFormatter().format(record))
    assert line["fields" if False else "route"] == "/health" and line["status"] == 200
    assert line["message"] == "GET /health -> 200"
    assert line["logger"] == "cairn.serve.access" and line["seconds"] >= 0


def test_the_json_formatter_includes_the_current_request_id_and_exceptions():
    token = request_id_var.set("req-7")
    try:
        try:
            raise ValueError("boom")
        except ValueError:
            record = logging.LogRecord(
                "cairn.x", logging.ERROR, "f", 1, "it broke %s", ("badly",), logging.sys.exc_info()
            )
        line = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert line["request_id"] == "req-7" and line["message"] == "it broke badly"
    assert line["level"] == "ERROR" and "ValueError: boom" in line["exception"]


def test_metrics_histogram_buckets_are_cumulative_and_labels_are_escaped():
    m = Metrics()
    for value in (0.003, 0.2, 7):
        m.observe("t", value, a='x"y')
    text = m.render()
    assert 't_bucket{a="x\\"y",le="0.005"} 1' in text
    assert 't_bucket{a="x\\"y",le="0.25"} 2' in text
    assert 't_bucket{a="x\\"y",le="10"} 3' in text and 't_count{a="x\\"y"} 3' in text


def test_a_broken_gauge_does_not_take_the_metrics_page_down():
    m = Metrics()
    m.gauge("ok", lambda: 1)
    m.gauge("bad", lambda: 1 / 0)
    assert "ok 1.0" in m.render() and "bad NaN" in m.render()


# --- the deliberate bug used by the load test ---------------------------------------------


def test_the_blocking_demo_route_exists_only_when_asked_for():
    plain = client_for(make_engine())
    assert plain.post("/debug/search-blocking", json={"query": "x"}).status_code == 404
    demo = client_for(make_engine(), blocking_demo=True)
    assert demo.post("/debug/search-blocking", json={"query": "x"}).status_code == 200


def test_a_blocking_route_stalls_health_and_a_sync_route_does_not():
    """The checkpoint question: a CPU-bound call on the event loop freezes everything."""
    import asyncio

    import httpx

    engine = make_engine(retriever=Retriever(delay=0.4))
    service = Service(lambda: engine)
    service.start()
    service.wait(5)
    app = create_app(service, blocking_demo=True)

    async def health_latency(path):
        """Seconds from the moment the search started until /health was answered."""
        engine.retriever.started_at = None
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            heavy = asyncio.create_task(client.post(path, json={"query": "cors"}))
            while engine.retriever.started_at is None:
                await asyncio.sleep(0.002)
            await client.get("/health")
            answered = time.perf_counter() - engine.retriever.started_at
            await heavy
            return answered

    blocked = asyncio.run(health_latency("/debug/search-blocking"))
    free = asyncio.run(health_latency("/search"))
    assert blocked > 0.25  # /health had to wait for the whole search: the loop was frozen
    assert free < 0.15  # a plain `def` route runs in a thread; the loop stayed responsive
