import json
import threading

import pytest
from fastapi.testclient import TestClient

from cairn.generation.errors import ServerError
from cairn.generation.fake import FakeLLM
from cairn.serve.app import create_app
from cairn.serve.engine import Engine, EngineConfig
from cairn.serve.state import Service
from cairn.types import Hit


class StubRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, k):
        self.calls.append((query, k))
        hits = [
            Hit("docs/cors.md", 1, 5, 0.9, "Use CORSMiddleware to allow other origins."),
            Hit("docs/a.md", 3, 4, 0.5, "Another chunk."),
        ]
        return hits[:k]


def make_engine(llm=None, retriever=None):
    llm = llm or FakeLLM(["Use CORSMiddleware [1]."])
    cfg = EngineConfig(data_dir=".")
    return Engine(retriever or StubRetriever(), llm, "fake-model", cfg, None)


def ready_service(engine=None):
    service = Service(lambda: engine or make_engine())
    service.start()
    assert service.wait(5)
    return service


def client_for(service=None, api_key=None):
    return TestClient(create_app(service or ready_service(), api_key=api_key))


def parse_events(text):
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.split("\n")
        name = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        events.append((name, data))
    return events


# --- /health -------------------------------------------------------------------------------


def test_health_is_200_when_ready():
    response = client_for().get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["ready"] is True


def test_health_is_503_while_loading_and_names_the_state():
    release = threading.Event()
    service = Service(lambda: release.wait(5) or make_engine())
    service.start()
    try:
        response = client_for(service).get("/health")
        assert response.status_code == 503
        assert response.json()["ready"] is False
        assert response.json()["status"] in ("loading", "starting")
    finally:
        release.set()
        service.wait(5)


def test_health_is_503_with_the_error_when_loading_failed():
    def loader():
        raise FileNotFoundError("data/chunks.jsonl not found")

    service = Service(loader)
    service.start()
    service.wait(5)
    response = client_for(service).get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "failed"
    assert "chunks.jsonl" in response.json()["error"]


# --- readiness for the other routes -------------------------------------------------------


def test_search_and_ask_say_503_until_ready_with_a_retry_after():
    release = threading.Event()
    service = Service(lambda: release.wait(5) or make_engine())
    service.start()
    try:
        client = client_for(service)
        for path, body in (("/search", {"query": "cors"}), ("/ask", {"question": "cors?"})):
            response = client.post(path, json=body)
            assert response.status_code == 503
            assert response.headers["retry-after"] == "5"
    finally:
        release.set()
        service.wait(5)


def test_a_failed_service_says_503_with_the_reason():
    service = Service(lambda: (_ for _ in ()).throw(RuntimeError("no index")))
    service.start()
    service.wait(5)
    response = client_for(service).post("/search", json={"query": "cors"})
    assert response.status_code == 503
    assert "no index" in response.json()["detail"]


# --- authentication ------------------------------------------------------------------------


def test_a_missing_or_wrong_key_is_401_and_the_right_key_works():
    client = client_for(api_key="s3cret")
    body = {"query": "cors"}
    assert client.post("/search", json=body).status_code == 401
    assert client.post("/search", json=body, headers={"X-API-Key": "nope"}).status_code == 401
    assert client.post("/search", json=body, headers={"X-API-Key": ""}).status_code == 401
    assert client.post("/search", json=body, headers={"X-API-Key": "s3cret"}).status_code == 200


def test_the_key_is_checked_before_readiness_so_a_stranger_learns_nothing():
    release = threading.Event()
    service = Service(lambda: release.wait(5) or make_engine())
    service.start()
    try:
        client = client_for(service, api_key="s3cret")
        assert client.post("/search", json={"query": "x"}).status_code == 401
    finally:
        release.set()
        service.wait(5)


def test_health_needs_no_key():
    assert client_for(api_key="s3cret").get("/health").status_code == 200


def test_no_key_configured_means_no_authentication():
    assert client_for(api_key=None).post("/search", json={"query": "cors"}).status_code == 200


# --- /search -------------------------------------------------------------------------------


def test_search_returns_the_hits_and_passes_k_on():
    retriever = StubRetriever()
    client = client_for(ready_service(make_engine(retriever=retriever)))
    response = client.post("/search", json={"query": "  how to enable CORS  ", "k": 1})
    assert response.status_code == 200
    hits = response.json()["hits"]
    assert len(hits) == 1
    assert hits[0]["path"] == "docs/cors.md"
    assert hits[0]["start_line"] == 1
    assert "CORSMiddleware" in hits[0]["text"]
    assert retriever.calls == [("how to enable CORS", 1)]  # stripped; k reached the retriever


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 2001},
        {"query": "x", "k": 0},
        {"query": "x", "k": 21},
        {"query": "x", "k": "many"},
    ],
)
def test_search_rejects_bad_input_with_422(body):
    assert client_for().post("/search", json=body).status_code == 422


# --- /ask (JSON) ---------------------------------------------------------------------------


def test_ask_returns_a_cited_answer_with_usage_and_sources():
    response = client_for().post("/ask", json={"question": "How do I enable CORS?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Use CORSMiddleware [1]."
    assert body["citations"] == [1]
    assert body["abstained"] is False
    assert body["abstain_reason"] is None
    assert body["model"] == "fake-model"
    assert body["input_tokens"] == 10 and body["output_tokens"] == 5
    assert body["latency_ms"] >= 0
    assert len(body["sources"]) == 1 and "docs/cors.md" in body["sources"][0]


def test_ask_reports_an_abstention():
    client = client_for(ready_service(make_engine(FakeLLM(["CANNOT_ANSWER"]))))
    body = client.post("/ask", json={"question": "What is the weather?"}).json()
    assert body["abstained"] is True
    assert body["abstain_reason"] == "model_declined"
    assert body["citations"] == []


def test_a_failing_language_model_is_502_not_a_crash():
    client = client_for(ready_service(make_engine(FakeLLM([ServerError("overloaded")]))))
    response = client.post("/ask", json={"question": "x?"})
    assert response.status_code == 502
    assert "overloaded" in response.json()["detail"]
    assert client.get("/health").status_code == 200  # the service is still up


@pytest.mark.parametrize("body", [{}, {"question": ""}, {"question": "x" * 2001}, {"question": 3}])
def test_ask_rejects_bad_input_with_422(body):
    assert client_for().post("/ask", json=body).status_code == 422


def test_each_request_gets_its_own_answerer():
    engine = make_engine()
    assert engine.make_answerer() is not engine.make_answerer()


# --- /ask (stream) -------------------------------------------------------------------------


def test_stream_sends_tokens_then_one_done_event_with_the_whole_answer():
    response = client_for().post("/ask", json={"question": "How do I enable CORS?", "stream": True})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_events(response.text)
    names = [name for name, _ in events]
    assert names[-1] == "done" and names.count("done") == 1
    assert set(names[:-1]) == {"token"} and len(names) > 2
    streamed = "".join(data["text"] for name, data in events if name == "token")
    done = events[-1][1]
    assert streamed.strip() == "Use CORSMiddleware [1]."
    assert done["answer"] == "Use CORSMiddleware [1]."
    assert done["citations"] == [1]
    assert done["input_tokens"] == 10


def test_a_token_with_a_newline_stays_inside_its_event():
    llm = FakeLLM(["line one\nline two [1]"])
    response = client_for(ready_service(make_engine(llm))).post(
        "/ask", json={"question": "x?", "stream": True}
    )
    events = parse_events(response.text)  # would fail to parse if a newline split an event
    streamed = "".join(data["text"] for name, data in events if name == "token")
    assert "line one\nline two" in streamed


def test_a_failure_during_the_stream_ends_with_an_error_event_and_no_done():
    client = client_for(ready_service(make_engine(FakeLLM([ServerError("overloaded")]))))
    response = client.post("/ask", json={"question": "x?", "stream": True})
    assert response.status_code == 200  # the status was sent before the failure
    names = [name for name, _ in parse_events(response.text)]
    assert names == ["error"]
    assert "overloaded" in parse_events(response.text)[0][1]["message"]


def test_an_unexpected_failure_in_the_stream_hides_the_details():
    class Broken:
        def search(self, query, k):
            raise RuntimeError("secret internal detail")

    client = client_for(ready_service(make_engine(retriever=Broken())))
    response = client.post("/ask", json={"question": "x?", "stream": True})
    ((name, data),) = parse_events(response.text)
    assert name == "error"
    assert data["message"] == "internal error"
    assert "secret" not in response.text
