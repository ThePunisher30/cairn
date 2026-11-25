import json

import pytest
from fastapi.testclient import TestClient

from cairn.generation.fake import FakeLLM
from cairn.ingest.pipeline import ingest
from cairn.serve.app import UI_CSP, create_app
from cairn.serve.engine import Engine, EngineConfig, build_engine
from cairn.serve.source import SourceError, in_scope, read_source
from cairn.serve.state import Service
from cairn.types import Hit


class Retriever:
    def search(self, query, k):
        hits = [
            Hit("docs/en/docs/tutorial/cors.md", 3, 5, 0.9, "Use CORSMiddleware."),
            Hit("fastapi/utils.py", 1, 2, 0.5, "def f(): pass"),
        ]
        return hits[:k]


def make_corpus(root):
    page = root / "docs/en/docs/tutorial/cors.md"
    page.parent.mkdir(parents=True)
    page.write_text("\n".join(f"line {i}" for i in range(1, 31)) + "\n", encoding="utf-8")
    code = root / "fastapi/utils.py"
    code.parent.mkdir(parents=True)
    code.write_text("def f():\n    pass\n", encoding="utf-8")
    (root / "secret.txt").write_text("do not serve", encoding="utf-8")
    (root / "docs/en/docs/notes.txt").write_text("not indexed", encoding="utf-8")
    return root


def client_for(tmp_path, llm="fake", api_key=None, with_root=True):
    root = make_corpus(tmp_path / "corpus") if with_root else None
    cfg = EngineConfig(data_dir=tmp_path, llm=llm)
    model = FakeLLM(["Use CORSMiddleware [1]."]) if llm != "none" else None
    name = "fake-model" if model else "none"
    engine = Engine(Retriever(), model, name, cfg, None, source_root=root)
    service = Service(lambda: engine)
    service.start()
    assert service.wait(5)
    return TestClient(create_app(service, api_key=api_key))


def events(text):
    out = []
    for block in text.strip().split("\n\n"):
        name, data = block.split("\n")
        out.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


# --- the page ------------------------------------------------------------------------------


def test_the_page_and_its_files_are_served_with_security_headers(tmp_path):
    client = client_for(tmp_path)
    redirect = client.get("/", follow_redirects=False)
    assert redirect.status_code in (302, 307) and redirect.headers["location"] == "/ui/"
    page = client.get("/ui/")
    assert page.status_code == 200 and "<title>Cairn" in page.text
    assert page.headers["content-security-policy"] == UI_CSP
    assert page.headers["x-content-type-options"] == "nosniff"
    assert page.headers["x-frame-options"] == "DENY"
    assert client.get("/ui/app.js").status_code == 200
    assert client.get("/ui/style.css").headers["content-type"].startswith("text/css")


def test_the_page_has_no_inline_script_or_style_so_the_policy_can_forbid_them(tmp_path):
    html = client_for(tmp_path).get("/ui/").text
    assert "<script src=" in html and "onclick" not in html and " style=" not in html
    assert "<style" not in html and "unsafe-inline" not in UI_CSP


def test_the_script_never_puts_text_from_the_server_into_html(tmp_path):
    script = client_for(tmp_path).get("/ui/app.js").text
    for dangerous in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert dangerous not in script


def test_api_responses_have_nosniff_but_no_page_policy(tmp_path):
    response = client_for(tmp_path).get("/health")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "content-security-policy" not in response.headers


def test_health_says_how_the_server_is_set_up(tmp_path):
    body = client_for(tmp_path, api_key="k" * 20).get("/health").json()
    assert body["auth_required"] is True
    assert body["mode"] == {"retriever": "hybrid", "llm": "fake", "model": "fake-model"}


# --- the source viewer ---------------------------------------------------------------------


def get_source(client, **params):
    return client.get("/source", params=params)


def test_source_returns_the_cited_lines_with_context_and_marks_them(tmp_path):
    body = get_source(
        client_for(tmp_path), path="docs/en/docs/tutorial/cors.md", start=10, end=12
    ).json()
    assert (body["first_line"], body["last_line"], body["total_lines"]) == (4, 18, 30)
    cited = [line["n"] for line in body["lines"] if line["cited"]]
    assert cited == [10, 11, 12] and body["lines"][0]["text"] == "line 4"


def test_source_clamps_to_the_file_and_to_a_maximum_size(tmp_path):
    body = get_source(client_for(tmp_path), path="fastapi/utils.py", start=1, end=99).json()
    assert body["end_line"] == 2 and body["last_line"] == 2
    big = tmp_path / "big"
    (big / "fastapi").mkdir(parents=True)
    (big / "fastapi/big.py").write_text("x = 1\n" * 2000, encoding="utf-8")
    shown = read_source(big, "fastapi/big.py", 1, 2000)
    assert len(shown["lines"]) == 400


@pytest.mark.parametrize(
    "path",
    [
        "../secret.txt",
        "docs/en/docs/../../../secret.txt",
        "/etc/passwd",
        "C:/Windows/win.ini",
        "secret.txt",  # inside the folder but not an indexed file
        "docs/en/docs/notes.txt",  # inside an indexed folder, wrong kind of file
        "fastapi/missing.py",
        "docs\\..\\secret.txt",
    ],
)
def test_source_refuses_paths_outside_the_indexed_corpus(tmp_path, path):
    client = client_for(tmp_path)
    response = get_source(client, path=path)
    assert response.status_code == 404
    assert "do not serve" not in response.text


def test_source_rejects_bad_numbers(tmp_path):
    client = client_for(tmp_path)
    assert get_source(client, path="fastapi/utils.py", start=0).status_code == 422
    assert get_source(client, path="fastapi/utils.py", start=5, end=2).status_code == 422
    assert get_source(client, path="fastapi/utils.py", start=99).status_code == 404
    assert get_source(client, start=1).status_code == 422


def test_source_without_a_corpus_says_so(tmp_path):
    response = get_source(client_for(tmp_path, with_root=False), path="fastapi/utils.py")
    assert response.status_code == 404 and "not available" in response.json()["detail"]


def test_source_needs_the_key_when_one_is_configured(tmp_path):
    client = client_for(tmp_path, api_key="k" * 20)
    assert get_source(client, path="fastapi/utils.py").status_code == 401
    ok = client.get("/source", params={"path": "fastapi/utils.py"}, headers={"X-API-Key": "k" * 20})
    assert ok.status_code == 200


def test_the_scope_check_is_the_corpus_definition():
    assert in_scope("fastapi/routing.py") and in_scope("docs_src/a/b.py")
    assert in_scope("docs/en/docs/tutorial/x.md")
    assert not in_scope("docs/en/docs/release-notes.md") and not in_scope("setup.py")
    with pytest.raises(SourceError):
        read_source(None, "fastapi/a.py", 1)


# --- answers carry structured references ---------------------------------------------------


def test_an_answer_lists_its_citations_as_references_with_urls(tmp_path):
    body = client_for(tmp_path).post("/ask", json={"question": "cors?"}).json()
    assert [r["number"] for r in body["references"]] == [1]
    ref = body["references"][0]
    assert ref["path"] == "docs/en/docs/tutorial/cors.md" and ref["start_line"] == 3
    assert ref["url"].endswith("/blob/0.143.0/docs/en/docs/tutorial/cors.md#L3-L5")


# --- retrieval only ------------------------------------------------------------------------


def test_without_a_model_ask_returns_the_passages_and_says_why(tmp_path):
    client = client_for(tmp_path, llm="none")
    body = client.post("/ask", json={"question": "cors?"}).json()
    assert body["degraded"] is True and body["answer"] == ""
    assert "no language model is attached" in body["error"]
    paths = [c["path"] for c in body["chunks"]]
    assert paths == ["docs/en/docs/tutorial/cors.md", "fastapi/utils.py"]
    assert [r["number"] for r in body["references"]] == [1, 2]
    assert client.app.state.metrics.counter_value("cairn_llm_fallback_total") == 0  # no failure


def test_without_a_model_the_stream_is_a_fallback_event_and_done(tmp_path):
    client = client_for(tmp_path, llm="none")
    response = client.post("/ask", json={"question": "cors?", "stream": True})
    assert [name for name, _ in events(response.text)] == ["fallback", "done"]
    assert client.get("/health").json()["mode"]["llm"] == "none"


# --- building the engine without torch -----------------------------------------------------


def test_a_bm25_only_engine_builds_and_answers_without_an_embedding_model(tmp_path):
    root = make_corpus(tmp_path / "raw" / "fastapi")
    data = tmp_path / "data"
    ingest(root, data / "chunks.jsonl")
    cfg = EngineConfig(data_dir=data, corpus_root=root, retriever="bm25", llm="none")
    engine = build_engine(cfg)
    assert engine.live.dim == 0 and engine.model == "none" and engine.retrieval_only
    assert engine.retriever.search("CORSMiddleware line", 3)
    assert engine.source_root == root
    page = root / "docs/en/docs/tutorial/cors.md"
    page.write_text("# CORS\n\nbrandnewword appears here.\n", encoding="utf-8")
    report = engine.indexer.reindex()  # a BM25-only index can be re-indexed too
    assert report.changed == 1
    assert engine.retriever.search("brandnewword", 1)[0].path == "docs/en/docs/tutorial/cors.md"
    assert "brandnewword" in (data / "chunks.jsonl").read_text(encoding="utf-8")
    assert not list(data.glob("dense-*.npz"))  # and it never invents a dense index


def test_the_semantic_cache_needs_the_embedding_model(tmp_path):
    cfg = EngineConfig(data_dir=tmp_path, retriever="bm25", semantic_cache=True)
    with pytest.raises(ValueError, match="embedding model"):
        build_engine(cfg)


def test_an_unknown_retriever_is_refused(tmp_path):
    with pytest.raises(ValueError, match="retriever"):
        build_engine(EngineConfig(data_dir=tmp_path, retriever="magic"))
