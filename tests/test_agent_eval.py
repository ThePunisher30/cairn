"""Phase 8: the helpers in `evaluation/agent.py` and the script `scripts/agent_eval.py`.

Provided, not part of the exercise. The script tests use the offline `hashing` embedding model and
`--llm fake` (a scripted model that searches once and then answers), so they run in seconds with no
network, no model and no cost.
"""

import json
import subprocess
import sys

import pytest

from cairn import config
from cairn.dense.models import load_embedder
from cairn.dense.retriever import build_dense_index
from cairn.evaluation.agent import (
    AgentQuestionError,
    estimate_cost,
    hop_recall,
    judge_context,
    load_multihop_questions,
    overlaps,
    percentile,
    summarize_agent_rows,
)
from cairn.ingest.pipeline import ingest, read_chunks
from cairn.retrieval.index import InvertedIndex
from cairn.types import Hit, Span

SCRIPTS = config.PROJECT_ROOT / "scripts"


# ------------------------------------------------------------------ helpers
def test_overlaps_needs_the_same_file_and_touching_lines():
    span = Span("a.md", 10, 20)
    assert overlaps(Hit("a.md", 1, 10), span) and overlaps(Hit("a.md", 20, 30), span)
    assert overlaps(Hit("a.md", 12, 14), span) and overlaps(Hit("a.md", 1, 99), span)
    assert not overlaps(Hit("a.md", 1, 9), span) and not overlaps(Hit("a.md", 21, 30), span)
    assert not overlaps(Hit("b.md", 10, 20), span)


def test_hop_recall_is_the_share_of_spans_some_hit_touches():
    spans = (Span("a.md", 1, 5), Span("b.py", 10, 20), Span("c.md", 3, 4))
    hits = [Hit("a.md", 4, 9), Hit("b.py", 100, 120), Hit("b.py", 15, 16)]
    assert hop_recall(hits, spans) == pytest.approx(2 / 3)
    assert hop_recall([], spans) == 0.0
    assert hop_recall(hits, ()) is None


def test_percentile_interpolates():
    assert percentile([], 95) is None
    assert percentile([7.0], 95) == 7.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
    assert percentile([4.0, 1.0, 3.0, 2.0], 100) == 4.0
    assert percentile([0.0, 10.0], 95) == pytest.approx(9.5)


def test_estimate_cost_prices_per_million_tokens():
    assert estimate_cost(1_000_000, 500_000, (4.0, 20.0)) == pytest.approx(14.0)
    assert estimate_cost(0, 0, (4.0, 20.0)) == 0.0


def test_judge_context_keeps_the_citation_numbers_and_the_token_budget():
    long_text = " ".join(f"word{i}" for i in range(2000))
    hits = [
        (3, Hit("a.md", 1, 5, 1.0, "short chunk about cherries")),
        (7, Hit("b.md", 1, 5, 1.0, long_text)),
    ]
    context = judge_context(hits, max_tokens=300)
    assert [c.number for c in context.chunks] == [3, 7]
    assert (
        context.chunks[0].text == "short chunk about cherries" and not context.chunks[0].truncated
    )
    assert context.chunks[1].truncated and len(context.chunks[1].text) < len(long_text)
    assert context.tokens <= 300


def test_judge_context_stops_when_almost_nothing_is_left():
    text = " ".join(f"w{i}" for i in range(100))
    context = judge_context([(1, Hit("a", 1, 2, 1.0, text)), (2, Hit("b", 1, 2, 1.0, text))], 120)
    assert [c.number for c in context.chunks] == [1]
    assert context.dropped == 1


def row(**kw):
    base = {
        "answer": "x", "abstained": False, "correct": True, "faithful": True, "hop_recall": 1.0,
        "retrievals": 1, "tool_calls": 0, "steps": 1, "tool_errors": 0, "duplicates": 0,
        "input_tokens": 100, "output_tokens": 10, "latency_ms": 1000.0, "est_cost_usd": 0.01,
        "stop_reason": "answered",
    }  # fmt: skip
    return {**base, **kw}


def test_the_summary_averages_and_counts():
    rows = [
        row(),
        row(correct=False, faithful=None, tool_calls=4, steps=3, latency_ms=3000.0, hop_recall=0.5),
        row(answer=None, correct=False, faithful=None, stop_reason="max_steps", abstained=False),
        row(correct=None, faithful=None, abstained=True),
    ]
    s = summarize_agent_rows(rows)
    assert s["n"] == 4 and s["n_graded"] == 3
    assert s["correctness"] == pytest.approx(1 / 3)
    assert s["faithfulness"] == 1.0  # only the one judged row counts
    assert s["abstained"] == 1 and s["no_answer"] == 1
    assert s["hop_recall"] == pytest.approx(0.875)
    assert s["tool_calls_per_question"] == 1.0 and s["model_calls_per_question"] == 1.5
    assert s["latency_p50_ms"] == 1000.0
    assert s["stop_reasons"] == {"answered": 3, "max_steps": 1}
    assert summarize_agent_rows([])["correctness"] is None


def test_the_multihop_loader_reads_one_split_and_reports_bad_rows(tmp_path):
    good = {"id": "m01", "split": "dev", "question": "Q one?", "reference": "R",
            "hops": [{"path": "a.md", "start_line": 1, "end_line": 2},
                     {"path": "b.py", "start_line": 3, "end_line": 4}]}  # fmt: skip
    held = {**good, "id": "m02", "split": "test", "question": "Held out?"}
    path = tmp_path / "m.jsonl"
    path.write_text(json.dumps(good) + "\n" + json.dumps(held) + "\n", encoding="utf-8")
    dev = load_multihop_questions(path)
    assert [q.id for q in dev] == ["m01"]
    assert dev[0].spans == (Span("a.md", 1, 2), Span("b.py", 3, 4))
    assert [q.id for q in load_multihop_questions(path, split="test")] == ["m02"]
    path.write_text(json.dumps({"id": "m03", "split": "dev"}) + "\n", encoding="utf-8")
    with pytest.raises(AgentQuestionError):
        load_multihop_questions(path)


# ------------------------------------------------------------------ the script
def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def hops(*spans):
    return [{"path": p, "start_line": a, "end_line": b} for p, a, b in spans]


A_MD, M_PY = ("docs/en/docs/a.md", 4, 5), ("fastapi/m.py", 1, 2)
MULTIHOP = [
    {"id": "m01", "split": "dev", "reference": "Part; x plus one.", "hops": hops(A_MD, M_PY),
     "question": "Which part talks about cherries and what does it return?"},
    {"id": "m02", "split": "dev", "reference": "x plus one.", "hops": hops(M_PY, A_MD),
     "question": "What does get_item return?"},
    {"id": "m03", "split": "test", "reference": "unseen", "hops": hops(M_PY, A_MD),
     "question": "Held out multi hop question?"},
]  # fmt: skip
SINGLE = [
    ("q1", "lexical", "dev", "Which part talks about cherries?", "docs/en/docs/a.md", 4, 5),
    ("q3", "semantic", "test", "Held out question?", "docs/en/docs/a.md", 1, 2),
]
GEN_ROWS = [
    {"id": "q1", "reference": "Cherries are in the Part section."},
    {"id": "q3", "reference": "The held-out reference."},
    {"id": "u01", "answerable": False, "question": "What is the airspeed of a swallow?"},
]


@pytest.fixture
def ws(tmp_path):
    for rel, text in {
        "docs/en/docs/a.md": "# Title\nsome words here\n\n## Part\nmore words about cherries\n",
        "fastapi/m.py": "def get_item(x):\n    return x + 1\n",
    }.items():
        path = tmp_path / "corpus" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    ingest(tmp_path / "corpus", tmp_path / "chunks.jsonl")
    chunks = read_chunks(tmp_path / "chunks.jsonl")
    InvertedIndex.build(chunks).save(tmp_path / "bm25.pkl")
    build_dense_index(chunks, load_embedder("hashing"))[0].save(tmp_path / "dense.npz")
    write_rows(tmp_path / "multihop.jsonl", MULTIHOP)
    write_rows(
        tmp_path / "questions.jsonl",
        [
            {"id": i, "category": c, "split": s, "question": q,
             "gold": [{"path": p, "start_line": a, "end_line": b}]}
            for i, c, s, q, p, a, b in SINGLE
        ],
    )  # fmt: skip
    write_rows(tmp_path / "gen_questions.jsonl", GEN_ROWS)
    return tmp_path


def agent_eval(ws, *extra, name="run"):
    command = [
        sys.executable, str(SCRIPTS / "agent_eval.py"),
        "--chunks", str(ws / "chunks.jsonl"),
        "--bm25-index", str(ws / "bm25.pkl"),
        "--dense-index", str(ws / "dense.npz"),
        "--embedding-model", "hashing",
        "--reranker", "overlap",
        "--llm", "fake",
        "--corpus", str(ws / "corpus"),
        "--multihop", str(ws / "multihop.jsonl"),
        "--gen-questions", str(ws / "gen_questions.jsonl"),
        "--questions", str(ws / "questions.jsonl"),
        "--out-dir", str(ws / "runs"), "--name", name,
        "--prices", str(ws / "none.json"),
        *extra,
    ]  # fmt: skip
    return subprocess.run(command, capture_output=True, text=True)


def run_rows(ws, name="run"):
    path = ws / "runs" / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_the_agent_mode_searches_answers_and_writes_rows_and_traces(ws):
    result = agent_eval(ws, "--mode", "agent")
    assert result.returncode == 0, result.stderr
    rows = run_rows(ws)
    assert [r["id"] for r in rows] == ["m01", "m02"]  # the test question is never run
    first = rows[0]
    assert first["mode"] == "agent" and first["answer"] == "This is a demo answer [1]."
    assert first["stop_reason"] == "answered" and first["correct"] is True
    assert (first["retrievals"], first["tool_calls"], first["steps"]) == (1, 1, 2)
    assert first["seen"] and first["input_tokens"] > 0 and first["latency_ms"] > 0
    assert 0.0 <= first["hop_recall"] <= 1.0
    traces = (ws / "runs" / "run.traces.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(traces) == 2
    assert json.loads(traces[0])["steps"][0]["tool_calls"][0]["name"] == "search"
    for word in ("correctness", "evidence seen", "searches", "tokens per question", "stop reasons"):
        assert word in result.stdout
    assert "Held out" not in json.dumps(rows)
    assert (ws / "runs" / "run.summary.json").is_file()


def test_the_single_mode_makes_one_retrieval_and_one_model_call(ws):
    result = agent_eval(ws, "--mode", "single")
    assert result.returncode == 0, result.stderr
    first = run_rows(ws)[0]
    assert first["mode"] == "single" and first["answer"] == "This is a demo answer [1]."
    assert (first["retrievals"], first["tool_calls"], first["steps"]) == (1, 0, 1)
    assert first["correct"] is True and first["stop_reason"] == "answered"
    assert not (ws / "runs" / "run.traces.jsonl").read_text(encoding="utf-8").strip()


def test_a_run_that_ends_without_an_answer_counts_as_incorrect(ws):
    result = agent_eval(ws, "--mode", "agent", "--max-steps", "1")
    assert result.returncode == 0, result.stderr
    rows = run_rows(ws)
    assert all(r["answer"] is None and r["stop_reason"] == "max_steps" for r in rows)
    assert all(r["correct"] is False and r["faithful"] is None for r in rows)
    assert "no answer at all: 2" in result.stdout


def test_the_dev_set_is_the_answerable_phase_7_dev_questions(ws):
    result = agent_eval(ws, "--mode", "single", "--set", "dev")
    assert result.returncode == 0, result.stderr
    assert [r["id"] for r in run_rows(ws)] == ["q1"]  # not the test question, not the unanswerable


def test_there_is_no_way_to_ask_for_the_test_split(ws):
    result = agent_eval(ws, "--mode", "agent", "--split", "test")
    assert result.returncode == 2 and "unrecognized arguments" in result.stderr


def test_resume_skips_the_questions_already_done(ws):
    assert agent_eval(ws, "--mode", "agent", "--limit", "1").returncode == 0
    assert [r["id"] for r in run_rows(ws)] == ["m01"]
    result = agent_eval(ws, "--mode", "agent", "--resume")
    assert result.returncode == 0, result.stderr
    assert [r["id"] for r in run_rows(ws)] == ["m01", "m02"]
    traces = (ws / "runs" / "run.traces.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(traces) == 2
    assert "questions: 2" in result.stdout


def test_bad_options_fail_with_a_message(ws):
    assert agent_eval(ws, "--mode", "agent", "--limit", "0").returncode == 1
    missing = agent_eval(ws, "--mode", "agent", "--corpus", str(ws / "nowhere"))
    assert missing.returncode == 1 and "not found" in missing.stderr
    assert agent_eval(ws).returncode == 2  # --mode is required
