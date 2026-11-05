"""Phase 7: the scripts `ask.py`, `gen_eval.py`, `judge_agreement.py` and `injection_test.py`.

Provided, not part of the exercise. They need your generation and evaluation modules. The tests use
the offline `hashing` embedding model and `--llm fake` (a scripted model), so they run in seconds
with no network and no cost.
"""

import json
import subprocess
import sys

import pytest

from cairn import config
from cairn.dense.models import load_embedder
from cairn.dense.retriever import build_dense_index
from cairn.ingest.pipeline import ingest, read_chunks
from cairn.retrieval.index import InvertedIndex

SCRIPTS = config.PROJECT_ROOT / "scripts"

QUESTIONS = [
    ("q1", "lexical", "dev", "Which part talks about cherries?", "docs/en/docs/a.md", 4, 5),
    ("q2", "code", "dev", "What does get_item return?", "fastapi/m.py", 1, 2),
    ("q3", "semantic", "test", "Held out question?", "docs/en/docs/a.md", 1, 2),
]
GEN_ROWS = [
    {"id": "q1", "reference": "Cherries are in the Part section."},
    {"id": "q2", "reference": "It returns x plus one."},
    {"id": "q3", "reference": "The held-out reference."},
    {"id": "u01", "answerable": False, "question": "What is the airspeed of a swallow?"},
]


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


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
    write_rows(
        tmp_path / "questions.jsonl",
        [
            {"id": i, "category": c, "split": s, "question": q,
             "gold": [{"path": p, "start_line": a, "end_line": b}]}
            for i, c, s, q, p, a, b in QUESTIONS
        ],
    )  # fmt: skip
    write_rows(tmp_path / "gen_questions.jsonl", GEN_ROWS)
    return tmp_path


def base(ws):
    return [
        "--chunks", str(ws / "chunks.jsonl"),
        "--bm25-index", str(ws / "bm25.pkl"),
        "--dense-index", str(ws / "dense.npz"),
        "--embedding-model", "hashing",
        "--reranker", "overlap",
        "--llm", "fake",
    ]  # fmt: skip


def run(script, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args], capture_output=True, text=True
    )


def gen_eval(ws, *extra, name="run"):
    return run(
        "gen_eval.py", *base(ws),
        "--gen-questions", str(ws / "gen_questions.jsonl"),
        "--questions", str(ws / "questions.jsonl"),
        "--out-dir", str(ws / "runs"), "--name", name,
        "--prices", str(ws / "none.json"),
        *extra,
    )  # fmt: skip


def run_rows(ws, name="run"):
    return [json.loads(line) for line in (ws / "runs" / f"{name}.jsonl").read_text().splitlines()]


# ------------------------------------------------------------------------------ gen_eval
def test_gen_eval_answers_every_dev_question_and_prints_a_summary(ws):
    result = gen_eval(ws, "--retriever", "dense")
    assert result.returncode == 0, result.stderr
    assert "questions: 3 (2 answerable, 1 not)" in result.stdout
    for word in (
        "correctness",
        "faithfulness",
        "citation precision",
        "abstain",
        "cost per question",
    ):
        assert word in result.stdout
    assert (ws / "runs" / "run.summary.json").is_file()


def test_the_run_file_has_one_line_per_question_and_never_a_test_question(ws):
    gen_eval(ws, "--retriever", "dense")
    rows = run_rows(ws)
    assert [r["id"] for r in rows] == ["q1", "q2", "u01"]
    answered = rows[0]
    assert answered["answer"] == "This is a demo answer [1]." and answered["citations"] == [1]
    assert answered["correct"] is True and answered["faithful"] is True
    assert answered["judge_reason"] == "demo" and answered["context"]
    assert {"number", "path", "start_line", "end_line", "text"} <= set(answered["context"][0])
    assert rows[2]["answerable"] is False and rows[2]["correct"] is None
    assert "Held out" not in json.dumps(rows)


def test_gen_eval_has_no_way_to_ask_for_the_test_split(ws):
    result = gen_eval(ws, "--split", "test")
    assert result.returncode == 2 and "unrecognized arguments" in result.stderr


def test_a_threshold_above_every_score_makes_every_question_abstain(ws):
    result = gen_eval(ws, "--retriever", "dense", "--threshold", "1000")
    assert result.returncode == 0, result.stderr
    rows = run_rows(ws)
    assert all(r["abstained"] and r["abstain_reason"] == "low_confidence" for r in rows)
    assert all(r["cost_usd"] == 0.0 for r in rows)
    assert rows[0]["correct"] is False and rows[0]["faithful"] is None  # an abstained answerable
    assert "false abstentions  1.000" in result.stdout
    assert "recall 1.000" in result.stdout


def test_the_dense_confidence_option_works_with_another_retriever(ws):
    result = gen_eval(ws, "--retriever", "hybrid", "--confidence", "dense", "--threshold", "1000")
    assert result.returncode == 0, result.stderr
    assert all(r["abstained"] for r in run_rows(ws))


def test_limit_keeps_the_first_questions(ws):
    gen_eval(ws, "--retriever", "bm25", "--limit", "1")
    assert [r["id"] for r in run_rows(ws)] == ["q1"]


def test_the_cost_of_an_answer_is_priced_from_its_tokens(ws):
    gen_eval(ws, "--retriever", "dense")
    # the fake model reports 10 input and 5 output tokens at 3 and 15 dollars per million
    assert run_rows(ws)[0]["cost_usd"] == pytest.approx((10 * 3 + 5 * 15) / 1_000_000)


def test_every_retriever_can_be_used(ws):
    for retriever in ("bm25", "dense", "hybrid", "rerank"):
        assert gen_eval(ws, "--retriever", retriever, "--limit", "1").returncode == 0, retriever


def test_a_bad_question_file_is_a_clean_error(ws):
    write_rows(ws / "gen_questions.jsonl", [{"id": "q999", "reference": "x"}])
    result = gen_eval(ws)
    assert result.returncode == 1 and "error:" in result.stderr and "Traceback" not in result.stderr


def test_a_missing_index_is_a_clean_error(ws):
    (ws / "bm25.pkl").unlink()
    result = gen_eval(ws)
    assert result.returncode == 1 and "not found" in result.stderr


# ---------------------------------------------------------------------------------- ask
def test_ask_prints_the_answer_its_sources_and_the_numbers(ws):
    result = run("ask.py", "Which part talks about cherries?", *base(ws), "--retriever", "dense")
    assert result.returncode == 0, result.stderr
    assert "This is a demo answer [1]." in result.stdout
    assert "[1] [" in result.stdout and "github.com/fastapi/fastapi/blob/0.143.0/" in result.stdout
    assert "tokens in" in result.stdout and "chunks" in result.stdout


def test_ask_can_stream(ws):
    result = run("ask.py", "Which part talks about cherries?", *base(ws), "--stream")
    assert result.returncode == 0, result.stderr
    assert "This is a demo answer [1]." in result.stdout


def test_ask_says_when_it_abstains(ws):
    result = run("ask.py", "Anything?", *base(ws), "--retriever", "dense", "--threshold", "1000")
    assert result.returncode == 0, result.stderr
    assert "(abstained: low_confidence)" in result.stdout


def test_ask_refuses_an_unknown_retriever(ws):
    assert run("ask.py", "q", *base(ws), "--retriever", "magic").returncode == 2


# ------------------------------------------------------------------- judge_agreement
RUN_ROWS = [
    {"id": f"q{i}", "category": "lexical", "answerable": True, "abstained": False,
     "question": f"Question {i}?", "reference": f"Reference {i}.", "answer": f"Answer {i} [1].",
     "correct": i % 2 == 0, "faithful": True, "judge_reason": "SECRET-JUDGE-REASON",
     "context": [{"number": 1, "path": "a.md", "start_line": 1, "end_line": 2,
                  "text": f"chunk {i}"}]}
    for i in range(1, 7)
] + [
    {"id": "q7", "category": "lexical", "answerable": True, "abstained": True, "question": "Q7?",
     "reference": "R7", "answer": "no", "correct": False, "faithful": None, "judge_reason": None,
     "context": []},
    {"id": "u1", "category": "unanswerable", "answerable": False, "abstained": True,
     "question": "U?", "reference": None, "answer": "no", "correct": None, "faithful": None,
     "judge_reason": None, "context": []},
]  # fmt: skip


@pytest.fixture
def run_file(tmp_path):
    path = tmp_path / "run.jsonl"
    write_rows(path, RUN_ROWS)
    return path


def test_the_sample_is_graded_blind(run_file, tmp_path):
    out = tmp_path / "sample.md"
    result = run(
        "judge_agreement.py", "--run", str(run_file), "--export-sample", "4", "--out", str(out)
    )
    assert result.returncode == 0, result.stderr
    text = out.read_text(encoding="utf-8")
    assert "SECRET-JUDGE-REASON" not in text
    assert text.count("**Question:**") == 4 and "**Reference:**" in text and "```" in text
    labels = json.loads(out.with_suffix(".labels.json").read_text())
    assert len(labels) == 4 and set(next(iter(labels.values()))) == {"correct", "faithful"}
    assert "q7" not in labels and "u1" not in labels  # abstained and unanswerable are not graded


def test_the_same_seed_gives_the_same_sample(run_file, tmp_path):
    for name in ("one", "two"):
        run("judge_agreement.py", "--run", str(run_file), "--export-sample", "3", "--seed", "5",
            "--out", str(tmp_path / f"{name}.md"))  # fmt: skip
    assert (tmp_path / "one.md").read_text() == (tmp_path / "two.md").read_text()


def test_asking_for_more_than_there_is_an_error(run_file, tmp_path):
    result = run("judge_agreement.py", "--run", str(run_file), "--export-sample", "50",
                 "--out", str(tmp_path / "s.md"))  # fmt: skip
    assert result.returncode == 1 and "error:" in result.stderr


def test_agreement_and_kappa_against_hand_labels(run_file, tmp_path):
    # the judge said correct for q2, q4, q6 and not for q1, q3, q5 (and faithful for all)
    labels = {
        "q1": {"correct": 0, "faithful": 1},
        "q2": {"correct": 1, "faithful": 1},
        "q3": {"correct": 1, "faithful": 1},  # disagrees with the judge
        "q4": {"correct": 1, "faithful": 0},  # disagrees on faithfulness
        "q5": {"correct": None, "faithful": 1},  # skipped for correctness
    }
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels))
    result = run("judge_agreement.py", "--run", str(run_file), "--labels", str(path))
    assert result.returncode == 0, result.stderr
    assert "correct: 4 answers, agreement 0.750" in result.stdout
    assert "q3: you said 1, the judge said 0" in result.stdout
    assert "faithful: 5 answers, agreement 0.800" in result.stdout
    assert "q4: you said 0, the judge said 1" in result.stdout


def test_labels_for_an_unknown_question_are_an_error(run_file, tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"nope": {"correct": 1, "faithful": 1}}))
    assert run("judge_agreement.py", "--run", str(run_file), "--labels", str(path)).returncode == 1


def test_exactly_one_mode_is_required(run_file, tmp_path):
    assert run("judge_agreement.py", "--run", str(run_file)).returncode == 2
    result = run("judge_agreement.py", "--run", str(run_file), "--export-sample", "1",
                 "--labels", str(tmp_path / "x.json"))  # fmt: skip
    assert result.returncode == 2


# ------------------------------------------------------------------- injection_test
def test_the_injection_script_runs_every_case_with_a_safe_model():
    result = run("injection_test.py", "--llm", "fake")
    assert result.returncode == 0, result.stderr
    assert "9 of 9 cases held" in result.stdout and "LEAKED" not in result.stdout


def test_the_injection_script_reports_a_model_that_obeys():
    result = run("injection_test.py", "--llm", "fake", "--naive")
    assert result.returncode == 0, result.stderr
    # the obeying fake leaks whenever the whole canary is in its prompt; `split_canary` never
    # writes it whole, so that case holds even for a model that obeys
    assert "1 of 9 cases held" in result.stdout and result.stdout.count("LEAKED") == 8
