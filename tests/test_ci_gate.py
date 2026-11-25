import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "ci_corpus"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = load("ci_gate")


def test_check_reports_every_metric_below_its_minimum_and_missing_ones():
    numbers = {"bm25": {"hit@5": 0.7, "mrr": 0.7}, "hybrid": {"hit@5": 0.8}}
    thresholds = {
        "bm25": {"hit@5": 0.75, "mrr": 0.6},
        "hybrid": {"hit@5": 0.8, "mrr": 0.5},
        "dense": {"hit@5": 0.1},
        "_note": "ignored",
    }
    failures = gate.check(numbers, thresholds)
    assert len(failures) == 3
    assert any("bm25 hit@5" in f for f in failures)
    assert any("hybrid mrr" in f and "None" in f for f in failures)  # a missing metric fails
    assert any(f.startswith("dense") for f in failures)


def test_a_metric_exactly_at_its_minimum_passes():
    assert gate.check({"bm25": {"hit@5": 0.775}}, {"bm25": {"hit@5": 0.775}}) == []


def test_the_committed_thresholds_cover_the_gated_metrics():
    thresholds = json.loads((ROOT / "eval" / "ci_gate.json").read_text(encoding="utf-8"))
    assert set(thresholds) == {"bm25", "hybrid"}
    for wanted in thresholds.values():
        assert set(wanted) == set(gate.MARGINS)
        assert all(0.3 < v <= 1.0 for v in wanted.values())


def test_the_fixture_holds_every_file_a_dev_question_points_into_at_the_right_lines():
    rows = [json.loads(x) for x in (ROOT / "eval" / "questions.jsonl").open(encoding="utf-8")]
    dev = [r for r in rows if r["split"] == "dev"]
    assert len(dev) == 40
    for row in dev:
        for span in row["gold"]:
            path = FIXTURE / span["path"]
            assert path.is_file(), f"{row['id']}: {span['path']} is not in the fixture"
            assert len(path.read_text(encoding="utf-8").splitlines()) >= span["end_line"]


def test_the_fixture_is_small_and_carries_the_license():
    files = [p for p in FIXTURE.rglob("*") if p.is_file()]
    assert (FIXTURE / "LICENSE-FASTAPI").is_file()
    assert sum(p.stat().st_size for p in files) < 1_500_000
    assert max(p.stat().st_size for p in files) < 500_000


def test_the_gate_passes_on_the_committed_code_and_fixture(capsys):
    assert gate.main([]) == 0
    out = capsys.readouterr().out
    assert "retrieval gate passed" in out and "40 dev questions" in out


def test_the_run_is_deterministic():
    questions = ROOT / "eval" / "questions.jsonl"
    assert gate.measure(FIXTURE, questions) == gate.measure(FIXTURE, questions)


def test_raising_a_threshold_above_the_measured_value_fails_the_gate(tmp_path, capsys):
    measured = gate.measure(FIXTURE, ROOT / "eval" / "questions.jsonl")
    too_high = {"bm25": {"hit@5": round(measured["bm25"]["hit@5"] + 0.025, 4)}}
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(too_high), encoding="utf-8")
    assert gate.main(["--thresholds", str(path)]) == 1
    assert "RETRIEVAL REGRESSION" in capsys.readouterr().err


def test_a_missing_corpus_or_thresholds_file_is_a_clean_error(tmp_path, capsys):
    assert gate.main(["--corpus", str(tmp_path / "nope")]) == 1
    assert gate.main(["--thresholds", str(tmp_path / "nope.json")]) == 1
    assert "error:" in capsys.readouterr().err


def test_update_prints_thresholds_and_never_fails(capsys):
    assert gate.main(["--update"]) == 0
    printed = capsys.readouterr().out
    assert json.loads(printed[printed.index("{") :])["bm25"]["hit@5"] > 0.5


@pytest.mark.skipif(not (ROOT / "data" / "raw" / "fastapi").is_dir(), reason="no corpus checkout")
def test_rebuilding_the_fixture_gives_the_same_files(tmp_path):
    maker = load("make_ci_fixture")
    assert maker.main(["--out", str(tmp_path / "fx")]) == 0
    mine = sorted(
        str(p.relative_to(tmp_path / "fx")) for p in (tmp_path / "fx").rglob("*") if p.is_file()
    )
    theirs = sorted(str(p.relative_to(FIXTURE)) for p in FIXTURE.rglob("*") if p.is_file())
    assert mine == theirs
