"""Phase 8, step 6: checks on YOUR file `eval/multihop_questions.jsonl` (15 multi-hop questions).

One row per question, one JSON object per line:

    {"id": "m01", "split": "dev", "question": "...",
     "hops": [{"path": "...", "start_line": 10, "end_line": 25}, {...}],
     "reference": "the answer in one or two sentences"}

A multi-hop question needs facts from at least two places, and the second lookup depends on the
first ("which function handles X, and what does the dependency Y it uses do?"). `hops` is one
span per place. Like `eval/questions.jsonl`, they validate whatever exists so you can run them while
you write, except `test_the_set_is_complete`, which wants all 15. Tests are SKIPPED when the file
or the downloaded corpus is missing.

Also yours to check, not testable here: that ONE search on the question text does not already bring
every hop. Run the Phase 6 retriever on each question and record, for each hop, whether a chunk
overlapping it is in the top 5. If most questions are solved by one search, they are not multi-hop
and the experiment cannot show anything.
"""

import json
from collections import Counter

import pytest

from cairn import config
from cairn.config import load_settings

PATH = config.PROJECT_ROOT / "eval" / "multihop_questions.jsonl"
CORPUS_ROOT = load_settings().data_dir / "raw" / "fastapi"
MAX_SPAN_LINES = 60


@pytest.fixture(scope="module")
def rows():
    if not PATH.exists():
        pytest.skip("eval/multihop_questions.jsonl does not exist yet")
    return [
        json.loads(line) for line in PATH.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


@pytest.fixture(scope="module")
def corpus():
    if not CORPUS_ROOT.is_dir():
        pytest.skip("corpus not downloaded: run scripts/fetch_corpus.py")
    return CORPUS_ROOT


def test_every_row_has_the_documented_fields(rows):
    for row in rows:
        assert set(row) == {"id", "split", "question", "hops", "reference"}, row.get("id")
        assert row["split"] in ("dev", "test")
        assert row["question"].strip().endswith("?"), row["id"]
        assert row["reference"].strip(), row["id"]


def test_ids_are_m01_style_and_unique(rows):
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids)
    assert all(i.startswith("m") and i[1:].isdigit() and len(i) == 3 for i in ids), ids


def test_no_question_is_repeated_and_none_is_in_the_single_hop_set(rows):
    texts = [" ".join(r["question"].lower().split()) for r in rows]
    assert len(set(texts)) == len(texts)
    single = config.PROJECT_ROOT / "eval" / "questions.jsonl"
    if single.exists():
        old = {
            " ".join(json.loads(line)["question"].lower().split())
            for line in single.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        assert not old & set(texts)


def test_every_question_has_at_least_two_hops_in_different_places(rows):
    for row in rows:
        hops = row["hops"]
        assert len(hops) >= 2, f"{row['id']} has {len(hops)} hop(s)"
        places = {(h["path"], h["start_line"], h["end_line"]) for h in hops}
        assert len(places) == len(hops), f"{row['id']} repeats a hop"
        for h in hops:
            assert set(h) == {"path", "start_line", "end_line"}


def test_hops_point_at_real_lines_and_are_small(rows, corpus):
    problems = []
    for row in rows:
        for h in row["hops"]:
            path = corpus / h["path"]
            if not path.is_file():
                problems.append(f"{row['id']}: {h['path']} does not exist")
                continue
            n_lines = len(path.read_text(encoding="utf-8").split("\n"))
            if not 1 <= h["start_line"] <= h["end_line"] <= n_lines + 1:
                problems.append(f"{row['id']}: {h['path']} lines {h['start_line']}-{h['end_line']}")
            elif h["end_line"] - h["start_line"] + 1 > MAX_SPAN_LINES:
                problems.append(f"{row['id']}: a hop longer than {MAX_SPAN_LINES} lines")
    assert not problems, "\n".join(problems)


def test_the_hops_of_a_question_are_in_different_files_or_far_apart(rows):
    for row in rows:
        spans = [(h["path"], h["start_line"], h["end_line"]) for h in row["hops"]]
        for i, a in enumerate(spans):
            for b in spans[i + 1 :]:
                same_file = a[0] == b[0]
                overlap = a[1] <= b[2] and b[1] <= a[2]
                assert not (same_file and overlap), f"{row['id']}: two hops overlap"


def test_the_set_is_complete(rows):
    assert len(rows) == 15, f"{len(rows)} of 15 questions so far"
    assert Counter(r["split"] for r in rows) == {"dev": 10, "test": 5}
    assert sorted(r["id"] for r in rows) == [f"m{i:02d}" for i in range(1, 16)]
