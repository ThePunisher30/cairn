"""Phase 1, step 6: candidate windows and the three baseline retrievers."""

import pytest

from cairn.evaluation.baselines import (
    FirstNRetriever,
    OracleRetriever,
    RandomRetriever,
    line_windows,
)
from cairn.types import Hit, Question, Span


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.md").write_text("".join(f"line {i}\n" for i in range(1, 101)), encoding="utf-8")
    body = "".join(f"x = {i}\n" for i in range(25))
    (tmp_path / "sub" / "b.py").write_text(body, encoding="utf-8")
    (tmp_path / "c.txt").write_text("ignored\n", encoding="utf-8")
    return tmp_path


def spans(hits):
    return [(h.path, h.start_line, h.end_line) for h in hits]


# ------------------------------------------------------------ line_windows
def test_line_windows_cover_every_line_once_in_sorted_order(corpus):
    assert spans(line_windows(corpus, window=40)) == [
        ("a.md", 1, 40),
        ("a.md", 41, 80),
        ("a.md", 81, 100),
        ("sub/b.py", 1, 25),
    ]


def test_line_windows_respects_suffixes(corpus):
    assert spans(line_windows(corpus, window=40, suffixes=(".py",))) == [("sub/b.py", 1, 25)]


def test_line_windows_returns_hits_with_forward_slash_paths(corpus):
    hits = line_windows(corpus, window=40)
    assert all(isinstance(h, Hit) for h in hits)
    assert all("\\" not in h.path for h in hits)


@pytest.mark.parametrize("window", [0, -5])
def test_line_windows_rejects_bad_window(corpus, window):
    with pytest.raises(ValueError):
        line_windows(corpus, window=window)


def test_empty_file_gives_no_windows(tmp_path):
    (tmp_path / "empty.md").write_text("", encoding="utf-8")
    assert line_windows(tmp_path) == []


def test_missing_final_newline_still_counts_the_last_line(tmp_path):
    (tmp_path / "a.md").write_text("x\ny", encoding="utf-8")
    assert spans(line_windows(tmp_path)) == [("a.md", 1, 2)]


def test_windows_line_endings_are_one_line_each(tmp_path):
    (tmp_path / "a.md").write_bytes(b"a\r\nb\r\n")
    assert spans(line_windows(tmp_path)) == [("a.md", 1, 2)]


def test_form_feed_does_not_start_a_new_line(tmp_path):
    (tmp_path / "a.md").write_text("a\n\x0cb\nc\n", encoding="utf-8")  # editors count 3 lines
    assert spans(line_windows(tmp_path)) == [("a.md", 1, 3)]


# --------------------------------------------------------------- baselines
@pytest.fixture
def pool():
    return [Hit(path=f"f{i}.md", start_line=1, end_line=10) for i in range(200)]


def test_random_retriever_returns_k_distinct_candidates(pool):
    hits = RandomRetriever(pool, seed=1).search("anything", 5)
    assert len(hits) == 5 and len(set(hits)) == 5
    assert all(h in pool for h in hits)


def test_random_retriever_same_seed_same_sequence(pool):
    a, b = RandomRetriever(pool, seed=7), RandomRetriever(pool, seed=7)
    assert a.search("q1", 5) == b.search("q1", 5)
    assert a.search("q2", 5) == b.search("q2", 5)


def test_random_retriever_different_seeds_differ(pool):
    first = RandomRetriever(pool, seed=1).search("q", 5)
    second = RandomRetriever(pool, seed=2).search("q", 5)
    assert first != second


def test_random_retriever_k_larger_than_pool_returns_everything(pool):
    assert sorted(spans(RandomRetriever(pool, seed=0).search("q", 1000))) == sorted(spans(pool))


def test_random_retriever_nonpositive_k_returns_nothing(pool):
    assert RandomRetriever(pool).search("q", 0) == []


def test_first_n_retriever_ignores_the_query(pool):
    r = FirstNRetriever(pool)
    assert r.search("one", 3) == pool[:3]
    assert r.search("something else", 3) == pool[:3]
    assert len(r.search("q", 1000)) == len(pool)
    assert r.search("q", 0) == []


@pytest.fixture
def questions():
    return [
        Question("q1", "code", "dev", "Where is X?", (Span("a.md", 3, 9), Span("b.py", 1, 5))),
        Question("q2", "lexical", "dev", "What is Y?", (Span("c.md", 10, 12),)),
    ]


def test_oracle_returns_the_gold_spans_for_a_known_question(questions):
    hits = OracleRetriever(questions).search("Where is X?", 10)
    assert spans(hits) == [("a.md", 3, 9), ("b.py", 1, 5)]
    assert all(h.score == 1.0 for h in hits)


def test_oracle_respects_k_and_unknown_questions(questions):
    r = OracleRetriever(questions)
    assert spans(r.search("Where is X?", 1)) == [("a.md", 3, 9)]
    assert r.search("never asked", 10) == []
