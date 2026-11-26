import json

from cairn.evaluation.freeze import ALLOWED_RUNS, freeze_problems, used_runs

ONE_RUN = "2026-10-31T02:46:31 retriever=ablate_hybrid\n"
ROWS = ["cairn-dense", "li-default"]


def frozen(rows=None, dev=True):
    return json.dumps({"rows": rows if rows is not None else ROWS, "dev": {"x": 1} if dev else {}})


def test_runs_are_counted_by_lines_not_blank_ones():
    assert used_runs("") == 0
    assert used_runs(ONE_RUN) == 1
    assert used_runs(ONE_RUN + "\n\n2026-11-30T10:00:00 phase11\n") == 2


def test_a_committed_freeze_and_one_earlier_run_allows_the_second_run():
    assert freeze_problems(frozen(), ONE_RUN, "", ROWS) == []
    assert ALLOWED_RUNS == 2


def test_the_order_of_the_rows_does_not_matter():
    assert freeze_problems(frozen(), ONE_RUN, "", list(reversed(ROWS))) == []


def test_a_third_run_is_refused():
    log = ONE_RUN + "2026-11-30T10:00:00 phase11\n"
    problems = freeze_problems(frozen(), log, "", ROWS)
    assert len(problems) == 1 and "already run 2 times" in problems[0]


def test_other_rows_than_the_frozen_ones_are_refused():
    problems = freeze_problems(frozen(), ONE_RUN, "", ["cairn-dense", "li-md-code-30"])
    assert len(problems) == 1 and "not the frozen rows" in problems[0]
    assert freeze_problems(frozen(), ONE_RUN, "", ["cairn-dense"])  # a subset is not the freeze


def test_a_freeze_that_is_modified_or_not_committed_is_refused():
    assert "not committed" in freeze_problems(frozen(), ONE_RUN, " M bench/frozen.json", ROWS)[0]
    assert "not committed" in freeze_problems(frozen(), ONE_RUN, "?? bench/frozen.json", ROWS)[0]


def test_a_freeze_without_dev_results_or_rows_is_refused():
    assert "no dev results" in freeze_problems(frozen(dev=False), ONE_RUN, "", ROWS)[0]
    assert "lists no rows" in freeze_problems(frozen(rows=[]), ONE_RUN, "", ROWS)[0]


def test_garbage_in_place_of_the_freeze_is_refused_not_a_crash():
    for text in ("", "not json", "null"):
        problems = freeze_problems(text, ONE_RUN, "", ROWS)
        assert problems and "frozen.json" in problems[0]


def test_every_problem_is_reported_at_once():
    problems = freeze_problems(frozen(dev=False), ONE_RUN * 2, " M f", ["other"])
    assert len(problems) == 4
