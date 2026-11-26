"""The rules for the last test-split run, as code (Phase 11).

The test split may be run exactly twice in the whole project. The first run is in
`eval/test_set_runs.log`. This module decides whether the second one may happen: the rows to run
must be exactly the ones frozen in `frozen.json`, that file must be committed and unchanged since,
and the log must show one earlier run, not two. It only reads text and a git status string, so it
is tested without a repository.
"""

import json

ALLOWED_RUNS = 2


def used_runs(log_text):
    """How many test-split runs the log records (one line each; blank lines do not count)."""
    return sum(1 for line in log_text.splitlines() if line.strip())


def freeze_problems(frozen_text, log_text, frozen_git_status, rows_requested):
    """Why the test run must not start, as a list of sentences (empty means it may).

    `frozen_git_status` is the output of `git status --porcelain -- <frozen file>`: empty when the
    file is committed and unchanged, non-empty when it is modified, new or untracked."""
    problems = []
    try:
        frozen = json.loads(frozen_text)
    except (ValueError, TypeError):
        return ["frozen.json is missing or is not valid JSON"]
    if not isinstance(frozen, dict):
        return ["frozen.json is not a JSON object"]
    rows = frozen.get("rows")
    if not isinstance(rows, list) or not rows:
        problems.append("frozen.json lists no rows")
    elif sorted(rows) != sorted(rows_requested):
        problems.append(
            "the rows to run are not the frozen rows: "
            f"frozen {sorted(rows)}, requested {sorted(rows_requested)}"
        )
    if not frozen.get("dev"):
        problems.append("frozen.json has no dev results recorded next to the rows")
    if frozen_git_status.strip():
        problems.append("frozen.json is not committed and unchanged: " + frozen_git_status.strip())
    runs = used_runs(log_text)
    if runs >= ALLOWED_RUNS:
        problems.append(f"the test split was already run {runs} times; {ALLOWED_RUNS} are allowed")
    return problems
