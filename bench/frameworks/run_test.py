"""The second and last run of the test split (Phase 11). Not to be run without the author's go.

    python bench/frameworks/run_test.py --check-only          # safe: is the run allowed?
    python bench/frameworks/run_test.py --the-author-said-go  # spends the last test run

It refuses unless the freeze is in order (`cairn.evaluation.freeze`): the rows are exactly the ones
in `bench/frameworks/frozen.json`, that file is committed and unchanged, and
`eval/test_set_runs.log` shows one earlier run. Then it writes the log line (so a crash cannot
give a free retry), runs every frozen row once in its own process, and prints dev and test side
by side. Use the framework environment's Python.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import RESULTS, ROOT  # noqa: E402

from cairn.evaluation.freeze import freeze_problems  # noqa: E402

FROZEN = Path(__file__).parent / "frozen.json"
LOG = ROOT / "eval" / "test_set_runs.log"


def git(*args):
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description="The final test-split run.")
    parser.add_argument("--the-author-said-go", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="say whether the run would be allowed; runs nothing",
    )
    args = parser.parse_args(argv)
    if not (args.the_author_said_go or args.check_only):
        parser.error("pass --the-author-said-go (after the author says go), or --check-only")

    frozen_text = FROZEN.read_text(encoding="utf-8") if FROZEN.is_file() else ""
    rows = json.loads(frozen_text)["rows"] if frozen_text else []
    status = git("status", "--porcelain", "--", str(FROZEN.relative_to(ROOT)))
    log_text = LOG.read_text(encoding="utf-8") if LOG.is_file() else ""
    problems = freeze_problems(frozen_text, log_text, status, rows)
    if problems:
        print("The test run is refused:", file=sys.stderr)
        for line in problems:
            print(f"  - {line}", file=sys.stderr)
        return 1

    if args.check_only:
        print(f"the freeze is in order: {len(rows)} rows would run as run 2 of 2 (nothing was run)")
        return 0

    commit = git("rev-parse", "HEAD")
    with open(LOG, "a", encoding="utf-8") as f:
        stamp = datetime.now().isoformat(timespec="seconds")
        f.write(f"{stamp} phase11 rows={len(rows)} commit={commit}\n")
    print(f"logged as run 2 of 2 at commit {commit[:10]}; running {len(rows)} rows", flush=True)

    env = {**os.environ, "CAIRN_FINAL_TEST_RUN": commit}
    failed = []
    for row in rows:
        code = subprocess.call(
            [sys.executable, str(Path(__file__).parent / "run_row.py"), "--row", row,
             "--split", "test"], env=env)  # fmt: skip
        if code:
            failed.append(row)

    dev = json.loads(frozen_text)["dev"]
    print(f"\n{'row':24} {'dev hit@5':>9} {'test hit@5':>10} {'dev mrr':>8} {'test mrr':>9}")
    for row in rows:
        path = RESULTS / f"test-{row}.json"
        if not path.is_file():
            print(f"{row:24} (no result)")
            continue
        test = json.loads(path.read_text(encoding="utf-8"))["summary"]["overall"]
        print(f"{row:24} {dev[row]['hit@5']:>9.3f} {test['hit@5']:>10.3f} "
              f"{dev[row]['mrr']:>8.3f} {test['mrr']:>9.3f}")  # fmt: skip
    if failed:
        print("FAILED rows (the run is still counted): " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
