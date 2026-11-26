"""Run several rows one after another, each in its own process (Phase 11).

    python bench/frameworks/run_all.py cairn-bm25 cairn-dense li-default
    python bench/frameworks/run_all.py --all-dev

A row that fails is reported and the next one still runs. Use the framework environment's Python
(the same one for every row, so the library versions are the same).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rows import ROWS  # noqa: E402

ALL_DEV = ["cairn-bm25", "cairn-dense", "cairn-hybrid", "li-default", "li-default-noprefix"]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("rows", nargs="*")
    parser.add_argument("--all-dev", action="store_true", help="the first five rows")
    args = parser.parse_args(argv)
    names = ALL_DEV if args.all_dev else args.rows
    unknown = [n for n in names if n not in ROWS]
    if unknown:
        print(f"unknown rows: {unknown}", file=sys.stderr)
        return 1
    failed = []
    for name in names:
        began = time.perf_counter()
        print(f"=== {name}", flush=True)
        code = subprocess.call(
            [sys.executable, str(Path(__file__).parent / "run_row.py"), "--row", name]
        )
        print(f"=== {name} exit {code} in {time.perf_counter() - began:.0f}s", flush=True)
        if code:
            failed.append(name)
    print("FAILED: " + ", ".join(failed) if failed else "ALL DONE", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
