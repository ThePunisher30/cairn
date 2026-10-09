"""Print numbered lines from a corpus file, to check a gold span before you save it.

Usage:
    python scripts/show_span.py docs/en/docs/tutorial/first-steps.md 10 25
    python scripts/show_span.py fastapi/routing.py 100 140
The path is relative to the corpus folder (data/raw/fastapi).
"""
import argparse
import sys
from pathlib import Path

from cairn.config import load_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show numbered lines of a corpus file.")
    parser.add_argument("path", help="path inside the corpus folder, with forward slashes")
    parser.add_argument("start", type=int, help="first line to show (1-based)")
    parser.add_argument("end", type=int, help="last line to show (inclusive)")
    parser.add_argument("--corpus", type=Path, default=load_settings().data_dir / "raw" / "fastapi")
    args = parser.parse_args(argv)

    file = args.corpus / args.path
    if not file.is_file():
        print(f"error: no such file: {file}", file=sys.stderr)
        return 1

    with open(file, encoding="utf-8") as f:
        total = 0
        for number, line in enumerate(f, start=1):
            total = number
            if args.start <= number <= args.end:
                print(f"{number:5d}  {line.rstrip()}")
    if args.end > total:
        print(f"(note: the file has only {total} lines)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
