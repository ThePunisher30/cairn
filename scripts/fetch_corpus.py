"""Download the corpus at the pinned tag.

Usage:
    python scripts/fetch_corpus.py
    python scripts/fetch_corpus.py --tag 0.143.0 --dest data/raw/fastapi
"""
import argparse
import sys
from pathlib import Path

from cairn.config import load_settings
from cairn.corpus import CorpusError, fetch_corpus


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()

    parser = argparse.ArgumentParser(description="Download the corpus at a pinned git tag.")
    parser.add_argument("--repo", default=settings.corpus_repo_url, help="git URL to clone from")
    parser.add_argument("--tag", default=settings.corpus_tag, help="tag to check out")
    parser.add_argument(
        "--dest",
        type=Path,
        default=settings.data_dir / "raw" / "fastapi",
        help="folder to clone into",
    )
    args = parser.parse_args(argv)

    try:
        path = fetch_corpus(args.repo, args.tag, args.dest)
    except CorpusError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
