"""Build the small corpus the CI retrieval gate runs on (tests/fixtures/ci_corpus).

It copies, byte for byte and under the same relative paths (so the line numbers in the gold spans
stay true): every file that a dev question's gold span points into, plus a seeded sample of other
corpus files as distractors. The test split is not read.

    python scripts/make_ci_fixture.py     # needs data/raw/fastapi (scripts/fetch_corpus.py)

The files are from the FastAPI repository (MIT license); the license text is copied next to them.
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from cairn import config
from cairn.config import load_settings
from cairn.ingest.documents import load_documents


def main(argv=None):
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Build the CI fixture corpus.")
    parser.add_argument("--corpus", type=Path, default=settings.data_dir / "raw" / "fastapi")
    parser.add_argument(
        "--out", type=Path, default=config.PROJECT_ROOT / "tests/fixtures/ci_corpus"
    )
    parser.add_argument("--distractors", type=int, default=40)
    parser.add_argument("--max-distractor-bytes", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if not args.corpus.is_dir():
        print(f"error: {args.corpus} not found (run scripts/fetch_corpus.py)", file=sys.stderr)
        return 1
    questions = config.PROJECT_ROOT / "eval" / "questions.jsonl"
    rows = [json.loads(line) for line in questions.open(encoding="utf-8")]
    gold = sorted({g["path"] for r in rows if r["split"] == "dev" for g in r["gold"]})

    pool = sorted(
        d.path
        for d in load_documents(args.corpus)
        if d.path not in gold and len(d.text.encode("utf-8")) <= args.max_distractor_bytes
    )
    chosen = sorted(random.Random(args.seed).sample(pool, args.distractors))

    if args.out.exists():
        shutil.rmtree(args.out)
    for path in [*gold, *chosen]:
        target = args.out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.corpus / path, target)
    license_file = args.corpus / "LICENSE"
    if license_file.is_file():
        shutil.copyfile(license_file, args.out / "LICENSE-FASTAPI")
    total = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(f"{len(gold)} gold files + {len(chosen)} distractors, {total // 1024} KB, in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
