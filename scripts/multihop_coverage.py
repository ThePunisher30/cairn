"""Check that the multi-hop questions are really multi-hop: does ONE search on the question text
already bring every hop?

For each dev question the retriever is asked once, with the question as the query. A hop counts
as found when a returned chunk overlaps its span (same file, line ranges that touch). A question
is "solved by one search" when every hop is found. If most are, they are not multi-hop and the
experiment cannot show anything.

Only the dev questions are looked at: the test ones of this set are kept unseen.

Usage:
    python scripts/multihop_coverage.py
    python scripts/multihop_coverage.py --retriever bm25 --k 5
"""

import argparse
import json
import sys

from cairn import config
from cairn.config import load_settings
from cairn.generation.factory import RETRIEVERS, build_retrievers


def overlaps(hit, hop):
    return (
        hit.path == hop["path"]
        and hit.start_line <= hop["end_line"]
        and hop["start_line"] <= hit.end_line
    )


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--retriever", choices=RETRIEVERS, default="hybrid")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--questions", default=config.PROJECT_ROOT / "eval" / "multihop_questions.jsonl"
    )
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in open(args.questions, encoding="utf-8") if line.strip()]
    rows = [r for r in rows if r["split"] == "dev"]
    retriever, _ = build_retrievers(
        args.retriever,
        chunks_path=settings.data_dir / "chunks.jsonl",
        bm25_path=settings.data_dir / "bm25.pkl",
        dense_path=settings.data_dir / "dense-bge-small.npz",
    )
    solved = found_hops = total_hops = 0
    for row in rows:
        hits = retriever.search(row["question"], args.k)
        flags = [any(overlaps(h, hop) for h in hits) for hop in row["hops"]]
        solved += all(flags)
        found_hops += sum(flags)
        total_hops += len(flags)
        marks = " ".join("found" if f else "MISSING" for f in flags)
        print(f"{row['id']}: {marks}")
    print(
        f"\n{args.retriever} top-{args.k}: {solved}/{len(rows)} questions have every hop found "
        f"by one search; {found_hops}/{total_hops} hops found"
    )


if __name__ == "__main__":
    main()
