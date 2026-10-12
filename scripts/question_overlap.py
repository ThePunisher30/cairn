"""Measure how much each question's wording is copied from its answer passage.

For every question: take its content words (lowercase, no stop words) and report the share that also
appears in the gold passage. A high share means the question repeats the passage's own words, which
flatters keyword search (BM25). `semantic` questions should score clearly LOWER than `lexical` ones.

Usage:
    python scripts/question_overlap.py
    python scripts/question_overlap.py --show 0.5      # list questions at or above this overlap
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

from cairn import config
from cairn.config import load_settings
from cairn.evaluation.questions import load_questions

STOP_WORDS = set(
    (
        "a an the and or of to in on for with by at as is are was were be been it its this that "
        "these those i my me we you your do does did how what which when where why who can could "
        "should would will if than then so not no from into out up over any each about between "
        "there their them they instead while also one two use used using"
    ).split()
)


def words(text: str) -> set[str]:
    found = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in found if w not in STOP_WORDS and len(w) > 1}


def passage(corpus: Path, path: str, start: int, end: int) -> str:
    with open(corpus / path, encoding="utf-8") as f:
        return "".join(line for number, line in enumerate(f, start=1) if start <= number <= end)


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Question/passage word overlap.")
    default_questions = config.PROJECT_ROOT / "eval" / "questions.jsonl"
    default_corpus = load_settings().data_dir / "raw" / "fastapi"
    parser.add_argument("--questions", type=Path, default=default_questions)
    parser.add_argument("--corpus", type=Path, default=default_corpus)
    parser.add_argument("--show", type=float, default=0.6, help="list questions at/above this")
    args = parser.parse_args(argv)

    scores: dict[str, list[float]] = defaultdict(list)
    rows = []
    for q in load_questions(args.questions):
        q_words = words(q.text)
        gold_words: set[str] = set()
        for g in q.gold:
            gold_words |= words(passage(args.corpus, g.path, g.start_line, g.end_line))
        overlap = len(q_words & gold_words) / len(q_words) if q_words else 0.0
        scores[q.category].append(overlap)
        rows.append((overlap, q.id, q.category, q.text))

    print(f"{'category':<12} {'n':>3} {'mean overlap':>13} {'min':>6} {'max':>6}")
    for category, values in scores.items():
        mean = sum(values) / len(values)
        low, high = min(values), max(values)
        print(f"{category:<12} {len(values):>3} {mean:>13.2f} {low:>6.2f} {high:>6.2f}")

    print(f"\nQuestions with overlap >= {args.show}:")
    for overlap, qid, category, text in sorted(rows, reverse=True):
        if overlap >= args.show:
            print(f"  {overlap:.2f}  {qid}  {category:<11} {text[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
