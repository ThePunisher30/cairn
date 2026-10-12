"""Score a retriever on the question set.

Usage:
    python scripts/run_eval.py --retriever oracle
    python scripts/run_eval.py --retriever random --seed 1
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from cairn import config
from cairn.config import load_settings
from cairn.evaluation.baselines import (
    FirstNRetriever,
    OracleRetriever,
    RandomRetriever,
    line_windows,
)
from cairn.evaluation.harness import format_report, run_retriever, summarize
from cairn.evaluation.questions import QuestionError, load_questions


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    default_questions = config.PROJECT_ROOT / "eval" / "questions.jsonl"
    default_test_log = config.PROJECT_ROOT / "eval" / "test_set_runs.log"
    parser = argparse.ArgumentParser(description="Score a retriever on the question set.")
    parser.add_argument("--questions", type=Path, default=default_questions)
    parser.add_argument("--corpus", type=Path, default=settings.data_dir / "raw" / "fastapi")
    parser.add_argument("--retriever", choices=["random", "first", "oracle"], default="random")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-test-set", action="store_true")
    parser.add_argument("--test-log", type=Path, default=default_test_log)
    args = parser.parse_args(argv)

    # TODO 1: refuse the test split unless --use-test-set was given
    if args.split == "test" and not args.use_test_set:
        parser.error("the test split is frozen: pass --use-test-set to spend a run on it")

    # TODO 2: load the questions (for args.split), turning a QuestionError into a clean error
    try:
        questions = load_questions(args.questions, split=args.split)
    except QuestionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    # TODO 3: build the retriever that args.retriever names
    if args.retriever == "oracle":
        retriever = OracleRetriever(questions)
    else:
        candidates = line_windows(args.corpus, window=args.window)
        if args.retriever == "random":
            retriever = RandomRetriever(candidates, seed=args.seed)
        else:
            retriever = FirstNRetriever(candidates)

    # TODO 4: if this is a test-split run, append one line to the log file
    if args.split == "test":
        with open(args.test_log, "a") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} retriever={args.retriever}\n")

    # TODO 5: run, summarise, print the report, return 0
    results = run_retriever(retriever, questions, k=args.k)
    summary = summarize(results)
    print(format_report(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
