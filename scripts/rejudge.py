"""Grade the answers of an existing gen_eval run again, with another judge prompt (Phase 7).

    python scripts/rejudge.py --run data/gen_runs/gemma4-hybrid.jsonl --judge-version 2 \
        --llm ollama --model gemma4:e4b --out data/gen_runs/gemma4-hybrid-judge2.jsonl
    python scripts/rejudge.py --run ... --llm fake --limit 3        no model, to see it work

The answers and chunks are read from the run file, nothing is answered again. Only answerable,
not-abstained rows are judged (the same rule as gen_eval). The output has the same format as the
input, so `judge_agreement.py --run <out> --labels <labels>` compares it with a hand-graded
sample. Note: if the prompt was written after reading the disagreements on that same sample, the
agreement on it is optimistic.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from cairn.evaluation.generation import GenResult, Judge, JudgeParseError, summarize_generation
from cairn.generation.errors import LLMError
from cairn.generation.factory import LLMS, build_llm
from cairn.generation.prompts import format_chunks, load_prompt


def context_text(row):
    chunks = tuple(
        SimpleNamespace(
            number=c["number"],
            text=c["text"],
            hit=SimpleNamespace(path=c["path"], start_line=c["start_line"], end_line=c["end_line"]),
        )
        for c in row["context"]
    )
    return format_chunks(SimpleNamespace(chunks=chunks))


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Re-judge a gen_eval run with another prompt.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--judge-version", type=int, default=None)
    parser.add_argument("--llm", choices=LLMS, default="ollama")
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None, help="only the first N judged rows")
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in args.run.read_text("utf-8").splitlines() if line]
    llm, model = build_llm(args.llm, model=args.model)
    judge = Judge(llm, load_prompt("judge", args.judge_version))
    out = args.out or args.run.with_name(args.run.stem + "-rejudged.jsonl")
    done, changed, errors = 0, [], 0
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            new = dict(row)
            if row["answerable"] and not row["abstained"]:
                if args.limit is not None and done >= args.limit:
                    continue
                try:
                    verdict = judge.grade(
                        row["question"], row["reference"], row["answer"], context_text(row)
                    )
                except JudgeParseError:
                    errors += 1
                    verdict = None
                except LLMError as e:
                    print(f"error on {row['id']}: {e}", file=sys.stderr)
                    return 1
                done += 1
                if verdict is not None:
                    new["correct"], new["faithful"] = verdict.correct, verdict.faithful
                    new["judge_reason"] = verdict.reason
                    if (verdict.correct, verdict.faithful) != (row["correct"], row["faithful"]):
                        changed.append(row["id"])
                print(f"[{done}] {row['id']} judged", flush=True)
            f.write(json.dumps(new, ensure_ascii=False) + "\n")

    kept = [r for r in map(json.loads, out.read_text("utf-8").splitlines()) if r]
    results = [
        GenResult(
            r["id"], r["category"], r["answerable"], r["abstained"], r["correct"], r["faithful"],
            r["citation_precision"], r["citation_recall"], r["cost_usd"], r["latency_ms"],
        )
        for r in kept
    ]  # fmt: skip
    s = summarize_generation(results)
    print(f"\njudge {model}, prompt version {args.judge_version or 'latest'}")
    print(f"  correctness  {s['correctness']:.3f}   faithfulness {s['faithfulness']:.3f}")
    print(
        f"  unreadable judge replies: {errors}; verdict changed on {len(changed)} rows: {changed}"
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
